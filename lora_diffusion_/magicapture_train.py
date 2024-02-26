# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py,
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/cli_lora_pti.py

# Edited by Jaeyo Shin, Junha Hyung

import argparse
import hashlib
import inspect
import itertools
import math
import os, sys
import random
import re
import numpy as np
import torchvision
from pathlib import Path
from typing import Optional, List, Literal

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import wandb
import fire
print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+"/lora_diffusion_")
print(sys.path)

from lora_diffusion_ import (
    PivotalTuningDatasetCapation,
    PivotalTuningDatasetCapationTI,
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    inspect_lora,
    save_lora_weight,
    save_all,
    prepare_clip_model_sets,
    evaluate_pipe,
    UNET_EXTENDED_TARGET_REPLACE,
    AttentionStore,
    register_attention_control,
    aggregate_attention,
    get_attentions,
    text_under_image,
    view_images
)

from loss_utils import get_face_feature_from_tensor, get_face_feature_from_folder
import matplotlib.pyplot as plt
from PIL import Image

cosloss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def decode_latents(vae, latents, t=None, new=False):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.permute(0, 2, 3, 1).float()

    return image

def decode_latents_save(vae, latents, name="", t=None, new=False, mask=None):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    if mask is not None:
        print(mask.shape)
        image = image * mask
    image = image.permute(0, 2, 3, 1).float()
    
    return image


def img_save(image, name=""):
    numpy_image = image.detach().cpu().numpy()
    numpy_image = numpy_to_pil(numpy_image)[0]
    numpy_image.save(name)
    return image

def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    placeholder_tokens: List[str],
    placeholder_tokens_style: List[str],
    initializer_tokens: List[str],
    initializer_tokens_style: List[str],
    device="cuda:0",
):

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    placeholder_token_ids = []

    for token, init_tok in zip(placeholder_tokens, initializer_tokens):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids.append(placeholder_token_id)

        # Load models and create wrapper for stable diffusion

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):
            # <rand-"sigma">, e.g. <rand-0.5>
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    placeholder_token_style_ids = []

    for token, init_tok in zip(placeholder_tokens_style, initializer_tokens_style):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_style_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_style_ids.append(placeholder_token_style_id)

        # Load models and create wrapper for stable diffusion

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):
            # <rand-"sigma">, e.g. <rand-0.5>
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_style_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_style_id].mean().item():.3f} +- {token_embeds[placeholder_token_style_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_style_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_style_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_style_id] = token_embeds[initializer_token_id]


    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder=None if pretrained_vae_name_or_path else "vae",
        revision=None if pretrained_vae_name_or_path else revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )


    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_ids,
        placeholder_token_style_ids,
    )


@torch.no_grad()
def text2img_dataloader(
    train_dataset,
    train_batch_size,
    tokenizer,
    vae,
    text_encoder,
    cached_latents: bool = False,
):

    if cached_latents:
        cached_latents_dataset = []
        for idx in tqdm(range(len(train_dataset))):
            batch = train_dataset[idx]
            # rint(batch)
            latents = vae.encode(
                batch["instance_images"].unsqueeze(0).to(dtype=vae.dtype).to(vae.device)
            ).latent_dist.sample()
            latents = latents * 0.18215
            batch["instance_images"] = latents.squeeze(0)

            if batch.get("class_images", None) is not None:
                class_latents = vae.encode(
                    batch["class_images"].unsqueeze(0).to(dtype=vae.dtype).to(vae.device)
                ).latent_dist.sample()
                class_latents = class_latents * 0.18215
                batch["class_images"] = class_latents.squeeze(0)
            cached_latents_dataset.append(batch)

            


    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        
        types = [example["types"] for example in examples]
        face_embs = [example["face_emb"] for example in examples]
        if examples[0].get("class_prompt_ids", None) is not None:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            types += [example["class_types"] for example in examples]
            face_embs += [example["class_face_emb"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "types": types,
            "face_emb": face_embs
        }

        if examples[0].get("face_mask", None) is not None:
            batch["face_mask"] = torch.cat([example["face_mask"] for example in examples], dim=0)
        return batch

    if cached_latents:

        train_dataloader = torch.utils.data.DataLoader(
            cached_latents_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        print("PTI : Using cached latent.")

    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    return train_dataloader


def loss_step(
    batch,
    unet,
    vae,
    text_encoder,
    scheduler,
    train_inpainting=False,
    t_mutliplier=1.0,
    mixed_precision=False,
    mask_temperature=1.0,
    cached_latents: bool = False,
    lam_sim=0.5,
    lambda_style=2.0,
    cur_t=None,
    inv=False,
    placeholder_token_ids=None,
    placeholder_style_token_ids=None
):
    torch.autograd.set_detect_anomaly(True)
    controller = AttentionStore()
    register_attention_control(unet, controller)

    global save_cnt
    weight_dtype = torch.float32
    if not cached_latents:
        latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype).to(unet.device)
        ).latent_dist.sample()
        latents = latents * 0.18215

        if train_inpainting:
            masked_image_latents = vae.encode(
                batch["masked_image_values"].to(dtype=weight_dtype).to(unet.device)
            ).latent_dist.sample()
            masked_image_latents = masked_image_latents * 0.18215
            mask = F.interpolate(
                batch["mask_values"].to(dtype=weight_dtype).to(unet.device),
                scale_factor=1 / 8,
            )
    else:
        latents = batch["pixel_values"]

        if train_inpainting:
            masked_image_latents = batch["masked_image_latents"]
            mask = batch["mask_values"]

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    id_timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents.device,
    )
    id_timesteps = id_timesteps.long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    id_noisy_latents = scheduler.add_noise(latents, noise, id_timesteps)

    if train_inpainting:
        latent_model_input = torch.cat(
            [noisy_latents, mask, masked_image_latents], dim=1
        )
    else:
        latent_model_input = noisy_latents
        id_latent_model_input = id_noisy_latents

    if mixed_precision:
        with torch.cuda.amp.autocast():

            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(text_encoder.device)
            )[0]

            model_pred = unet(
                latent_model_input, timesteps, encoder_hidden_states
            ).sample
            # bsz,  (b*h, 16*16, 77(token))
            #attention_maps = aggregate_attention(controller, 16, ['up', 'down'], True, bsz) # (bsz, 16, 16, 77) #[start] [a] [photo] [of] [sks] 
            attention_maps = get_attentions(controller, 16, ['up', 'down'], True, bsz) # (bsz, 40, 16, 16, 77) #[start] [a] [photo] [of] [sks] 
            controller.reset()

            id_model_pred = unet(
                id_latent_model_input, id_timesteps, encoder_hidden_states
            ).sample
            # bsz,  (b*h, 16*16, 77(token))
            #attention_maps_id = aggregate_attention(controller, 16, ['up', 'down'], True, bsz) # (bsz, 16, 16, 77) #[start] [a] [photo] [of] [sks] 
            attention_maps_id = get_attentions(controller, 16, ['up', 'down'], True, bsz) # (bsz, 40, 16, 16, 77) #[start] [a] [photo] [of] [sks] 
            controller.reset()
    else:

        encoder_hidden_states = text_encoder(
            batch["input_ids"].to(text_encoder.device)
        )[0]

        model_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
        # bsz,  (b*h, 16*16, 77(token))
        #attention_maps = aggregate_attention(controller, 16, ['up', 'down'], True, bsz) # (bsz, 16, 16, 77) #[start] [a] [photo] [of] [sks] 
        attention_maps = get_attentions(controller, 16, ['up', 'down'], True, bsz) # (bsz, 40, 16, 16, 77) #[start] [a] [photo] [of] [sks] 
        controller.reset()

        id_model_pred = unet(id_latent_model_input, id_timesteps, encoder_hidden_states).sample
        # bsz,  (b*h, 16*16, 77(token))
        #attention_maps_id = aggregate_attention(controller, 16, ['up', 'down'], True, bsz) # (bsz, 16, 16, 77) #[start] [a] [photo] [of] [sks] 
        attention_maps_id = get_attentions(controller, 16, ['up', 'down'], True, bsz) # (bsz, 40, 16, 16, 77) #[start] [a] [photo] [of] [sks] 
        controller.reset()

    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")


    tgt = torch.tensor(1).to(device="cuda")
    sum_cos_loss = 0
    # predict z_0, by tweedie's formula
    # => decode it
    # => get embedding. If not - just recon loss. Exist - add cosine loss
    if inv is False:
        for n in range(bsz):
            if batch["types"][n] != "style":
                
                ts = id_timesteps[n]
                z_t = id_noisy_latents[n].unsqueeze(0)
                eps = id_model_pred[n].unsqueeze(0)
                alpha_cumprod = scheduler.alphas_cumprod[ts]
                z_0 = z_t / (alpha_cumprod) ** 0.5 - eps * ((1 - alpha_cumprod) / alpha_cumprod) ** 0.5
                x_0 = decode_latents(vae, z_0)
                _, emb, _ = get_face_feature_from_tensor(x_0)
                if inv is False and emb is not None:
                    cos_loss = cosloss(emb[0], batch["face_emb"][n][0], tgt)
                    sum_cos_loss += lam_sim * cos_loss
                
    else:
        z_0 = None
        sum_cos_loss = 0

    if "class" in batch["types"]: # prior preservation
        c = 2
    else:
        c = 1
    if batch.get("face_mask", None) is not None:
        
        mask = (
            batch["face_mask"][:, 0, :, :].unsqueeze(0)
            .to(model_pred.device)
            .reshape(
                model_pred.shape[0]//c, 1, model_pred.shape[2] * 8, model_pred.shape[3] * 8
            )
        )
        # resize to match model_pred
        mask = F.interpolate(
            mask.float(),
            size=model_pred.shape[-2:],
            mode="nearest",
        )

        mask = (mask + 0.01).pow(mask_temperature)

        mask = mask / mask.max()
        """
        mask = (
            batch["face_mask"]
            .to(model_pred.device)
        )
        """
        model_pred = model_pred * mask
        target = target * mask 

    loss = (
        F.mse_loss(model_pred.float(), target.float(), reduction="none")
        .mean([1, 2, 3])
    )

    for n in range(bsz):
        mask_area = mask[n].sum()
        all_area = mask.shape[3] * mask.shape[2] * mask.shape[1]
        loss = loss * all_area / mask_area # for exact mse loss
        if cur_t < 0.5 and (batch["types"][n] == "both" or batch["types"][n] == "style"):
            loss[n] = loss[n] * lambda_style # weight for style loss

    loss = loss.mean()

    attention_loss = 0
    for n in range(bsz):
        indexs = []
        #_ind_obj = []
        #_ind_style = []
        types = []
        ids = batch["input_ids"][n]
        ids = [int(id) for id in ids]

        for tok in placeholder_token_ids:
            if tok in ids:
                indexs.append(ids.index(tok))
                #_ind_obj.append(ids.index(tok))
                types.append('obj')

        for tok in placeholder_style_token_ids:
            if tok in ids:
                indexs.append(ids.index(tok))
                #_ind_style.append(ids.index(tok))
                types.append('style')

        maps = []
        for idx in indexs:
            _map = attention_maps[n:n+1,:,:,:,idx] # (1,40,16,16)
            _map_resized = F.interpolate(
                _map.float(),
                size=model_pred.shape[-2:],
                mode="bilinear"
            ) # (1,40,64,64)
            for i in range(len(_map_resized[0])):
                _map_resized[0][i] = _map_resized[0][i] - _map_resized[0][i].min().item()
                _map_resized[0][i] = _map_resized[0][i] / _map_resized[0][i].max().item()
            maps.append(_map_resized)
        maps = torch.cat(maps)
        
        if batch['types'][n] != 'both':
            assert len(indexs) == 1

            target_mask = mask[n:n+1] # (1,1,64,64)

            maps = torch.where(target_mask<0.5, maps, 1.)
        else:
            assert len(indexs) == 2

            target_mask = []

            for t in types:
                if t == 'obj':
                    target_mask.append((1-mask)[n:n+1])
                else: # 'style'
                    target_mask.append(mask[n:n+1])

            target_mask = torch.cat(target_mask, dim=0)
            maps[0] = torch.where(target_mask[0]<0.5, maps[0], 1.)
            maps[1] = torch.where(target_mask[1]<0.5, maps[1], 1.)
        
        # if cur_t % 100 == 0:
        #     maps_vis = maps.reshape(-1,1, 64,64)
        #     torchvision.utils.save_image(torchvision.utils.make_grid(maps_vis, nrow=8), f'./maps_vis/{cur_t}_mapvis.png')
        #     torchvision.utils.save_image(torchvision.utils.make_grid(target_mask, nrow=1), f'./maps_vis/{cur_t}_target.png')

        attention_loss += F.mse_loss(maps.mean(dim=1, keepdim=True), target_mask, reduction="none").mean([1,2,3]).mean()

    if sum_cos_loss != 0:
       loss = loss + sum_cos_loss
       sum_cos_loss = sum_cos_loss.clone().detach()
    attention_maps = [attention_maps, attention_maps_id]

    if batch["types"][n] == "style":
        return loss, None, None, attention_maps, attention_loss, target_mask
    return loss, z_0, sum_cos_loss, attention_maps, attention_loss, target_mask


def train_inversion(
    unet,
    vae,
    text_encoder,
    dataloader,
    num_steps: int,
    scheduler,
    index_no_updates,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path: str,
    tokenizer,
    lr_scheduler,
    test_image_path: str,
    cached_latents: bool,
    accum_iter: int = 1,
    log_wandb: bool = True,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False,
    mixed_precision: bool = False,
    clip_ti_decay: bool = True,
    placeholder_style_tokens = None,
    placeholder_style_token_ids = None,
    lambda_style = 2.0,
):

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0
    stop_flag = False

    # Original Emb for TI
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        preped_clip = prepare_clip_model_sets()

    index_updates = ~index_no_updates
    loss_sum = 0.0
    embs = []
    for epoch in range(math.ceil(num_steps / len(dataloader))):
        unet.eval()
        text_encoder.train()
        for batch in dataloader:
            
            if batch["types"][0] == "instance" and len(embs) < 5:
                embs.append(batch["face_emb"][0][0])

            if "both" in batch["types"]:
                continue # actually, "both" is not in batch["types"] at inversion process
            lr_scheduler.step()
           
            with torch.set_grad_enabled(True):
                loss, _, _, attn_maps, attn_loss, tgt_mask = (
                    loss_step(
                        batch,
                        unet,
                        vae,
                        text_encoder,
                        scheduler,
                        train_inpainting=train_inpainting,
                        mixed_precision=mixed_precision,
                        cached_latents=cached_latents,
                        cur_t=global_step / num_steps,
                        inv=True,
                        placeholder_token_ids=placeholder_token_ids,
                        placeholder_style_token_ids=placeholder_style_token_ids,
                        lambda_style=lambda_style,
                    )
                )
                attn_loss *= 0
                loss_recon = loss.detach().item()
                loss = loss + attn_loss
                loss /= accum_iter
                loss.backward()
                loss_sum += loss.detach().item()

                if log_wandb is True and global_step % 10 == 0:
                    wandb.log({"attn loss": attn_loss.detach().item()}, step=global_step)
                    wandb.log({"recon loss": loss_recon}, step=global_step)
                    wandb.log({"total loss": loss.detach().item()}, step=global_step)

                if global_step % accum_iter == 0:
                    # print gradient of text encoder embedding
                    print(
                        text_encoder.get_input_embeddings()
                        .weight.grad[index_updates, :]
                        .norm(dim=-1)
                        .mean()
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    with torch.no_grad():

                        # normalize embeddings
                        if clip_ti_decay:
                            pre_norm = (
                                text_encoder.get_input_embeddings()
                                .weight[index_updates, :]
                                .norm(dim=-1, keepdim=True)
                            )

                            lambda_ = min(1.0, 100 * lr_scheduler.get_last_lr()[0])
                            text_encoder.get_input_embeddings().weight[
                                index_updates
                            ] = F.normalize(
                                text_encoder.get_input_embeddings().weight[
                                    index_updates, :
                                ],
                                dim=-1,
                            ) * (
                                pre_norm + lambda_ * (0.4 - pre_norm)
                            )
                            print(pre_norm)

                        current_norm = (
                            text_encoder.get_input_embeddings()
                            .weight[index_updates, :]
                            .norm(dim=-1)
                        )

                        text_encoder.get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                        print(f"Current Norm : {current_norm}")
                        print("Types: " + batch["types"][0])

                global_step += 1
                progress_bar.update(1)

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if log_wandb is True and global_step % 50 == 0:
                print('visualize images')
                images = []
                for b in range(len(batch['input_ids'])):
                    for i in range(len(batch['input_ids'][b])):
                        if int(batch['input_ids'][b][i]) != 49407:
                            tok = tokenizer.decode(int(batch['input_ids'][b][i]))
                            image = attn_maps[0][b, :, :, :, i].detach().cpu()
                            image = image.mean(dim=0) # (64, 64)
                            image = 255 * image / image.max()
                            image = image.unsqueeze(-1).expand(*image.shape, 3)
                            image = image.numpy().astype(np.uint8)
                            image = np.array(Image.fromarray(image).resize((256, 256)))
                            image = text_under_image(image, tok)
                            images.append(image)

                for b in range(len(tgt_mask)):
                    image = tgt_mask[b].detach().cpu()[0]
                    image = 255 * image / image.max()
                    image = image.unsqueeze(-1).expand(*image.shape, 3)
                    image = image.numpy().astype(np.uint8)
                    image = np.array(Image.fromarray(image).resize((256, 256)))
                    image = text_under_image(image, 'tgt mask')
                    images.append(image)
                
                pil_attn_image = view_images(images, num_rows=len(batch['input_ids']))
                wandb_attn_image = wandb.Image(pil_attn_image)
                wandb.log({'TI': wandb_attn_image}, step=global_step)



            ###
            if global_step % save_steps == 0:
                save_all(
                    unet=unet,
                    text_encoder=text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(
                        save_path, f"step_inv_{global_step}.safetensors"
                    ),
                    save_lora=False,
                )
                if log_wandb:
                    with torch.no_grad():
                        pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                        )

                        # open all images in test_image_path
                        images = []
                        for file in os.listdir(test_image_path):
                            if (
                                file.lower().endswith(".png")
                                or file.lower().endswith(".jpg")
                                or file.lower().endswith(".jpeg")
                            ):
                                images.append(
                                    Image.open(os.path.join(test_image_path, file))
                                )


                        loss_sum = 0.0
                        id_dict = evaluate_pipe(
                            pipe,
                            target_images=images,
                            class_token=class_token,
                            learnt_token="".join(placeholder_tokens),
                            n_test=wandb_log_prompt_cnt,
                            n_step=50,
                            clip_model_sets=preped_clip,
                            embs = embs
                        )
                        """
                        pil_attn_image_val = view_images(images_val, num_rows=len(images))
                        wandb_attn_image_val = wandb.Image(pil_attn_image_val)
                        wandb.log({'Val_images': wandb_attn_image_val}, step=global_step)
                        """
                        wandb.log(
                            id_dict,
                            step=global_step
                        )

            if global_step >= num_steps:
                stop_flag = True
                return
        if stop_flag:
            break


def perform_tuning(
    unet,
    vae,
    text_encoder,
    dataloader,
    num_steps,
    scheduler,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path,
    lr_scheduler_lora,
    lora_unet_target_modules,
    lora_clip_target_modules,
    mask_temperature,
    out_name: str,
    tokenizer,
    test_image_path: str,
    cached_latents: bool,
    placeholder_style_tokens = "",
    placeholder_style_token_ids = "",
    log_wandb: bool = True,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False,
    accum_iter: int = 4,
    max_train_steps_ti=None,
    lambda_arc: float = 0.5,
    lambda_style: float = 2.0,
):

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0
    if max_train_steps_ti is not None:
        global_step += max_train_steps_ti
        num_steps += max_train_steps_ti
    else:
        max_train_steps_ti = 0
    save_flag = False
    stop_flag = False
    weight_dtype = torch.float16

    unet.train()
    text_encoder.train()

    if log_wandb:
        preped_clip = prepare_clip_model_sets()
    embs = []
    loss_sum = 0.0
    #optimizer.zero_grad()
    for epoch in range(math.ceil(num_steps / len(dataloader))):
        for batch in dataloader:
            if batch["types"][0] == "instance" and len(embs) < 5:
                embs.append(batch["face_emb"][0][0])

            if global_step % 50 == 0:
                save_flag = False
            lr_scheduler_lora.step()

            optimizer.zero_grad()

            loss, z_0, id_loss, attn_maps, attn_loss, tgt_mask = loss_step(
                batch,
                unet,
                vae,
                text_encoder,
                scheduler,
                train_inpainting=train_inpainting,
                t_mutliplier=0.8,
                mixed_precision=True,
                mask_temperature=mask_temperature,
                cached_latents=cached_latents,
                cur_t=(global_step - max_train_steps_ti) / (num_steps - max_train_steps_ti),
                placeholder_token_ids=placeholder_token_ids,
                placeholder_style_token_ids=placeholder_style_token_ids,
                lam_sim=lambda_arc,
                lambda_style=lambda_style,
            )

            attn_loss *= 2.5
            loss_recon = loss.detach().item()
            loss = loss + attn_loss
            if id_loss != None and id_loss != 0:
                loss = loss + id_loss
            loss.backward()
            loss_sum += loss.detach().item()

            if log_wandb is True and global_step % 10 == 0:
                wandb.log({"attn loss": attn_loss.detach().item()}, step=global_step)
                wandb.log({"recon loss": loss_recon}, step=global_step)
                if id_loss != None and id_loss != 0:
                    wandb.log({"id loss": id_loss.detach().item()}, step=global_step)
                wandb.log({"total loss": loss.detach().item()}, step=global_step)
                
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(unet.parameters(), text_encoder.parameters()), 1.0
            )

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            global_step += 1

            del z_0


            if log_wandb is True and global_step % 100 == 0:
                print('visualize images')
                images = []
                for b in range(len(batch['input_ids'])):
                    for i in range(len(batch['input_ids'][b])):
                        if int(batch['input_ids'][b][i]) != 49407:
                            tok = tokenizer.decode(int(batch['input_ids'][b][i]))
                            image = attn_maps[0][b, :, :, :, i].detach().cpu()
                            image = image.mean(dim=0) # (64, 64)
                            image = 255 * image / image.max()
                            image = image.unsqueeze(-1).expand(*image.shape, 3)
                            image = image.numpy().astype(np.uint8)
                            image = np.array(Image.fromarray(image).resize((256, 256)))
                            image = text_under_image(image, tok)
                            images.append(image)
                
                for b in range(len(tgt_mask)):
                    image = tgt_mask[b].detach().cpu()[0]
                    image = 255 * image / image.max()
                    image = image.unsqueeze(-1).expand(*image.shape, 3)
                    image = image.numpy().astype(np.uint8)
                    image = np.array(Image.fromarray(image).resize((256, 256)))
                    image = text_under_image(image, 'tgt mask')
                    images.append(image)

                pil_attn_image = view_images(images, num_rows=len(batch['input_ids']))
                wandb_attn_image = wandb.Image(pil_attn_image)

                wandb.log({'LoRA': wandb_attn_image}, step=global_step)


            if global_step % save_steps == 0:
                save_all(
                    unet,
                    text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    placeholder_style_tokens = placeholder_style_tokens,
                    placeholder_style_token_ids = placeholder_style_token_ids,
                    save_path=os.path.join(
                        save_path, f"step_{global_step}.safetensors"
                    ),
                    target_replace_module_text=lora_clip_target_modules,
                    target_replace_module_unet=lora_unet_target_modules,
                )
                moved = (
                    torch.tensor(list(itertools.chain(*inspect_lora(unet).values())))
                    .mean()
                    .item()
                )

                print("LORA Unet Moved", moved)
                moved = (
                    torch.tensor(
                        list(itertools.chain(*inspect_lora(text_encoder).values()))
                    )
                    .mean()
                    .item()
                )

                print("LORA CLIP Moved", moved)

                
                if log_wandb:
                    with torch.no_grad():
                        controller = AttentionStore()
                        register_attention_control(unet, controller)
                        pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                        )

                        # open all images in test_image_path
                        images = []
                        for file in os.listdir(test_image_path):
                            if file.endswith(".png") or file.endswith(".jpg"):
                                images.append(
                                    Image.open(os.path.join(test_image_path, file))
                                )

                        loss_sum = 0.0
                        wandb.log(
                            evaluate_pipe(
                                pipe,
                                target_images=images,
                                class_token=class_token,
                                learnt_token="".join(placeholder_tokens),
                                n_test=wandb_log_prompt_cnt,
                                n_step=50,
                                clip_model_sets=preped_clip,
                                embs = embs
                            ),
                            step=global_step
                        )

            if global_step >= num_steps:
                stop_flag = True
                break
        if stop_flag:
            break

    save_all(
        unet,
        text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        placeholder_style_tokens = placeholder_style_tokens,
        placeholder_style_token_ids = placeholder_style_token_ids,
        save_path=os.path.join(save_path, f"{out_name}.safetensors"),
        target_replace_module_text=lora_clip_target_modules,
        target_replace_module_unet=lora_unet_target_modules,
    )


def train(
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    class_data_dir: str = None,
    class_prompt: str = None,
    num_class_images: int = None,
    style_data_dir: str = None,
    train_text_encoder: bool = True,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    perform_inversion: bool = True,
    use_template: Literal[None, "object", "style", "person"] = "object",
    train_inpainting: bool = False,
    placeholder_tokens: str = "",
    placeholder_tokens_style: str = "",
    placeholder_token_at_data: Optional[str] = None,
    initializer_tokens: Optional[str] = None,
    initializer_tokens_style: Optional[str] = None,
    seed: int = 42,
    resolution: int = 512,
    color_jitter: bool = True,
    train_batch_size: int = 1,
    sample_batch_size: int = 1,
    max_train_steps_tuning: int = 1000,
    max_train_steps_ti: int = 1000,
    save_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = False,
    lora_rank: int = 4,
    lora_unet_target_modules={"CrossAttention", "Attention", "GEGLU"},
    lora_clip_target_modules={"CLIPAttention"},
    lora_dropout_p: float = 0.0,
    lora_scale: float = 1.0,
    use_extended_lora: bool = False,
    clip_ti_decay: bool = True,
    learning_rate_unet: float = 1e-4,
    learning_rate_text: float = 1e-5,
    learning_rate_ti: float = 5e-4,
    continue_inversion: bool = False,
    continue_inversion_lr: Optional[float] = None,
    use_face_segmentation_condition: bool = False,
    cached_latents: bool = True,
    use_mask_captioned_data: bool = False,
    mask_temperature: float = 1.0,
    scale_lr: bool = False,
    lr_scheduler: str = "linear",
    lr_warmup_steps: int = 0,
    lr_scheduler_lora: str = "linear",
    lr_warmup_steps_lora: int = 0,
    weight_decay_ti: float = 0.00,
    weight_decay_lora: float = 0.001,
    use_8bit_adam: bool = False,
    device="cuda:0",
    extra_args: Optional[dict] = None,
    log_wandb: bool = True,
    wandb_log_prompt_cnt: int = 10,
    wandb_project_name: str = "lora",
    wandb_entity: str = "junha",
    proxy_token: str = "person",
    enable_xformers_memory_efficient_attention: bool = False,
    out_name: str = "final_lora",
    with_prior_preservation: bool = False,
    lambda_arc: float = 0.5,
    lambda_style: float = 2.0,
):
    torch.manual_seed(seed)

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=output_dir,
            reinit=True,
            config={
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    # print(placeholder_tokens, initializer_tokens)
    if len(placeholder_tokens) == 0:
        placeholder_tokens = []
        print("PTI : Placeholder Tokens not given, using null token")
    else:
        placeholder_tokens = placeholder_tokens.split("|")

        assert (
            sorted(placeholder_tokens) == placeholder_tokens
        ), f"Placeholder tokens should be sorted. Use something like {'|'.join(sorted(placeholder_tokens))}'"

    # style_token
    if len(placeholder_tokens_style) == 0:
        placeholder_tokens_style = []
        print("PTI : Placeholder style Tokens not given, using null token")
    else:
        placeholder_tokens_style = placeholder_tokens_style.split("|")

        assert (
            sorted(placeholder_tokens_style) == placeholder_tokens_style
        ), f"Placeholder tokens should be sorted. Use something like {'|'.join(sorted(placeholder_tokens_style))}'"

    if initializer_tokens is None:
        print("PTI : Initializer Tokens not given, doing random inits")
        initializer_tokens = ["<rand-0.017>"] * len(placeholder_tokens)
    else:
        initializer_tokens = initializer_tokens.split("|")

    if initializer_tokens_style is None:
        print("PTI : Style Initializer Tokens not given, doing random inits")
        initializer_tokens_style = ["<rand-0.017>"] * len(placeholder_tokens_style)
    else:
        initializer_tokens_style = initializer_tokens_style.split("|")


    assert len(initializer_tokens) == len(
        placeholder_tokens
    ), "Unequal Initializer token for Placeholder tokens."

    if proxy_token is not None:
        class_token = proxy_token
    class_token = "".join(initializer_tokens)

    if placeholder_token_at_data is not None:
        tok, pat = placeholder_token_at_data.split("|")
        token_map = {tok: pat}

    else:
        token_map = {"DUMMY": "".join(placeholder_tokens), "DUMMY_style": "".join(placeholder_tokens_style)}


    print("PTI : Placeholder Tokens", placeholder_tokens)
    if placeholder_tokens_style != "":
        print("PTI : Placeholder style Tokens", placeholder_tokens_style)
    print("PTI : Initializer Tokens", initializer_tokens)
    print("PTI : Style Initializer Tokens", initializer_tokens_style)

    # get the models
    text_encoder, vae, unet, tokenizer, placeholder_token_ids, placeholder_token_style_ids = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_tokens,
        placeholder_tokens_style,
        initializer_tokens,
        initializer_tokens_style,
        device=device,
    )

    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if scale_lr:
        unet_lr = learning_rate_unet * gradient_accumulation_steps * train_batch_size
        text_encoder_lr = (
            learning_rate_text * gradient_accumulation_steps * train_batch_size
        )
        ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size
    else:
        unet_lr = learning_rate_unet
        text_encoder_lr = learning_rate_text
        ti_lr = learning_rate_ti

    if with_prior_preservation is True:
        train_dataset = PivotalTuningDatasetCapation(
            instance_data_root=instance_data_dir,
            token_map=token_map,
            style_data_root=style_data_dir,
            use_template=use_template,
            tokenizer=tokenizer,
            size=resolution,
            color_jitter=color_jitter,
            use_face_segmentation_condition=use_face_segmentation_condition,
            class_data_root=class_data_dir,
            class_prompt=class_prompt,
        )

        train_dataset_ti = PivotalTuningDatasetCapationTI(
            instance_data_root=instance_data_dir,
            token_map=token_map,
            style_data_root=style_data_dir,
            use_template=use_template,
            tokenizer=tokenizer,
            size=resolution,
            color_jitter=color_jitter,
            use_face_segmentation_condition=use_face_segmentation_condition,
            class_data_root=class_data_dir,
            class_prompt=class_prompt,
        )
    else:
        train_dataset = PivotalTuningDatasetCapation(
            instance_data_root=instance_data_dir,
            token_map=token_map,
            style_data_root=style_data_dir,
            use_template=use_template,
            tokenizer=tokenizer,
            size=resolution,
            color_jitter=color_jitter,
            use_face_segmentation_condition=use_face_segmentation_condition,
        )
        train_dataset_ti = PivotalTuningDatasetCapationTI(
            instance_data_root=instance_data_dir,
            token_map=token_map,
            style_data_root=style_data_dir,
            use_template=use_template,
            tokenizer=tokenizer,
            size=resolution,
            color_jitter=color_jitter,
            use_face_segmentation_condition=use_face_segmentation_condition,
        )

    train_dataset.blur_amount = 200

    train_dataloader = text2img_dataloader(
        train_dataset,
        train_batch_size,
        tokenizer,
        vae,
        text_encoder,
        cached_latents=cached_latents,
    )
    train_dataloader_ti = text2img_dataloader(
        train_dataset_ti,
        train_batch_size,
        tokenizer,
        vae,
        text_encoder,
        cached_latents=cached_latents,
    )

    index_no_updates = torch.arange(len(tokenizer)) != -1

    for tok_id in placeholder_token_ids:
        index_no_updates[tok_id] = False
    
    for tok_id in placeholder_token_style_ids:
        index_no_updates[tok_id] = False

    unet.requires_grad_(False)
    vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    #if cached_latents:
    #    vae = None
    # STEP 1 : Perform Inversion
    if perform_inversion:
        ti_optimizer = optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),
            lr=ti_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay_ti,
        )

        lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=ti_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps_ti,
        )

        train_inversion(
            unet,
            vae,
            text_encoder,
            train_dataloader_ti,
            max_train_steps_ti,
            cached_latents=cached_latents,
            #accum_iter=gradient_accumulation_steps,
            scheduler=noise_scheduler,
            index_no_updates=index_no_updates,
            optimizer=ti_optimizer,
            lr_scheduler=lr_scheduler,
            save_steps=save_steps,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_ids=placeholder_token_ids,
            placeholder_style_tokens = placeholder_tokens_style,
            placeholder_style_token_ids = placeholder_token_style_ids,
            save_path=output_dir,
            test_image_path=instance_data_dir,
            log_wandb=log_wandb,
            wandb_log_prompt_cnt=wandb_log_prompt_cnt,
            class_token=class_token,
            train_inpainting=train_inpainting,
            mixed_precision=False,
            tokenizer=tokenizer,
            clip_ti_decay=clip_ti_decay,
            lambda_style=lambda_style,
        )

        del ti_optimizer

    # Next perform Tuning with LoRA:
    if not use_extended_lora:
        unet_lora_params, _ = inject_trainable_lora(
            unet,
            r=lora_rank,
            target_replace_module=lora_unet_target_modules,
            dropout_p=lora_dropout_p,
            scale=lora_scale,
        )
    else:
        print("PTI : USING EXTENDED UNET!!!")
        lora_unet_target_modules = (
            lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
        )
        print("PTI : Will replace modules: ", lora_unet_target_modules)

        unet_lora_params, _ = inject_trainable_lora_extended(
            unet, r=lora_rank, target_replace_module=lora_unet_target_modules
        )
    print(f"PTI : has {len(unet_lora_params)} lora")

    print("PTI : Before training:")
    inspect_lora(unet)

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
    ]

    text_encoder.requires_grad_(False)

    if continue_inversion:
        params_to_optimize += [
            {
                "params": text_encoder.get_input_embeddings().parameters(),
                "lr": continue_inversion_lr
                if continue_inversion_lr is not None
                else ti_lr,
            }
        ]
        text_encoder.requires_grad_(True)
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        for param in params_to_freeze:
            param.requires_grad = False
    else:
        text_encoder.requires_grad_(False)
    if train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            text_encoder,
            target_replace_module=lora_clip_target_modules,
            r=lora_rank,
        )
        params_to_optimize += [
            {
                "params": itertools.chain(*text_encoder_lora_params),
                "lr": text_encoder_lr,
            }
        ]
        inspect_lora(text_encoder)

    lora_optimizers = optim.AdamW(params_to_optimize, weight_decay=weight_decay_lora)

    unet.train()
    if train_text_encoder:
        text_encoder.train()

    train_dataset.blur_amount = 70

    lr_scheduler_lora = get_scheduler(
        lr_scheduler_lora,
        optimizer=lora_optimizers,
        num_warmup_steps=lr_warmup_steps_lora,
        num_training_steps=max_train_steps_tuning,
    )

    perform_tuning(
        unet,
        vae,
        text_encoder,
        train_dataloader,
        max_train_steps_tuning,
        cached_latents=cached_latents,
        scheduler=noise_scheduler,
        optimizer=lora_optimizers,
        save_steps=save_steps,
        placeholder_tokens=placeholder_tokens,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_style_tokens = placeholder_tokens_style,
        placeholder_style_token_ids = placeholder_token_style_ids,
        save_path=output_dir,
        lr_scheduler_lora=lr_scheduler_lora,
        lora_unet_target_modules=lora_unet_target_modules,
        lora_clip_target_modules=lora_clip_target_modules,
        mask_temperature=mask_temperature,
        tokenizer=tokenizer,
        out_name=out_name,
        test_image_path=instance_data_dir,
        log_wandb=log_wandb,
        wandb_log_prompt_cnt=wandb_log_prompt_cnt,
        class_token=class_token,
        train_inpainting=train_inpainting,
        max_train_steps_ti=max_train_steps_ti,
        lambda_arc=lambda_arc,
        lambda_style=lambda_style,
    )


def main():
    fire.Fire(train)

if __name__ == "__main__":
    main()

