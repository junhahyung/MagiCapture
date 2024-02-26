from typing import List, Union
import cv2
import torch
from PIL import Image
import numpy as np
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch.nn.functional as F

from transformers import (
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import StableDiffusionPipeline
from .lora import patch_pipe, tune_lora_scale, _text_lora_path, _ti_lora_path
from .loss_utils import get_face_feature_from_tensor
import os
import abc
import glob
import math

EXAMPLE_PROMPTS = [
    "<obj> swimming in a pool",
    "<obj> at a beach with a view of seashore",
    "<obj> in times square",
    "<obj> wearing sunglasses",
    "<obj> in a construction outfit",
    "<obj> playing with a ball",
    "<obj> wearing headphones",
    "<obj> oil painting ghibli inspired",
    "<obj> working on the laptop",
    "<obj> with mountains and sunset in background",
    "Painting of <obj> at a beach by artist claude monet",
    "<obj> digital painting 3d render geometric style",
    "A screaming <obj>",
    "A depressed <obj>",
    "A sleeping <obj>",
    "A sad <obj>",
    "A joyous <obj>",
    "A frowning <obj>",
    "A sculpture of <obj>",
    "<obj> near a pool",
    "<obj> at a beach with a view of seashore",
    "<obj> in a garden",
    "<obj> in grand canyon",
    "<obj> floating in ocean",
    "<obj> and an armchair",
    "A maple tree on the side of <obj>",
    "<obj> and an orange sofa",
    "<obj> with chocolate cake on it",
    "<obj> with a vase of rose flowers on it",
    "A digital illustration of <obj>",
    "Georgia O'Keeffe style <obj> painting",
    "A watercolor painting of <obj> on a beach",
]

EXAMPLE_PROMPTS = [
    "a photo of a <sks> person, with style <style1>",
    "a photo of a <sks> person in the style of <style1>",
]


def image_grid(_imgs, rows=None, cols=None):

    if rows is None and cols is None:
        rows = cols = math.ceil(len(_imgs) ** 0.5)

    if rows is None:
        rows = math.ceil(len(_imgs) / cols)
    if cols is None:
        cols = math.ceil(len(_imgs) / rows)

    w, h = _imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(_imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def text_img_alignment(img_embeds, text_embeds, target_img_embeds):
    # evaluation inspired from textual inversion paper
    # https://arxiv.org/abs/2208.01618

    # text alignment
    assert img_embeds.shape[0] == text_embeds.shape[0]
    text_img_sim = (img_embeds * text_embeds).sum(dim=-1) / (
        img_embeds.norm(dim=-1) * text_embeds.norm(dim=-1)
    )

    # image alignment
    img_embed_normalized = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

    avg_target_img_embed = (
        (target_img_embeds / target_img_embeds.norm(dim=-1, keepdim=True))
        .mean(dim=0)
        .unsqueeze(0)
        .repeat(img_embeds.shape[0], 1)
    )

    img_img_sim = (img_embed_normalized * avg_target_img_embed).sum(dim=-1)

    return {
        "text_alignment_avg": text_img_sim.mean().item(),
        "image_alignment_avg": img_img_sim.mean().item(),
        "text_alignment_all": text_img_sim.tolist(),
        "image_alignment_all": img_img_sim.tolist(),
    }


def prepare_clip_model_sets(eval_clip_id: str = "openai/clip-vit-large-patch14"):
    text_model = CLIPTextModelWithProjection.from_pretrained(eval_clip_id)
    tokenizer = CLIPTokenizer.from_pretrained(eval_clip_id)
    vis_model = CLIPVisionModelWithProjection.from_pretrained(eval_clip_id)
    processor = CLIPProcessor.from_pretrained(eval_clip_id)

    return text_model, tokenizer, vis_model, processor


def evaluate_pipe(
    pipe,
    target_images: List[Image.Image],
    class_token: str = "",
    learnt_token: str = "",
    guidance_scale: float = 5.0,
    seed=0,
    clip_model_sets=None,
    eval_clip_id: str = "openai/clip-vit-large-patch14",
    n_test: int = 10,
    n_step: int = 50,
    embs = None,
):

    if clip_model_sets is not None:
        text_model, tokenizer, vis_model, processor = clip_model_sets
    else:
        text_model, tokenizer, vis_model, processor = prepare_clip_model_sets(
            eval_clip_id
        )

    images = []
    img_embeds = []
    text_embeds = []
    id_embeds = []
    for seed in range(0, 5):
        for prompt in EXAMPLE_PROMPTS:
            #prompt = prompt.replace("<obj>", learnt_token)
            torch.manual_seed(seed)
            with torch.autocast("cuda"):
                img = pipe(
                    prompt, num_inference_steps=n_step, guidance_scale=guidance_scale, output_type="img"
                ).images
                
                #print('one image!!!!!!!!!!!!!!!!!!!!!!!!!!')
            images.append(img)
            image = torch.from_numpy(img[0]).to(dtype=torch.float16, device="cuda").unsqueeze(0)
            # image
            """
            inputs = processor(images=img, return_tensors="pt")
            img_embed = vis_model(**inputs).image_embeds
            img_embeds.append(img_embed)

            prompt = prompt.replace(learnt_token, class_token)
            # prompts
            inputs = tokenizer([prompt], padding=True, return_tensors="pt")
            outputs = text_model(**inputs)
            text_embed = outputs.text_embeds
            text_embeds.append(text_embed)
            """
            _, emb, _ = get_face_feature_from_tensor(image)
            id_embeds.append(emb)
    # target images
    #inputs = processor(images=target_images, return_tensors="pt")
    #target_img_embeds = vis_model(**inputs).image_embeds

    #img_embeds = torch.cat(img_embeds, dim=0)
    #text_embeds = torch.cat(text_embeds, dim=0)
    cnt = 0
    sum_emb = 0
    for i in embs:
        for j in id_embeds:
            if j is not None:
                cnt += 1
                sum_emb += torch.nn.functional.cosine_similarity(i, j).item()
    #return images, {"id_score_val": sum_emb / cnt}
    return {"id_score_val": sum_emb / cnt}


def visualize_progress(
    path_alls: Union[str, List[str]],
    prompt: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device="cuda:0",
    patch_unet=True,
    patch_text=True,
    patch_ti=True,
    unet_scale=1.0,
    text_sclae=1.0,
    num_inference_steps=50,
    guidance_scale=5.0,
    offset: int = 0,
    limit: int = 10,
    seed: int = 0,
):

    imgs = []
    if isinstance(path_alls, str):
        alls = list(set(glob.glob(path_alls)))

        alls.sort(key=os.path.getmtime)
    else:
        alls = path_alls

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)

    print(f"Found {len(alls)} checkpoints")
    for path in alls[offset:limit]:
        print(path)

        patch_pipe(
            pipe, path, patch_unet=patch_unet, patch_text=patch_text, patch_ti=patch_ti
        )

        tune_lora_scale(pipe.unet, unet_scale)
        tune_lora_scale(pipe.text_encoder, text_sclae)

        torch.manual_seed(seed)
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        imgs.append(image)

    return imgs


LOW_RESOURCE=False

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                #attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
                attn = self.forward(attn, is_cross, place_in_unet)

        self.cur_att_layer += 1
        #print(self.cur_att_layer, self.num_att_layers, self.num_uncond_att_layers, is_cross, place_in_unet, attn.shape)
        #print('~!~!')
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
            #print(key, attn.shape)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}



def register_attention_control(unet, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = hidden_states.shape
            h = self.heads
            q = self.to_q(hidden_states)
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            #q = self.reshape_heads_to_batch_dim(q)
            #k = self.reshape_heads_to_batch_dim(k)
            #v = self.reshape_heads_to_batch_dim(v)
            q = self.head_to_batch_dim(q) 
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, bsz: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(bsz, -1, res, res, item.shape[-1])
                out.append(cross_maps)
    out = torch.cat(out, dim=1)
    out = out.sum(1) / out.shape[1]
    return out


def get_attentions(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, bsz: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(bsz, -1, res, res, item.shape[-1])
                out.append(cross_maps)
    out = torch.cat(out, dim=1)
    return out

def get_attentions_v2(attention_store: AttentionStore, from_where: List[str], is_cross: bool, bsz: int):
    # 16 32
    out = []
    attention_maps = attention_store.get_average_attention()
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == 1024:
                res = 32
            elif item.shape[1] == 256:
                res = 16
            else:
                assert False
            cross_maps = item.reshape(-1, res, res, item.shape[-1])
            cross_maps = cross_maps.permute(0,3,1,2)
            cross_maps = F.interpolate(cross_maps, size=(64,64), mode='bilinear')
            cross_maps = cross_maps.permute(0,2,3,1).reshape(bsz, -1, 64, 64, item.shape[-1])
            out.append(cross_maps)
    out = torch.cat(out, dim=1)
    return out




def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img



from scipy.ndimage import filters
import matplotlib.pyplot as plt


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = (1-0.5*attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (0.5*attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(img, attn_map, filename, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    #plt.show()
    plt.savefig(filename)
    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.
