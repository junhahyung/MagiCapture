# Bootstrapped from:
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/dataset.py
# Edited by Jaeyo Shin, Junha Hyung

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from .loss_utils import get_face_feature_from_tensor, get_face_feature_from_folder
import torch
from face_parsing_PyTorch.test import evaluate
from face_parsing_PyTorch.model import BiSeNet
import os.path as osp
from segment_anything import SamPredictor, sam_model_registry
from transformers import ViTImageProcessor, ViTModel

OBJECT_TEMPLATE = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

OBJECT_TEMPLATE2 = [
    "a photo of a {} person",
    "the photo of a {} person",
    "a photo of a clean {} person",
    "a bright photo of the {} person",
    "a photo of the {} person",
    "a good photo of the {} person",
    "a photo of one {} person",
    "a photo of a nice {} person",
    "a good photo of a {} person",
    "a photo of the nice {} person",
]

STYLE_TEMPLATE = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

STYLE_TEMPLATE2 = [
    "a painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a bright painting in the style of {}",
    "a good painting in the style of {}",
    "a nice painting in the style of {}",
]

STYLE_PERSON_TEMPLATE = [
    "a painting of a person in the style of {}",
    "a rendering of a person in the style of {}",
    "a cropped painting of a person in the style of {}",
    "the painting of a person in the style of {}",
    "a clean painting of a person in the style of {}",
    "a dirty painting of a person in the style of {}",
    "a dark painting of a person in the style of {}",
    "a picture of a person in the style of {}",
    "a cool painting of a person in the style of {}",
    "a close-up painting of a person in the style of {}",
    "a bright painting of a person in the style of {}",
    "a cropped painting of a person in the style of {}",
    "a good painting of a person in the style of {}",
    "a close-up painting of a person in the style of {}",
    "a rendition of a person in the style of {}",
    "a nice painting of a person in the style of {}",
    "a small painting of a person in the style of {}",
    "a weird painting of a person in the style of {}",
    "a large painting of a person in the style of {}",
]

STYLE_PERSON_TEMPLATE2 = [
    "a painting of a person in the style of {}",
    "the painting of a person in the style of {}",
    "a clean painting of a person in the style of {}",
    "a picture of a person in the style of {}",
    "a bright painting of a person in the style of {}",
    "a good painting of a person in the style of {}",
    "a nice painting of a person in the style of {}",
]

BOTH_TEMPLATE = [
    "a photo of a {} in the style of {}",
    "a rendering of a {} in the style of {}",
    "a cropped photo of the {} in the style of {}",
    "the photo of a {} in the style of {}",
    "a photo of a clean {} in the style of {}",
    "a photo of a dirty {} in the style of {}",
    "a dark photo of the {} in the style of {}",
    "a photo of my {} in the style of {}",
    "a photo of the cool {} in the style of {}",
    "a close-up photo of a {} in the style of {}",
    "a bright photo of the {} in the style of {}",
    "a cropped photo of a {} in the style of {}",
    "a photo of the {} in the style of {}",
    "a good photo of the {} in the style of {}",
    "a photo of one {} in the style of {}",
    "a close-up photo of the {} in the style of {}",
    "a rendition of the {} in the style of {}",
    "a photo of the clean {} in the style of {}",
    "a rendition of a {} in the style of {}",
    "a photo of a nice {} in the style of {}",
    "a good photo of a {} in the style of {}",
    "a photo of the nice {} in the style of {}",
    "a photo of the small {} in the style of {}",
    "a photo of the weird {} in the style of {}",
    "a photo of the large {} in the style of {}",
    "a photo of a cool {} in the style of {}",
    "a photo of a small {} in the style of {}",
]

BOTH_TEMPLATE2 = [
    "a photo of a {} person in the style of {}",
    "the photo of a {} person in the style of {}",
    "a photo of a clean {} person in the style of {}",
    "a bright photo of the {} person in the style of {}",
    "a photo of the {} person in the style of {}",
    "a good photo of the {} person in the style of {}",
    "a photo of one {} person in the style of {}",
    "a photo of a nice {} person in the style of {}",
    "a good photo of a {} person in the style of {}",
    "a photo of the nice {} person in the style of {}",
]
def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret


def _shuffle(lis):

    return random.sample(lis, len(lis))

class PivotalTuningDatasetCapation(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        #stochastic_attribute,
        tokenizer,
        style_data_root = "",
        token_map: Optional[dict] = None,
        use_template: Optional[str] = None,
        class_data_root=None,
        class_prompt=None,
        size=512,
        h_flip=True,
        color_jitter=False,
        resize=True,
        use_face_segmentation_condition=False,
        blur_amount: int = 70,
        device="cuda",
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.resize = resize
        self.device = device
        self.all_images_path = []
        joint_ablation = True
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        if style_data_root != "":
            self.style_data_root = Path(style_data_root)
            if not self.style_data_root.exists():
                raise ValueError("Style images root doesn't exists.")



        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        if self.style_data_root.exists():
            self.style_images_path = list(Path(style_data_root).iterdir())
            self.num_style_images = len(self.style_images_path)
            ratio = self.num_style_images // self.num_instance_images
            print("style / obj ratio: ", ratio)
            if ratio < 1:
                ratio = 1
            for _ in range(ratio):
                for p in self.instance_images_path:
                    self.all_images_path.append((p, "instance"))
            for p in self.style_images_path:
                self.all_images_path.append((p, "style"))
            cur_len = len(self.all_images_path)
            card = self.num_instance_images * self.num_style_images
            drop_p = cur_len / (card * 2)
            print("Obj+style img drop prob: ", drop_p)
            # (instance : style : both) must be almost 1 : 1 : 1
            add_img = 0
            joined = []
            
            while add_img != cur_len // 2:
                for p in self.instance_images_path:
                    if add_img == cur_len // 2:
                        break
                    for q in self.style_images_path:
                        if (p, q) not in joined and random.random() < drop_p: 
                            self.all_images_path.append(((p, q), "both"))
                            add_img += 1
                            joined.append((p, q))
                        if add_img == cur_len // 2:
                            break

        self.token_map = token_map

        self.use_template = use_template

        if use_template == "object":
            self.templates = OBJECT_TEMPLATE2
        elif use_template == "person":
            self.templates = STYLE_PERSON_TEMPLATE2
        else:
            self.templates = STYLE_PERSON_TEMPLATE2
        
        self.style_templates = STYLE_PERSON_TEMPLATE2

        self.both_templates = BOTH_TEMPLATE2

        self._length = self.num_instance_images
        if self.style_data_root.exists():
            self._length *= ratio
            self._length += self.num_style_images
            self._length += add_img
            self.num_all_images = self._length

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_all_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
        self.h_flip = h_flip
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.CenterCrop(self.size),
                transforms.ColorJitter(0.1, 0.1)
                if color_jitter
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


        self.blur_amount = blur_amount

        # face parsing model
        n_classes = 19
        seg_net = BiSeNet(n_classes=n_classes)
        seg_net.to(device=device)
        save_pth = osp.join('face_parsing_PyTorch/res/cp', '79999_iter.pth')
        seg_net.load_state_dict(torch.load(save_pth))
        seg_net.eval()

        self.seg_net = seg_net

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        device = self.device
        temp_tuple = self.all_images_path[index % self.num_all_images]

        if temp_tuple[1] == "instance":
            # instance part
            instance_image = Image.open(
                temp_tuple[0]
            )
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            example["instance_images"] = self.image_transforms(instance_image)
            temp_img = (example["instance_images"].permute(1, 2, 0).unsqueeze(0).to(device=device) + 1) / 2
            temp_mask, example["face_emb"], det = get_face_feature_from_tensor(temp_img)
            example["face_mask"] = evaluate(imgpth=temp_tuple[0], net=self.seg_net, mask=det)
            example["det"] = det
            #temp_maskk = torch.where(example["face_mask"]==1, 1, 0.25)
            #masked_img = (temp_img.permute(0, 3, 1, 2))*temp_maskk
            #l = len(str(temp_tuple[0]))
            #transforms.functional.to_pil_image(masked_img[0]).save("_masksamples/"+str(temp_tuple[0])[l-7:])
            if self.use_template:
                assert self.token_map is not None
                input_tok = list(self.token_map.values())[0]

                text = random.choice(self.templates).format(input_tok)
            else:
                text = self.instance_images_path[index % self.num_instance_images].stem
                if self.token_map is not None:
                    for token, value in self.token_map.items():
                        text = text.replace(token, value)

            print(text)

            if self.h_flip and random.random() > 0.5:
                hflip = transforms.RandomHorizontalFlip(p=1)

                example["instance_images"] = hflip(example["instance_images"])
                example["face_mask"] = hflip(example["face_mask"])

            example["instance_prompt_ids"] = self.tokenizer(
                text,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        elif temp_tuple[1] == "style":
            # style part
            style_image = Image.open(
                temp_tuple[0]
            )
            if not style_image.mode == "RGB":
                style_image = style_image.convert("RGB")
            example["instance_images"] = self.image_transforms(style_image)
            temp_img = (example["instance_images"].permute(1, 2, 0).unsqueeze(0).to(device=device) + 1) / 2
            temp_mask, _, det = get_face_feature_from_tensor(temp_img)
            example["det"] = det
            example["face_mask"] = evaluate(imgpth=temp_tuple[0], net=self.seg_net, mask=det)
            example["face_mask"] = 1 - example["face_mask"]
            example["face_emb"] = None # None means style image, so have not face embedding

            #temp_maskk = torch.where(example["face_mask"]==0, 0.25, 1)
            ##masked_img = (temp_img.permute(0, 3, 1, 2))*temp_maskk
            #l = len(str(temp_tuple[0]))
            #transforms.functional.to_pil_image(masked_img[0]).save("_masksamples/"+str(temp_tuple[0])[l-7:])
            if self.use_template:
                assert self.token_map is not None
                input_tok = list(self.token_map.values())[1]

                text = random.choice(self.style_templates).format(input_tok)
            else:
                text = self.style_images_path[index % self.num_style_images].stem
                if self.token_map is not None:
                    for token, value in self.token_map.items():
                        text = text.replace(token, value)

            print(text)

            if self.h_flip and random.random() > 0.5:
                hflip = transforms.RandomHorizontalFlip(p=1)

                example["instance_images"] = hflip(example["instance_images"])
                example["face_mask"] = hflip(example["face_mask"])

            example["instance_prompt_ids"] = self.tokenizer(
                text,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        elif temp_tuple[1] == "both":
            # composed
            style_image = Image.open(
                temp_tuple[0][1]
            )
            if not style_image.mode == "RGB":
                style_image = style_image.convert("RGB")
            example["instance_images"] = self.image_transforms(style_image)
            temp_img = (example["instance_images"].permute(1, 2, 0).unsqueeze(0).to(device=device) + 1) / 2
            temp_mask, _, det = get_face_feature_from_tensor(temp_img)
            example["det"] = det
            example["face_mask"] = evaluate(imgpth=temp_tuple[0][1], net=self.seg_net, mask=det)
            example["face_mask"] = 1 - example["face_mask"]
            instance_image = Image.open(
                temp_tuple[0][0]
            )

            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            temp = self.image_transforms(instance_image)
            temp = (temp.permute(1, 2, 0).unsqueeze(0).to(device=device) + 1) / 2
            _, example["face_emb"], _ = get_face_feature_from_tensor(temp)

            if self.use_template:
                assert self.token_map is not None
                input_tok_instance, input_tok_style = list(self.token_map.values())[0], list(self.token_map.values())[1]

                text = random.choice(self.both_templates).format(input_tok_instance, input_tok_style)
            else:
                text = self.style_images_path[index % self.num_style_images].stem
                if self.token_map is not None:
                    for token, value in self.token_map.items():
                        text = text.replace(token, value)

            print(text)

            if self.h_flip and random.random() > 0.5:
                hflip = transforms.RandomHorizontalFlip(p=1)

                example["instance_images"] = hflip(example["instance_images"])
                example["face_mask"] = hflip(example["face_mask"])

            example["instance_prompt_ids"] = self.tokenizer(
                text,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            example["class_types"] = "class"
            example["class_face_emb"] = None
            example["class_masks"] = torch.ones_like(example["class_images"]).to(device=device)
        example["types"] = temp_tuple[1]
        return example


class PivotalTuningDatasetCapationTI(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        #stochastic_attribute,
        tokenizer,
        style_data_root = "",
        token_map: Optional[dict] = None,
        use_template: Optional[str] = None,
        class_data_root=None,
        class_prompt=None,
        size=512,
        h_flip=True,
        color_jitter=False,
        resize=True,
        use_face_segmentation_condition=False,
        blur_amount: int = 70,
        device="cuda",
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.resize = resize
        self.device = device
        self.all_images_path = []

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        if style_data_root != "":
            self.style_data_root = Path(style_data_root)
            if not self.style_data_root.exists():
                raise ValueError("Style images root doesn't exists.")



        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        if self.style_data_root.exists():
            self.style_images_path = list(Path(style_data_root).iterdir())
            self.num_style_images = len(self.style_images_path)
            ratio = self.num_style_images // self.num_instance_images
            print("style / obj ratio: ", ratio)
            if ratio < 1:
                ratio = 1
            for _ in range(ratio):
                for p in self.instance_images_path:
                    self.all_images_path.append((p, "instance"))
            for p in self.style_images_path:
                self.all_images_path.append((p, "style"))
            

        self.token_map = token_map

        self.use_template = use_template

        self.templates = OBJECT_TEMPLATE2
        self.style_templates = STYLE_PERSON_TEMPLATE2
        self.both_templates = BOTH_TEMPLATE2

        self._length = self.num_instance_images
        if self.style_data_root.exists():
            self._length *= ratio
            self._length += self.num_style_images
            self.num_all_images = self._length

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_all_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
        self.h_flip = h_flip
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.CenterCrop(self.size),
                transforms.ColorJitter(0.1, 0.1)
                if color_jitter
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.use_face_segmentation_condition = use_face_segmentation_condition
        if self.use_face_segmentation_condition:
            import mediapipe as mp

            mp_face_detection = mp.solutions.face_detection
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        self.blur_amount = blur_amount


        n_classes = 19
        seg_net = BiSeNet(n_classes=n_classes)
        seg_net.to(device=device)
        save_pth = osp.join('face_parsing_PyTorch/res/cp', '79999_iter.pth')
        seg_net.load_state_dict(torch.load(save_pth))
        seg_net.eval()

        self.seg_net = seg_net

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        device = self.device
        temp_tuple = self.all_images_path[index % self.num_all_images]

        if temp_tuple[1] == "instance":
            # instance part
            instance_image = Image.open(
                temp_tuple[0]
            )
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            example["instance_images"] = self.image_transforms(instance_image)
            temp_img = (example["instance_images"].permute(1, 2, 0).unsqueeze(0).to(device=device) + 1) / 2
            temp_mask, example["face_emb"], det = get_face_feature_from_tensor(temp_img)
            example["det"] = det
            example["face_mask"] = evaluate(imgpth=temp_tuple[0], net=self.seg_net, mask=det)

            if self.use_template:
                assert self.token_map is not None
                input_tok = list(self.token_map.values())[0]

                text = random.choice(self.templates).format(input_tok)
            else:
                text = self.instance_images_path[index % self.num_instance_images].stem
                if self.token_map is not None:
                    for token, value in self.token_map.items():
                        text = text.replace(token, value)

            print(text)

            if self.h_flip and random.random() > 0.5:
                hflip = transforms.RandomHorizontalFlip(p=1)

                example["instance_images"] = hflip(example["instance_images"])
                example["face_mask"] = hflip(example["face_mask"])

            example["instance_prompt_ids"] = self.tokenizer(
                text,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        elif temp_tuple[1] == "style":
            # style part
            print(temp_tuple[0])
            style_image = Image.open(
                temp_tuple[0]
            )
            if not style_image.mode == "RGB":
                style_image = style_image.convert("RGB")
            example["instance_images"] = self.image_transforms(style_image)
            temp_img = (example["instance_images"].permute(1, 2, 0).unsqueeze(0).to(device=device) + 1) / 2
            temp_mask, _, det = get_face_feature_from_tensor(temp_img)
            example["det"] = det
            example["face_mask"] = evaluate(imgpth=temp_tuple[0], net=self.seg_net, mask=det)
            example["face_mask"] = 1 - example["face_mask"]
            example["face_emb"] = None # None means style image, so have not face embedding
            
            if self.use_template:
                assert self.token_map is not None
                input_tok = list(self.token_map.values())[1]

                text = random.choice(self.style_templates).format(input_tok)
            else:
                text = self.style_images_path[index % self.num_style_images].stem
                if self.token_map is not None:
                    for token, value in self.token_map.items():
                        text = text.replace(token, value)

            print(text)

            if self.h_flip and random.random() > 0.5:
                hflip = transforms.RandomHorizontalFlip(p=1)

                example["instance_images"] = hflip(example["instance_images"])
                example["face_mask"] = hflip(example["face_mask"])

            example["instance_prompt_ids"] = self.tokenizer(
                text,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            example["class_types"] = "class"
            example["class_face_emb"] = None
            example["class_masks"] = torch.ones_like(example["class_images"]).to(device=device)
        example["types"] = temp_tuple[1]
        return example



class PivotalTuningDatasetCapationGeneral(Dataset):


    def __init__(
        self,
        instance_data_root,
        #stochastic_attribute,
        instance_data_point_root,
        tokenizer,
        style_data_root = "",
        token_map: Optional[dict] = None,
        use_template: Optional[str] = None,
        class_data_root=None,
        class_prompt=None,
        size=512,
        h_flip=True,
        color_jitter=False,
        resize=True,
        use_face_segmentation_condition=False,
        blur_amount: int = 70,
        device="cuda",
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.resize = resize
        self.device = device
        self.all_images_path = []
        joint_ablation = True
        self.instance_data_root = Path(instance_data_root)
        self.instance_data_point_root = Path(instance_data_point_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        if style_data_root != "":
            self.style_data_root = Path(style_data_root)
            if not self.style_data_root.exists():
                raise ValueError("Style images root doesn't exists.")



        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.instance_data_point_path = list(Path(instance_data_point_root).iterdir())
        self.instance_images_path = [(x, y) for x, y in zip(self.instance_images_path, self.instance_data_point_path)]
        self.num_instance_images = len(self.instance_data_point_path)
        if self.style_data_root.exists():
            self.style_images_path = list(Path(style_data_root).iterdir())
            self.num_style_images = len(self.style_images_path)
            ratio = self.num_style_images // self.num_instance_images
            print("style / obj ratio: ", ratio)
            if ratio < 1:
                ratio = 1
            for _ in range(ratio):
                for p in self.instance_images_path:
                    self.all_images_path.append((p, "instance"))
            for p in self.style_images_path:
                self.all_images_path.append((p, "style"))
            cur_len = len(self.all_images_path)
            card = self.num_instance_images * self.num_style_images
            drop_p = cur_len / (card * 2)
            print("Obj+style img drop prob: ", drop_p)
            # (instance : style : both) must be almost 1 : 1 : 1
            add_img = 0
            joined = []
            
            while add_img != cur_len // 2:
                print(add_img, cur_len)
                for p in self.instance_images_path:
                    if add_img == cur_len // 2:
                        break
                    for q in self.style_images_path:
                        if (p, q) not in joined and random.random() < drop_p: 
                            self.all_images_path.append(((p, q), "both"))
                            add_img += 1
                            joined.append((p, q))
                        if add_img == cur_len // 2:
                            break

        self.token_map = token_map

        self.use_template = use_template

        self.templates = OBJECT_TEMPLATE
        self.style_templates = STYLE_TEMPLATE
        self.both_templates = BOTH_TEMPLATE

        self._length = self.num_instance_images
        if self.style_data_root.exists():
            self._length *= ratio
            self._length += self.num_style_images
            self._length += add_img
            self.num_all_images = self._length

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_all_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
        self.h_flip = h_flip
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.CenterCrop(self.size),
                transforms.ColorJitter(0.1, 0.1)
                if color_jitter
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.use_face_segmentation_condition = use_face_segmentation_condition
        if self.use_face_segmentation_condition:
            import mediapipe as mp

            mp_face_detection = mp.solutions.face_detection
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        self.blur_amount = blur_amount


        sam_checkpoint = "../sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)


        self.dino_processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        """
        model_vit_dino = timm.create_model(
            'vit_small_patch16_224.dino',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        """
        self.model_vit_dino = ViTModel.from_pretrained(
            'facebook/dino-vits16'
        )
        self.model_vit_dino = self.model_vit_dino.eval()
        self.model_vit_dino.requires_grad_(False)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        device = self.device
        temp_tuple = self.all_images_path[index % self.num_all_images]

        if temp_tuple[1] == "instance":
            # instance part
            instance_image = Image.open(
                temp_tuple[0][0]
            )
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")

            example["instance_images"] = self.image_transforms(instance_image)
            point_f = open(str(temp_tuple[0][1]), "r")
            points = []
            while True:
                line = point_f.readline()
                if not line : break
                points.append(list(map(lambda x: int(x), line.split(','))))
            
            image = cv2.imread(str(temp_tuple[0][0]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512))

            input_point = np.array(points)
            input_label = np.array([1])
            input_label2= np.array([1, 1])
            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array([input_point[0]]),
                point_labels=input_label,
                multimask_output=True,
            )

            mask_input = logits[np.argmax(scores), :, :]
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label2,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )

            h, w = masks.shape[-2:]
            masked = torch.from_numpy(np.ones_like(image) * masks.reshape(h, w, 1))
            example["obj_mask"] = masked.permute(2, 0, 1).unsqueeze(0)

            if self.use_template:
                assert self.token_map is not None
                input_tok = list(self.token_map.values())[0]

                text = random.choice(self.templates).format(input_tok)
            else:
                text = self.instance_images_path[index % self.num_instance_images].stem
                if self.token_map is not None:
                    for token, value in self.token_map.items():
                        text = text.replace(token, value)

            print(text)

            if self.h_flip and random.random() > 0.5:
                hflip = transforms.RandomHorizontalFlip(p=1)

                example["instance_images"] = hflip(example["instance_images"])
                example["obj_mask"] = hflip(example["obj_mask"])

            example["instance_prompt_ids"] = self.tokenizer(
                text,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            example["style_images"] = None

            dino_inputs = self.dino_processor(images=instance_image, return_tensors="pt")
            dino_outputs = self.model_vit_dino(**dino_inputs).last_hidden_state[:, 0]
            example["dino_emb"] = dino_outputs

        elif temp_tuple[1] == "style":
            # style part
            style_image = Image.open(
                temp_tuple[0]
            )
            if not style_image.mode == "RGB":
                style_image = style_image.convert("RGB")
            example["instance_images"] = self.image_transforms(style_image)
            example["obj_mask"] = None
            if self.use_template:
                assert self.token_map is not None
                input_tok = list(self.token_map.values())[1]

                text = random.choice(self.style_templates).format(input_tok)
            else:
                text = self.style_images_path[index % self.num_style_images].stem
                if self.token_map is not None:
                    for token, value in self.token_map.items():
                        text = text.replace(token, value)

            print(text)

            if self.h_flip and random.random() > 0.5:
                hflip = transforms.RandomHorizontalFlip(p=1)

                example["instance_images"] = hflip(example["instance_images"])

            example["instance_prompt_ids"] = self.tokenizer(
                text,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            example["style_images"] = None # for ease
            example["dino_emb"] = None
            example["obj_mask"] = torch.ones((1, 3, 512, 512))
        elif temp_tuple[1] == "both":

            style_image = Image.open(
                temp_tuple[0][1]
            )
            if not style_image.mode == "RGB":
                style_image = style_image.convert("RGB")
            example["style_images"] = self.image_transforms(style_image)
            
            instance_image = Image.open(
                temp_tuple[0][0][0]
            )

            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            example["instance_images"] = self.image_transforms(instance_image)

            point_f = open(str(temp_tuple[0][0][1]), "r")
            points = []
            while True:
                line = point_f.readline()
                if not line : break
                points.append(list(map(lambda x: int(x), line.split(','))))
            
            image = cv2.imread(str(temp_tuple[0][0][0]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512))

            input_point = np.array(points)
            input_label = np.array([1])
            input_label2= np.array([1, 1])
            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array([input_point[0]]),
                point_labels=input_label,
                multimask_output=True,
            )

            mask_input = logits[np.argmax(scores), :, :]
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label2,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )

            h, w = masks.shape[-2:]
            masked = torch.from_numpy(np.ones_like(image) * masks.reshape(h, w, 1))
            example["obj_mask"] = masked.permute(2, 0, 1).unsqueeze(0) # (h, w, 1)?

            if self.use_template:
                assert self.token_map is not None
                input_tok_instance, input_tok_style = list(self.token_map.values())[0], list(self.token_map.values())[1]

                text = random.choice(self.both_templates).format(input_tok_instance, input_tok_style)
            else:
                text = self.style_images_path[index % self.num_style_images].stem
                if self.token_map is not None:
                    for token, value in self.token_map.items():
                        text = text.replace(token, value)

            print(text)

            if self.h_flip and random.random() > 0.5:
                hflip = transforms.RandomHorizontalFlip(p=1)

                example["instance_images"] = hflip(example["instance_images"])
                example["obj_mask"] = hflip(example["obj_mask"])

            example["instance_prompt_ids"] = self.tokenizer(
                text,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

            dino_inputs = self.dino_processor(images=instance_image, return_tensors="pt")
            dino_outputs = self.model_vit_dino(**dino_inputs).last_hidden_state[:, 0]
            example["dino_emb"] = dino_outputs
        
        

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            example["class_types"] = "class"
            example["class_face_emb"] = None
            example["class_masks"] = torch.ones_like(example["class_images"]).to(device=device)
        example["types"] = temp_tuple[1]
        return example
