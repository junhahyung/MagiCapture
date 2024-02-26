import os
import sys
sys.path.append(os.path.join(os.getcwd(), './CodeFormer/basicsr'))

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler, KDPM2AncestralDiscreteScheduler, PNDMScheduler
from lora_diffusion_ import monkeypatch_or_replace_lora, tune_lora_scale, patch_pipe
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import clip
from PIL import Image
import cv2
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union

import insightface
from insightface.app import FaceAnalysis
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from diffusers.utils import randn_tensor
from arcface_torch.backbones import get_model
from CodeFormer.inference_codeformer import Codeformer_img, get_net_and_face_helper
import argparse
import datetime


parser = argparse.ArgumentParser(description='Inference lora with iterative face identity injection')
parser.add_argument('--lora_path', type=str, default=None, help='path of lora weight which you want to use')
parser.add_argument('--pretrained_model_name', type=str, default="stabilityai/stable-diffusion-2-1-base", help='name of model card at huggingface which you want to use')
parser.add_argument('--step', type=int, default=50, help='number of denoising processing step')
parser.add_argument('--output_folder_prefix', type=str, default="", help='the prefix of name of output folder')
parser.add_argument('--start_scheduler', type=str, default="euler", help='choose among of euler, dpm, kdpm')
parser.add_argument('--guidance_scale', type=float, default=7.0, help='CFG scale value')
parser.add_argument('--prompt', type=str, default="A photo of a <sks> person with style <style1>")
parser.add_argument('--negative_prompt', type=str, default="")
parser.add_argument('--seed', type=int, default=777)
args = parser.parse_args()



################################################################################################################
# settings



schedulers = {
    "euler": EulerAncestralDiscreteScheduler,
    "dpm": DPMSolverMultistepScheduler,
    "kdpm": KDPM2AncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
}

codeformer_net, face_helper = get_net_and_face_helper()

torch.manual_seed(args.seed)

pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name, torch_dtype=torch.float16).to(
    "cuda"
)

pipe.scheduler = schedulers[args.start_scheduler].from_config(pipe.scheduler.config)

patch_pipe(
    pipe,
    args.lora_path+"/final_lora.safetensors",
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
)
tune_lora_scale(pipe.unet, 0.60)
tune_lora_scale(pipe.text_encoder, 0.90)


# save folder
output_folder_name = "output/"+args.output_folder_prefix
os.makedirs(output_folder_name, exist_ok=True)

with torch.no_grad():
    img = pipe(args.prompt, negative_prompt=args.negative_prompt, num_inference_steps=args.step, guidance_scale=args.guidance_scale, output_type="img").images
    refined_img = Codeformer_img(
        input_path="temp.jpg",
        input_image=(img[0]*255).astype(np.uint8),
        output_path="",
        fidelity_weight=0.5,
        upscale=2,
        has_aligned=False,
        only_center_face=False,
        draw_box=False,
        detection_model='retinaface_resnet50',
        bg_upsampler=None,
        face_upsample=False,
        bg_tile=400,
        suffix=None,
        save_video_fps=None
    )
    
    refined_img = to_pil_image(refined_img.to(dtype=torch.float16, device="cuda")[0].permute(2, 0, 1))
    refined_img.save(output_folder_name+str(datetime.datetime.now())+".jpg")
