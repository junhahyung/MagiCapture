from lora_diffusion_ import monkeypatch_or_replace_lora, tune_lora_scale, patch_pipe
import torch
import torch.nn.functional as F
import clip
from PIL import Image
import os
import sys
from kornia.geometry.transform import warp_affine

import cv2
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
# for face detection
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from diffusers.utils import randn_tensor
from build_eval_pack import get_norm_crop, get_norm_crop2
from arcface_torch.backbones import get_model

import torchvision.transforms as T
from PIL import Image

transform = T.ToPILImage()

# cosface, r100
# face_model_path = "arcface_torch/models/glint360k_cosface_r100_fp16_0.1/backbone.pth"
# net = get_model('r100', fp16=True)
# net.load_state_dict(torch.load(face_model_path))
# net.to("cuda")
# net.eval()

# arcface, r50
face_model_path = "arcface_torch/models/ms1mv3_arcface_r100_fp16/backbone.pth"
net = get_model('r100', fp16=True)
net.load_state_dict(torch.load(face_model_path))
net.to("cuda")
net.eval()

def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask

def get_face_feature_from_tensor(img, dtype='uint8', app=None):
    # extract face feature from the img (tensor)
    # use arcface, return the masked face bbox
    # img = img[0]
    # img is in [0, 1]
    #img = img.permute(0, 2, 3, 1)
    detection_img = img[0].clone().detach()
    detection_img = detection_img.to(dtype=torch.float32)
    detection_img = detection_img.cpu().numpy()
    detection_img = (detection_img * 255).round()
    prev_dtype = img.dtype

    detection_img = cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
    
    height, width = img.shape[1:3]
    mask = np.zeros((height, width, 1), dtype=dtype)

    img = img.permute(0, 3, 1, 2)
    wraped_output = get_norm_crop2(detection_img, img)
    if wraped_output is not None:
        warped_img, det = wraped_output
        det = [int(x) for x in det[0]]
        warped_img.sub_(0.5).div_(0.5)
        face_feature = net(warped_img)
        mask[det[1]:det[3], det[0]:det[2]] = 1
        mask = torch.from_numpy(mask).cuda().permute(2, 0, 1).unsqueeze(0)
        box = det
    else: 
        # if there is no face, return None mask
        mask = None
        face_feature = None
        box = None

    return mask, face_feature, box


def get_face_feature_from_folder(path, h, w, device):
    # from the imgs in folder at "path", extract face feature
    filelist = os.listdir(path)

    num_data = 0
    features = torch.zeros((1, 512)).to(device=device)
    feature_list = []
    det_list = []
    for name in filelist:
        num_data += 1
        img = cv2.imread(path+"/"+name)
        img_aligned, det = get_norm_crop(path+"/"+name)

        det = [int(x) for x in det[0]]
        cv2.imwrite("test.jpg", img[det[1]:det[3], det[0]:det[2]])
        img = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(device=device)
        img.div_(255).sub_(0.5).div_(0.5)
        with torch.no_grad():
            faces = net(img)
            feature_list.append(faces[0])
        features += faces
        det_list.append(det)
    return feature_list, det_list


def get_face_feature_from_tensor_debug(img, dtype='uint8', app=None):
    # extract face feature from the img (tensor)
    # use arcface, return the masked face bbox
    # img = img[0]
    # img is in [0, 1]
    #img = img.permute(0, 2, 3, 1)
    detection_img = img[0].clone().detach()
    detection_img = detection_img.to(dtype=torch.float32)
    detection_img = detection_img.cpu().numpy()
    detection_img = (detection_img * 255).round()
    prev_dtype = img.dtype
    detection_img = cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
    
    height, width = img.shape[1:3]
    mask = np.zeros((height, width, 1), dtype=dtype)

    img = img.permute(0, 3, 1, 2)
    wraped_output = get_norm_crop2(detection_img, img)
    if wraped_output is not None:
        warped_img, det = wraped_output
        det = [int(x) for x in det[0]]

        warped_img.sub_(0.5).div_(0.5)
        import pdb;pdb.set_trace()
        face_feature = net(warped_img)
        mask[det[1]:det[3], det[0]:det[2]] = 1
        mask = torch.from_numpy(mask).cuda().permute(2, 0, 1).unsqueeze(0)
        box = det
    else: 
        # if there is no face, return None mask
        mask = None
        face_feature = None
        box = None

    return mask, face_feature, box
