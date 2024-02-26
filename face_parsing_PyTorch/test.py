#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .logger import setup_logger
from .model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='res/test_res/test.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im.cpu())
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)
    """
    index1 = np.where(1 <= vis_parsing_anno)
    index2 = np.where(vis_parsing_anno <= 13)
    index = index1 and index2
    for pi in range(1, 14):
        vis_parsing_anno_color[index[0], index[1], :] = [111, 111, 111]
    """
    for pi in range(1, 14):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = [0, 0, 0]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    #vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.0, vis_parsing_anno_color, 1.0, 0)
    vis_im = vis_parsing_anno_color
    # Save result or not
    vis_im = torch.from_numpy(vis_im // 255).cuda().permute(2, 0, 1)
    vis_im = 1 - vis_im
    #print(vis_im.shape)
    # return vis_im
    return vis_im[0].unsqueeze(0)

def vis_parsing_maps2(im, parsing_anno, stride, save_im=False, save_path='res/test_res/test.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im2 = np.array(im[0].permute(1, 2, 0).cpu())
    im = np.array(im.cpu())
    
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    #print(im.shape, vis_parsing_anno.shape)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + np.array(cv2.cvtColor(im2, cv2.COLOR_RGB2BGR))
    #print(im.shape, vis_parsing_anno_color.shape)
    num_of_class = np.max(vis_parsing_anno)
    """
    index1 = np.where(1 <= vis_parsing_anno)
    index2 = np.where(vis_parsing_anno <= 13)
    index = index1 and index2
    for pi in range(1, 14):
        vis_parsing_anno_color[index[0], index[1], :] = [111, 111, 111]
    """
    for pi in range(1, 14):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = [0, 0, 0]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    print(vis_parsing_anno_color.shape, vis_im.shape)
    
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR), 0.2, vis_parsing_anno_color, 0.8, 0)
    vis_im = torch.from_numpy(vis_im // 255).cuda().permute(2, 0, 1)
    vis_im = 1 - vis_im
    #print(vis_im.shape)
    # return vis_im
    return vis_im[0].unsqueeze(0)


def evaluate(respth='./res/test_res', imgpth='./data', cp='model_final_diss.pth', net=None, mask=None):

    if not os.path.exists(respth):
        os.makedirs(respth)
    """
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    """
    if net is None:
        print("Net is None. Abort")
        exit()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    resize0 = transforms.Resize(512)
    resize1 = transforms.Resize((512, 512))
    resize2 = transforms.Resize((mask[3]-mask[1], mask[2]-mask[0]), interpolation=transforms.InterpolationMode.NEAREST)
    centercrop = transforms.CenterCrop(512)
    with torch.no_grad():
        img = Image.open(imgpth)
        #image = img.resize((512), Image.BILINEAR)
        img = centercrop(resize0(to_tensor(img)))
        img = torch.unsqueeze(img, 0)
        if mask is not None:
            #print(img.shape)
            img = img[:, :, mask[1]:mask[3], mask[0]:mask[2]]
            img = resize1(img)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        #print(np.unique(parsing))

        return_masks = vis_parsing_maps(img, parsing, stride=1, save_im=False)
        return_masks = resize2(return_masks)
        return_masks = (return_masks + 0.01).pow(1.0)

        return_masks = return_masks / return_masks.max()
        return_mask = torch.zeros_like(img)
        return_mask[:, :, mask[1]:mask[3], mask[0]:mask[2]] = return_masks

    #return return_masks.unsqueeze(0)
    return return_mask
    # N x H x W x C





if __name__ == "__main__":
    evaluate(imgpth='data/6.jpg', cp='79999_iter.pth')


