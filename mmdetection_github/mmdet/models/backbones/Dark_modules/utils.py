# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 01:14:23 2021

@author: chxy
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union


def process(img, metadata=None, predictdata=None):
    if metadata is not None:
        return cam_process(img, metadata)
    else:
        return cam_process(img, predictdata)

def normlize(images, wl, bl):
    b, c, h, w = images.shape
    scale = wl - bl
    bl = bl.view(b,c,1,1)
    scale = scale.view(b,c,1,1)
    images = (images - bl) / scale
    images = torch.clamp(images, 0., 1)
    return images

def unnormalize(images, wl, bl):
    scale = wl - bl
    images = images * scale + bl
    return images

def apply_wb(bayer_images, wbs):
    """""Applies white balance to a batch of Bayer images."""""
    N, C, H, W = bayer_images.shape
    bayer_images = bayer_images * wbs.view(N, C, 1, 1)
    # bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    """RGBG -> RGB"""
    images = torch.stack([
        bayer_images[:,0,...],
        torch.mean(bayer_images[:, [1,3], ...], dim=1),
        bayer_images[:,2,...]], dim=1)
    return images

def apply_wb_ccm(bayer_images, wbs, ccms): # 应用颜色校正矩阵
    """""Applies white balance to a batch of Bayer images."""""
    N, C, H, W = bayer_images.shape
    bayer_images = bayer_images * wbs.view(N, C, 1, 1)
    # bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    """RGBG -> RGB"""
    images = torch.stack([
        bayer_images[:,0,...],
        torch.mean(bayer_images[:, [1,3], ...], dim=1),
        bayer_images[:,2,...]], dim=1)
    # images = raw2rgb_linear(bayer_images)
    """Applies a color correction matrix."""
    # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images.permute(0, 2, 3, 1)  
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    images = torch.sum(images * ccms, dim=-1)
    images = images.permute(0,3,1,2)

    return images.clip(0., 1)

def color_transform(images, ccms, dmat=None):
    B, C, H, W = images.shape
    images = images.permute(0, 2, 3, 1)  
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    if dmat is not None:
        # dmat: [B, 12, H, W]
        ccm_h, ccm_w = ccms.shape[-2:]
        dmat = dmat.permute(0, 2, 3, 1)
        dmat = dmat.view(B, H, W, ccm_h, ccm_w)
        ccms = ccms + dmat
    images = torch.sum(images * ccms, dim=-1)
    images = images.permute(0,3,1,2)
    # images = torch.clamp(images, 1e-5, 1)
    return images, ccms

def gamma_expansion(images, gamma=2.2): # gamma扩展，将linear值转换为non-linear
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    outs = torch.clamp(outs*255, min=0, max=255).float() / 255
    return outs

def cam_process(image, params):
    image = apply_wb_ccm(image, params['wb'], params['ccm']) # 白平衡和颜色校正
    # image = torch.clamp(image, min=0.0, max=1.0)
    return image

def default_ISP(image, metainfo):
    image_g = cam_process(image, metainfo)
    image = gamma_expansion(image_g)
    return image, image_g

def default_ISP_3c(image, metainfo):
    image_g = cam_process_3c(image, metainfo)
    image = gamma_expansion(image_g)
    return image, image_g

def cam_process_3c(image, params):
    image = apply_wb_ccm_3c(image, params['wb'], params['ccm']) # 白平衡和颜色校正
    image = torch.clamp(image, min=0.0, max=1.0)
    return image

def apply_wb_ccm_3c(images, wbs, ccms): # 应用颜色校正矩阵
    """""Applies white balance to a batch of Bayer images."""""
    N, C, _, _ = images.shape
    wbs = torch.stack([
        wbs[:,0,...],
        torch.mean(wbs[:, [1,3], ...], dim=1),
        wbs[:,2,...]], dim=1)
    images = images * wbs.view(N, C, 1, 1)
    images = torch.clamp(images, min=0.0, max=1.0)
    """Applies a color correction matrix."""
    images = images.permute(0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    images = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format

    return images.permute(0,3,1,2)


def gaussian_kernel(size: int, sigma: float):
    """Generate Gaussian Kernel"""
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size),
    )
    kernel = torch.Tensor(kernel)
    kernel /= torch.sum(kernel)
    return kernel


def Tayler(img: torch.Tensor, weights: List) -> torch.Tensor:
    
    img = img.clip(0, 1)
    outs = []
    for i in range(3):
        x = img[:, [i], :, :]
        embed = torch.cat([
            x,
            (x - x ** 2 + x),
            (x ** 3) - 3 * (x ** 2) + 3 * x,
            -(x ** 4) + 4 * x**3 - 6*x**2 + 3*x +x,
            -6*(x ** 5) + 15 * x**4 - 10* x**3 +2*x,
            -1.2 * (x**6) + 3.0 * x**5 -2.4*x**4 + 1.6* x,
            -1.344*x**7 + 3.36*x**6 - 2.688*x**5 + 0.672*x + x,
            -12*x**8+48*x **7-60*x **6+24*x **5 +x,
            ], dim=1)
        out = (embed * weights[i]).sum(dim=1, keepdim=True)
        outs.append(out)
    outs = torch.cat(outs, dim=1)
    # img = (img + outs).clip(0,1)
    img = outs
    return img


def Tayler_res(img: torch.Tensor, weights: List) -> torch.Tensor:

    img = img.clip(0, 1)
    outs = []
    for i in range(3):
        x = img[:, [i], :, :]
        embed = torch.cat([
            torch.zeros_like(x),
            (x - x ** 2),
            (x ** 3) - 3 * (x ** 2) + 2 * x,
            -(x ** 4) + 4 * x**3 - 6*x**2 + 3*x,
            -6*(x ** 5) + 15 * x**4 - 10* x**3 +x,
            -1.2 * (x**6) + 3.0 * x**5 -2.4*x**4 + 0.6* x,
            -1.344*x**7 + 3.36*x**6 - 2.688*x**5 + 0.672*x ,
            -12*x**8+48*x **7-60*x **6+24*x **5 ,
            ], dim=1)
        out = (embed * weights[i]).sum(dim=1, keepdim=True)
        outs.append(out)
    outs = torch.cat(outs, dim=1)
    # img = (img + outs).clip(0,1)
    img = outs
    return img

def cosine_similarity(tensorA, tensorB):
    # tensorA: [4, h, w, 3, 4] 
    # tensorB: [4, h, w, 3, 4]
    dot_product = (tensorA * tensorB).sum(dim=4)  # 形状为 [4, h, w, 3]

    # calculate norm
    norm1 = torch.norm(tensorB, dim=-1) + 1e-5  # 形状为 [4, h, w, 3]
    norm2 = torch.norm(tensorA, dim=-1) + 1e-5 # 形状为 [4, h, w, 3]

    # calculate similarity
    similarity = dot_product / (norm1 * norm2)
    loss_mat = (1 - similarity).sum(dim=3).mean()

    return loss_mat







