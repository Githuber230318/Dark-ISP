import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from os import path as osp
import matplotlib.pyplot as plt
import PIL.Image as Image
from tqdm import tqdm
import random
import rawpy
from glob import glob
import torchvision
from multiprocessing import Pool
from data_preparation.raw_utils import metainfo

raw_path = r'/public/home/guojs/data/LOD/RAW_dark' # Dark RAW image directory
clean_path = r'/public/home/guojs/data/LOD/RAW_normal' # Normal RAW image directory

out_path = r'/public/home/guojs/data/LOD/RAW_dark_denoised' # RGB format saved path

save_folder = r'/public/home/guojs/data/LOD/RAW_dark_denoised_npz' # RAW npz files saved path

def apply_wb_ccm(bayer_images, wbs, ccms): # 应用颜色校正矩阵
    """""Applies white balance to a batch of Bayer images."""""
    N, C, _, _ = bayer_images.shape
    bayer_images = bayer_images * wbs.view(N, C, 1, 1)
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    """RGBG -> RGB"""
    images = torch.stack([
        bayer_images[:,0,...],
        torch.mean(bayer_images[:, [1,3], ...], dim=1),
        bayer_images[:,2,...]], dim=1)
    """Applies a color correction matrix."""
    images = images.permute(0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    images = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format

    return images.permute(0,3,1,2)


def gamma_expansion(images, gamma=2.2): # gamma扩展，将linear值转换为non-linear
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    outs = torch.clamp((outs*255).int(), min=0, max=255).float() / 255
    return outs

def default_ISP(image, wb, ccm):
    wb = torch.from_numpy(wb).float().contiguous().unsqueeze(0)
    ccm = torch.from_numpy(ccm).float().contiguous().unsqueeze(0)
    image_g = cam_process(image, wb, ccm)
    image = gamma_expansion(image_g)
    return image

def cam_process(image, wb, ccm):
    image = apply_wb_ccm(image, wb, ccm) # 白平衡和颜色校正
    image = torch.clamp(image, min=0.0, max=1.0)
    return image


def depack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    _, H, W = raw.shape
    raw = raw.astype(np.uint16)

    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw_flatten = np.zeros((H * 2, W * 2))
    raw_flatten[R[0][0]::2,  R[1][0]::2] = raw[0]
    raw_flatten[G1[0][0]::2,  G1[1][0]::2] = raw[1]
    raw_flatten[B[0][0]::2,  B[1][0]::2] = raw[2]
    raw_flatten[G2[0][0]::2,  G2[1][0]::2] = raw[3]

    raw_flatten = raw_flatten.astype(np.uint16)
    return raw_flatten

def worker(path, out_path):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    file = osp.basename(path)
    img_name, extension = osp.splitext(osp.basename(path))
    # Setp 0: Load RAW data
    raw = rawpy.imread(path)
    iso, exp_time = metainfo(path)
    clean_raw = f'{clean_path}/{str(int(img_name)-1)}.CR2'
    iso_gt, exp_time_gt = metainfo(clean_raw)
    ratio = (iso_gt * exp_time_gt)/(iso * exp_time)
    # rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
    raw_pattern = raw.raw_pattern
    mosaic = raw.raw_image_visible.copy() 
    wb = np.array(raw.camera_whitebalance).copy()
    wb /= wb[1]
    ccm = np.array(raw.rgb_camera_matrix[:3, :3]).copy()


    black_level = np.array(raw.black_level_per_channel).reshape(4, 1, 1)
    white_level = np.array(raw.camera_white_level_per_channel).reshape(4, 1, 1)

    scale = white_level[0] - black_level[0]


    raw = mosaic
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    H, W = raw.shape
    if H % 2 == 1:
        H -= 1
    if W % 2 == 1:
        W -= 1
    packed = np.stack((raw[R[0][0]:H:2,  R[1][0]:W:2], #RGBG
                    raw[G1[0][0]:H:2, G1[1][0]:W:2],
                    raw[B[0][0]:H:2,  B[1][0]:W:2],
                    raw[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0).astype(np.uint16)
    packed = np.ascontiguousarray(packed)

    img = (packed - black_level[0]) / scale
    img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, (1200, 800),interpolation=cv2.INTER_CUBIC)
    img = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)
    img = img * ratio
    # img = default_ISP(img, wb, ccm)
    img = torch.stack([img[:,0,...],
                torch.mean(img[:, [1,3], ...], dim=1),
                img[:,2,...]], dim=1)
    img = img.clip(0,1)
    torchvision.utils.save_image(img, os.path.join(out_path, file.replace('.CR2', '.png')))

    packed = np.transpose(packed, (1, 2, 0))
    raw_resize = cv2.resize(packed, (1200, 800),interpolation=cv2.INTER_AREA)
    raw_low = raw_resize
    raw_pattern = raw_pattern
    np.savez(
        osp.join(save_folder, f'{img_name}.npz'),
        im = raw_low,
        ratio = ratio,
        raw_pattern = raw_pattern,
        black_level = black_level,
        white_level = white_level,
        wb = wb,
        ccm = ccm,
        iso = iso,
        exp_time = exp_time
        )
    process_info = f'Processing {img_name} ...'
    return process_info


os.makedirs(out_path, exist_ok=True)
# os.makedirs(save_folder, exist_ok=True)


if __name__ == '__main__':
    suffix = '.CR2'
    img_list = glob(f'{raw_path}/*{suffix}')
    pool = Pool(1)
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    for path in img_list:
        pool.apply_async(worker, args=(path, out_path), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()