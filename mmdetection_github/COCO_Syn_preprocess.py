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
from mmdet.datasets.transforms.noisemodel.unprocess import *
from pycocotools.coco import COCO

# data_dir = r'/public/home/guojs/data/COCO/coco'
# annotation_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
#   # 注释文件路径

data_path = r'/public/home/guojs/data/COCO/coco/val2017'
out_path = r'/public/home/guojs/data/COCO/coco/val2017_SynRAW'

save_folder = r'/public/home/guojs/data/COCO/coco/val2017_npz'
# out_path_oe = r'/data/unagi0/cui_data/light_dataset/PASCAL_RAW/PASCALRAW/demosaic_oe'

def _sample_params():
        # if self.fix_noise:
            # return self.fix_params
        camera='CanonEOS5D4'
        Q_step = 1  # 量化层级

        profiles = ['Profile-1']
        saturation_level = 16383 - 512  # 相机标定值，白点减去黑电平值

        # 调取相机标定参数
        camera_params = {}
        param_dir = '/home1/guojs/codes/Dark-ISP/mmdetection_github/mmdet/datasets/transforms/noisemodel/camera_params'
        camera_params[camera] = np.load(join(
            param_dir, camera+'_params.npy'), allow_pickle=True).item()  # 加载指定相机标定参数
        camera_params = camera_params[camera]
        Kmin = camera_params['Kmin']
        Kmax = camera_params['Kmax']

        # 抽取标定参数中的随机数
        G_shape = np.random.choice(camera_params['G_shape'])
        ind = np.random.randint(0, camera_params['color_bias'].shape[0])
        color_bias = camera_params['color_bias'][ind, :]
        profile = np.random.choice(profiles)
        camera_params = camera_params[profile]

        # 初始化noisemodel模型中的随机数
        log_K = np.random.uniform(low=np.log(Kmin), high=np.log(Kmax))
        log_g_scale = np.random.standard_normal() * camera_params['g_scale']['sigma'] * 1 +\
            camera_params['g_scale']['slope'] * \
            log_K + camera_params['g_scale']['bias']
        log_G_scale = np.random.standard_normal() * camera_params['G_scale']['sigma'] * 1 +\
            camera_params['G_scale']['slope'] * \
            log_K + camera_params['G_scale']['bias']
        log_R_scale = np.random.standard_normal() * camera_params['R_scale']['sigma'] * 1 +\
            camera_params['R_scale']['slope'] * \
            log_K + camera_params['R_scale']['bias']

        # 计算noisemodel中的参数
        K = np.exp(log_K)
        g_scale = np.exp(log_g_scale)
        G_scale = np.exp(log_G_scale)
        R_scale = np.exp(log_R_scale)

        # radio控制噪声的大小，ratio取值越大噪声越明显
        noise_ratio=(10, 100)
        ratio = np.random.uniform(low=noise_ratio[0], high=noise_ratio[1])
        # ratio = np.random.uniform(low=100, high=300)
        # ratio = np.random.uniform(low=20, high=100)
        # ratio = np.random.uniform(low=1, high=300)
        # ratio = 1
        # ratio = np.random.uniform(low=20, high=50)
        # , dtype=np.float32
        # print(K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio)
        return K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio


def add_color_bias(img, color_bias):  # 添加随机色偏
    channel = img.shape[2]
    img = img + color_bias.reshape((1, 1, channel))
    return img

def add_banding_noise(img, scale):  # 添加banding噪声，即论文中row noise
    channel = img.shape[2]
    img = img + \
        np.random.randn(img.shape[0], 1, channel).astype(np.float32) * scale
    return img


def add_noise(y, params=None, model='PGRU'):
        if params is None:
            K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio = _sample_params()
        else:
            K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio = params

        y = y * saturation_level
        y = y / ratio

        # noisemodel中的photon shot noise
        if 'P' in model:
            z = np.random.poisson(y / K).astype(np.float32) * K
        elif 'p' in model:
            z = y + np.random.randn(*y.shape).astype(np.float32) * \
                np.sqrt(np.maximum(K * y, 1e-10))
        else:
            z = y

        # noisemodel中的read noise
        if 'g' in model:
            z = z + np.random.randn(*y.shape).astype(np.float32) * \
                np.maximum(g_scale, 1e-10)  # Gaussian noise
        elif 'G' in model:
            z = z + stats.tukeylambda.rvs(G_shape, loc=0, scale=G_scale,
                                          size=y.shape).astype(np.float32)  # Tukey Lambda

        # noisemodel会议版本未提及此噪声, 根据代码是根据相机标定参数中的随机数随机添加色偏
        if 'B' in model:
            z = add_color_bias(z, color_bias=color_bias)

        # noisemodel中的row noise
        if 'R' in model:
            z = add_banding_noise(z, scale=R_scale)

        # noisemodel中的quantization noise
        if 'U' in model:
            z = z + np.random.uniform(low=-0.5*Q_step, high=0.5*Q_step)

        # z = z * ratio
        # z = z / saturation_level

        # post
        # z = np.clip(z, 0, 1)
        # do not adjust brightness
        # z = adjust_random_brightness(z)

        return z, saturation_level, ratio


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
    image = cv2.imread(path)
    dark_ratio=(0.2, 0.5)
    _dark = np.random.uniform(dark_ratio[0], dark_ratio[1])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image.shape
    image = cv2.resize(image, (W  * 2, H  * 2))
    image = np.array(image).astype(np.float32) / 255.  # 像素值归一化
    raw, ccm, wb = unprocess(image)
    noisy_raw, wl, ratio = add_noise(raw.copy())  # 添加噪声
    noisy_raw = noisy_raw * _dark
    
    black_level = np.array([0, 0, 0, 0], dtype=float)
    white_level = np.array([wl, wl, wl, wl], dtype=float)
    noisy_raw_ = (noisy_raw - black_level) / (white_level - black_level)
    img = torch.from_numpy(noisy_raw_).float().permute(2,0,1).unsqueeze(0)
    # img = cv2.resize(img, (1200, 800),interpolation=cv2.INTER_CUBIC)

    img = img * ratio
    # img = default_ISP(img, wb, ccm)
    img = torch.stack([img[:,0,...],
                torch.mean(img[:, [1,3], ...], dim=1),
                img[:,2,...]], dim=1)
    img = img.clip(0,1)
    # ratio = random.uniform(1.5, 2)
    torchvision.utils.save_image(img, os.path.join(out_path, file.replace('.jpg', '.jpg')))


    # raw_pattern = np.array([[0, 1],
    #                         [3, 2]], dtype=float)
    # iso = np.array([200], dtype=float)
    # exp_time = np.array([0.04], dtype=float)
    # np.savez(
    #     osp.join(save_folder, f'{img_name}.npz'),
    #     im = noisy_raw,
    #     ratio = ratio,
    #     raw_pattern = raw_pattern,
    #     black_level = black_level,
    #     white_level = white_level,
    #     wb = wb,
    #     ccm = ccm,
    #     iso = iso,
    #     exp_time = exp_time)
    process_info = f'Processing {img_name} ...'
    return process_info


os.makedirs(out_path, exist_ok=True)
os.makedirs(save_folder, exist_ok=True)


if __name__ == '__main__':
    suffix = '.jpg'
    img_list = glob(f'{data_path}/*{suffix}')
    # coco = COCO(annotation_file)

    # # 指定类别名称
    # image_id = []
    # category_names = ['bicycle', 'chair', 'dining table', 'bottle', 'motorcycle', 'car', 'tv', 'bus']  # 你想要筛选的类别
    # for category in category_names:
    #     category_ids = coco.getCatIds(catNms=category)
    #     image_ids = coco.getImgIds(catIds=category_ids)
    #     image_id = set(image_ids) | set(image_id)
    
    # img_list = []
    # # 获取图像的完整路径
    # for img_id in image_ids:
    #     img_info = coco.imgs[img_id]
    #     img_path = os.path.join(data_dir, 'train2017', img_info['file_name'])  # 图像路径
    #     img_list.append(img_path)
    # # 获取指定类别的图像 ID

    pool = Pool(50)
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    for path in img_list:
        # worker(path, out_path)
        pool.apply_async(worker, args=(path, out_path), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
        

        # im = raw_low * int((saturation - black) / uint12_max） + black
        # im = im.numpy()[0]  # 4 2251 3372
        # im = depack_raw_bayer(im, raw_pattern) 
        # H, W = im.shape # 4502 6744
        # raw.raw_image_visible[:H, :W] = im
        # raw_rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=output_bps)
        # raw_rgb_resize = cv2.resize(raw_rgb, (600, 400),interpolation=cv2.INTER_CUBIC)  # match detection labels in PASCAL
        
        
        # ## Over-Exposure Scene 
        # raw_rgb_oe = oe_light_trans(raw_rgb_resize_t)
        # torchvision.utils.save_image(raw_rgb_oe, os.path.join(out_path_oe, file.replace('.nef', '.png')))
        

