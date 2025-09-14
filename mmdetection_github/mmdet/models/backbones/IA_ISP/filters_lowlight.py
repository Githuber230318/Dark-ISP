import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2


def lrelu(x, leak=0.2, name=None):
    """PyTorch 版本的 LeakyReLU（与 TensorFlow 实现完全一致）
    
    参数:
        x: 输入张量
        leak: 负半轴的斜率 (默认 0.2)
        name: 兼容性参数（PyTorch 中无实际作用）
    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)

def tanh_range(l, r, initial=None):
    """PyTorch 版本的 tanh 范围映射（与 TensorFlow 实现完全一致）
    
    参数:
        l: 输出下限
        r: 输出上限
        initial: 初始偏置值
    """
    def tanh01(x):
        return torch.tanh(x) * 0.5 + 0.5  # 将 tanh 输出映射到 [0, 1]

    def activation(x):
        if initial is not None:
            bias = math.atanh(2 * (initial - l) / (r - l) - 1)
        else:
            bias = 0
        return tanh01(x + bias) * (r - l) + l

    return activation

def lerp(a, b, l):
    """线性插值函数 (Linear Interpolation)
    
    参数:
        a: 起始值 (张量或标量)
        b: 结束值 (张量或标量)
        l: 插值系数 [0,1] (张量或标量)
    返回:
        (1 - l) * a + l * b
    """
    return (1 - l) * a + l * b

def rgb2lum(image):
    """将 RGB 图像转换为亮度通道 (CIE 1931 标准)
    
    参数:
        image: 输入图像张量 [B, H, W, C] 或 [H, W, C] (RGB 顺序)
    返回:
        亮度张量 [B, H, W, 1] 或 [H, W, 1]
    """
    # 按 CIE 1931 标准计算亮度
    lum = 0.27 * image[:, 0, ...] + 0.67 * image[:, 1, ...] + 0.06 * image[:, 2, ...]
    return lum.unsqueeze(1)  # 添加通道维度


class Filter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.short_name = None
        self.begin_filter_parameter = 0
        self.num_filter_parameters = 0

    def forward(self, img, img_features=None, specified_parameter=None):
        if img_features is not None:
            features = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(features)
        else:
            filter_parameters = specified_parameter
        
        # 处理图像
        output = self.process(img, filter_parameters)
        return output, filter_parameters

    def extract_parameters(self, features):
        start = self.begin_filter_parameter
        end = start + self.num_filter_parameters
        return features[:, start:end]

class ExposureFilter(Filter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'E'
        self.begin_filter_parameter = cfg.exposure_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(-self.cfg.exposure_range, self.cfg.exposure_range, initial=0)(features)

    def process(self, img, param):
        # img: [B, C, H, W], param: [B, 1]
        return img * torch.exp(param.view(-1, 1, 1, 1) * math.log(2))
    

class UsmFilter(Filter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1
        self.blur_kernel = self._make_gaussian_kernel(5)

    def _make_gaussian_kernel(self, sigma):
        radius = 12
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma)**2)
        k = k / k.sum()
        return (k[None, :] * k[:, None]).view(1, 1, 2*radius+1, 2*radius+1)

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param):
        # 高斯模糊
        pad = (25 - 1) // 2
        padded = F.pad(img, [pad]*4, mode='reflect')
        blurred = F.conv2d(padded, self.blur_kernel.repeat(3,1,1,1).to(padded.device), stride=1, padding=0, groups=3)
        # USM处理
        usm = (img - blurred) * param.view(-1, 1, 1, 1) + img
        return usm


class GammaFilter(Filter):
    def __init__(self, cfg):
        Filter.__init__(self, cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1
    def filter_param_regressor(self, features):
        log_gamma_range = math.log(self.cfg.gamma_range)
        return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def process(self, img, param):
        param = param.repeat(1, 3)  # 扩展到3通道
        return torch.clamp(img, min=0.001).pow(param.view(-1, 3, 1, 1))

class ImprovedWhiteBalanceFilter(Filter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'W'
        self.channels = 3
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = self.channels

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor([[0, 1, 1]], dtype=torch.float32)
        features = features * mask.to(features.device)
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))
        # 亮度归一化
        luminance = 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] + 0.06 * color_scaling[:, 2]
        return color_scaling / (luminance.view(-1, 1) + 1e-5)
    
    def process(self, img, param):
        """应用白平衡
        输入: 
            img: [B, C, H, W] 
            param: [B, 3]
        输出: [B, C, H, W]
        """
        return img * param.unsqueeze(2).unsqueeze(3)  # 广播乘

class ToneFilter(Filter):
    def __init__(self, cfg):
        """色调曲线滤镜 (PyTorch 版)
        
        参数:
            net: 主干网络
            cfg: 配置对象，需包含:
                - curve_steps: 曲线分段数
                - tone_begin_param: 参数起始索引
                - tone_curve_range: 曲线数值范围 (min, max)
        """
        super().__init__(cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param
        self.num_filter_parameters = cfg.curve_steps
        self.tone_range = cfg.tone_curve_range

    def filter_param_regressor(self, features):
        """生成色调曲线参数"""
        # 重塑为 [B, 1, 1, curve_steps] 并应用 tanh 范围限制
        tone_curve = features.view(-1, 1, self.curve_steps).unsqueeze(1)
        return tanh_range(*self.tone_range)(tone_curve)

    def process(self, img, param):
        """应用色调曲线变换
        参数:
            img: 输入图像 [B, C, H, W]
            param: 曲线参数 [B, 1, 1, curve_steps]
        返回:
            处理后的图像 [B, C, H, W]
        """
        # 确保输入在 [0,1] 范围内
        img = torch.clamp(img, 0, 1.0)
        
        # 计算归一化系数
        tone_curve_sum = param.sum(dim=-1, keepdim=True) + 1e-30
        
        # 分段处理色调曲线
        total_image = torch.zeros_like(img)
        for i in range(self.curve_steps):
            band = torch.clamp(img - i / self.curve_steps, 0, 1.0 / self.curve_steps)
            total_image += band * param[:, :, :, i].unsqueeze(1)
        
        # 归一化并缩放
        return total_image * (self.curve_steps / tone_curve_sum)

class ContrastFilter(Filter):
    def __init__(self, cfg):
        """对比度滤镜 (PyTorch 版)
        
        参数:
            net: 主干网络
            cfg: 配置对象，需包含:
                - contrast_begin_param: 参数起始索引
                - contrast_range: 可选参数范围 (未在原始代码中使用)
        """
        super().__init__(cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        """生成对比度参数 (使用 tanh 激活)"""
        return torch.tanh(features)  # 原始代码未使用 sigmoid/range 限制

    def process(self, img, param):
        """应用对比度调整
        参数:
            img: 输入图像 [B, C, H, W] (RGB)
            param: 调整参数 [B, 1]
        返回:
            处理后的图像 [B, C, H, W]
        """
        # 计算亮度并限制范围
        luminance = torch.clamp(rgb2lum(img), 0.0, 1.0)  # [B, 1, H, W]
        
        # 对比度曲线计算 (余弦变换)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        
        # 保持色彩比例
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        
        # 线性插值混合
        return lerp(img, contrast_image, param.unsqueeze(-1).unsqueeze(-1))

