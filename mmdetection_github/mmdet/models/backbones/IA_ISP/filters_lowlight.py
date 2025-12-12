import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2


def lrelu(x, leak=0.2, name=None):
    """
    
    Parameters:
        x: Input tensor
        leak: Negative half-axis slope (default 0.2)
        name: Compatibility parameter (no actual function in PyTorch)
    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)

def tanh_range(l, r, initial=None):
    """
    
    Parameters:
        l: Output lower limit
        r: Output upper limit
        initial: Initial bias value
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
    """Linear Interpolation
    
    Parameters:
        a: Starting value (tensor or scalar)
        b: Ending value (tensor or scalar)
        l: Interpolation coefficient [0,1] (tensor or scalar)
    Returns:
        (1 - l) * a + l * b
    """
    return (1 - l) * a + l * b

def rgb2lum(image):
    """Convert RGB image to luminance channel (CIE 1931 standard)
    
    Parameters:
        image: Input image tensor [B, H, W, C] or [H, W, C] (RGB order)
    Returns:
        Luminance tensor [B, H, W, 1] or [H, W, 1]
    """
    # Calculate luminance according to CIE 1931 standard
    lum = 0.27 * image[:, 0, ...] + 0.67 * image[:, 1, ...] + 0.06 * image[:, 2, ...]
    return lum.unsqueeze(1)  # Add channel dimension


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
        
        # Process image
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
        # Gaussian blur
        pad = (25 - 1) // 2
        padded = F.pad(img, [pad]*4, mode='reflect')
        blurred = F.conv2d(padded, self.blur_kernel.repeat(3,1,1,1).to(padded.device), stride=1, padding=0, groups=3)
        # USM processing
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
        param = param.repeat(1, 3)  # Expand to 3 channels
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
        # Luminance normalization
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
        """Tone curve filter 
        
        Parameters:
            net: Main network
            cfg: Configuration object, must contain:
                - curve_steps: Number of curve segments
                - tone_begin_param: Parameter starting index
                - tone_curve_range: Curve value range (min, max)
        """
        super().__init__(cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param
        self.num_filter_parameters = cfg.curve_steps
        self.tone_range = cfg.tone_curve_range

    def filter_param_regressor(self, features):
        """Generate tone curve parameters"""
        # Reshape to [B, 1, 1, curve_steps] and apply tanh range limit
        tone_curve = features.view(-1, 1, self.curve_steps).unsqueeze(1)
        return tanh_range(*self.tone_range)(tone_curve)

    def process(self, img, param):
        """Apply tone curve transformation
        Parameters:
            img: Input image [B, C, H, W]
            param: Curve parameters [B, 1, 1, curve_steps]
        Returns:
            Processed image [B, C, H, W]
        """
        # Ensure input is in [0,1] range
        img = torch.clamp(img, 0, 1.0)
        
        # Calculate normalization coefficient
        tone_curve_sum = param.sum(dim=-1, keepdim=True) + 1e-30
        
        # Process tone curve in segments
        total_image = torch.zeros_like(img)
        for i in range(self.curve_steps):
            band = torch.clamp(img - i / self.curve_steps, 0, 1.0 / self.curve_steps)
            total_image += band * param[:, :, :, i].unsqueeze(1)
        
        # Normalize and scale
        return total_image * (self.curve_steps / tone_curve_sum)

class ContrastFilter(Filter):
    def __init__(self, cfg):
        """Contrast filter
        
        Parameters:
            net: Main network
            cfg: Configuration object, must contain:
                - contrast_begin_param: Parameter starting index
                - contrast_range: Optional parameter range (not used in original code)
        """
        super().__init__(cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        """Generate contrast parameters (using tanh activation)"""
        return torch.tanh(features) 

    def process(self, img, param):
        """Apply contrast adjustment
        Parameters:
            img: Input image [B, C, H, W] (RGB)
            param: Adjustment parameters [B, 1]
        Returns:
            Processed image [B, C, H, W]
        """
        # Calculate luminance and limit range
        luminance = torch.clamp(rgb2lum(img), 0.0, 1.0)  # [B, 1, H, W]
        
        # Calculate contrast curve (cosine transformation)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        
        # Preserve color ratio
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        
        # Linear interpolation mixing
        return lerp(img, contrast_image, param.unsqueeze(-1).unsqueeze(-1))

