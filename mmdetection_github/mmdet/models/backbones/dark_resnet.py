# Copyright (c) OpenMMLab. All rights reserved.
from ast import Mod
import warnings
import os
import cv2
import time
import datetime
import torchvision
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Union
import torch.utils.checkpoint as cp
from .LOD_Adapter.input_adapter import Input_level_Adapeter
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from .FeaEnhancer.feat_enhancer import FeatEnHancer
from .RAOD_Adapter.RAOD import Adaptive_Module
from .IA_ISP.IA_ISP import ImageProcessor
from .IA_ISP.config_lowlight import cfg
from mmdet.registry import MODELS

from ..layers import ResLayer
from .LIS.CustomConv import *
from .RAW_Adapter.input_adapter import Input_level_Adapeter
# Model-level Adapter
from .RAW_Adapter.model_adapter import Model_level_Adapeter, Merge_block
# Input-level Adapter
# Model-level Adapter
from .Dark_modules.utils import color_transform, default_ISP, cosine_similarity
from .Dark_modules.lut import lut_transform
from .Dark_modules.noise_predictor import LUT3D,  Matrix_Predictor, Color_Level_Process, Zero_DCE, NILUT


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@MODELS.register_module()
class Dark_ResNet(BaseModule):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 w_lut=False,   # with or without 3DLUT
                 fea_c_s = [256, 512, 1024],
                 merge_ratio=1.0,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 model=None,
                 is_removedk=True,
                 ISP_version = 'v1'):
        super(Dark_ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.w_lut = w_lut
        self.is_removedk = is_removedk
        self.version = ISP_version
        self.color_mapper = LUT3D(n_colors=3, n_vertices=17, n_ranks=3)
        self.gamma = nn.Parameter(torch.FloatTensor([2.2]), requires_grad=True)
        self.params_predictor = Matrix_Predictor(attn_drop=0.1, proj_drop=0.1, head_drop=0.2, data='LOD')
        self.nonlinear = Color_Level_Process()
        self.mlp = NILUT()
        self.nonlinear3 = Zero_DCE()
        
        self.model = model
        if self.model == "Raw-adapter":
            self.pre_encoder = Input_level_Adapeter(mode = dict(type='low'), lut_dim = 32, k_size=3, w_lut=True)    
            self.model_adapter = Model_level_Adapeter(in_c=3, in_dim=12, w_lut=True)
            self.merge_1 = Merge_block(fea_c=64, ada_c=12, mid_c=32, return_ada=True)
            self.merge_2 = Merge_block(fea_c=128, ada_c=24, mid_c=32, return_ada=True)
            self.merge_3 = Merge_block(fea_c=256, ada_c=48, mid_c=64, return_ada=False)
            self.merge_blocks = [self.merge_1, self.merge_2, self.merge_3]
            self.merge_ratio = merge_ratio  # Feature Merge Ratio
        elif self.model == "Enhancer":
            self.enhancer = FeatEnHancer(in_channels)
        elif self.model == "LIS":
            self.AdaD = nn.ModuleList([
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    # nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
                    # nn.MaxPool2d(kernel_size=2, stride=1),
                    # AdaDConv(in_channels=64, kernel_size=3, stride=2, groups=64),
                ),
                AdaDConv(in_channels=fea_c_s[0], kernel_size=3, stride=1, groups=fea_c_s[0]),
                AdaDConv(in_channels=fea_c_s[1], kernel_size=3, stride=1, groups=fea_c_s[1]),
                AdaDConv(in_channels=fea_c_s[2], kernel_size=3, stride=1, groups=fea_c_s[2]),
            ])
        elif self.model == "RAOD":
            self.pre_processor = Adaptive_Module(in_ch=3, nf=16, gamma_range=[1.,4.])
        elif self.model == "IA_ISP":
            self.pre_processor = ImageProcessor(cfg)
        
        self._make_stem_layer(in_channels, stem_channels)
        self.fea_c_s = fea_c_s
        
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)
        self.norm = nn.BatchNorm1d(num_features=(4+9))

    def make_stage_plugins(self, plugins, stage_idx):
        
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def joint_mat(self, metainfo, increment: List = None):
        """ 
        Args : 
            metainfo: Dictionary of parameters.
            increment: A list of increment for ccm r1 b1 (dccm, dr1, db1)
        Returns :
            joint parameter matrix
        """
        ccm = metainfo['ccm']
        second = ccm[:,:,1].unsqueeze(-1) 
        ccm = torch.cat((ccm, second), dim=2)
        wb = metainfo['wb'].unsqueeze(1)
        if increment is not None:
            dccm = increment[0]
            second = dccm[:,:,1].unsqueeze(-1) 
            dccm = torch.cat((dccm, second), dim=2)
            dr1 = increment[1]
            db1 = increment[2]
            ccm += dccm
            wb[:,:,0] += dr1
            wb[:,:,2] += db1
        wb[:,:,1] *= 0.5
        wb[:,:,3] *= 0.5
        a = wb * ccm
        return a

    def forward(self, x, metainfo, cur_epoch=None, batch_data_samples=None):
        B, C, H, W = x.shape
        x_i = x.clip(0, 1)
        if self.version == 'v0':  # Default ISP
            x_l, x_g = default_ISP(x_i, metainfo)
            loss_mat = torch.zeros_like(x_l)
            x_l = x_l.clip(0, 1)
            x_l = x_l.detach()
        elif self.version == 'v1': # Demosaic
            x_l = torch.stack([x_i[:,0,...],
                torch.mean(x_i[:, [1,3], ...], dim=1),
                x_i[:,2,...]], dim=1)
            x_l = x_l.clip(0, 1)
            loss_mat = torch.zeros_like(x_l)
        elif self.version == 'v2': # Dark-ISP
            mat = self.joint_mat(metainfo)
            dmat, glo, local = self.params_predictor(x_i, mat)
            x_g, mat = color_transform(x_i, mat, dmat=dmat)
            x_g = x_g.clip(0, 1)
            x_l = self.nonlinear(x_g)
            if self.training: 
                loss_mat = self.mat_loss2(x_i, x_l, mat)
                 # loss_mat = self.mat_loss(x_i, x_l, mat)
            else:
                loss_mat = torch.zeros_like(x_l)
            # loss_mat = torch.zeros_like(x_l)  # For Inference
        
        if self.model == "Enhancer":
            x_l = self.enhancer(x_l)
        elif self.model == "Raw-adapter":
            x = self.pre_encoder(x_l) # Input-level Adapter
            # ada = self.model_adapter([x[0], x[1], x[2], x[3]]) # 1 24 104 152
            x_l = x[-1]  # 4 3 416 608
        elif self.model == "RAOD":
            x_l = self.pre_processor(x_l)
        elif self.model == "IA_ISP":
            x_l = self.pre_processor(x_l)
        """Forward function."""
        if self.deep_stem:  # False
            x = self.stem(x_l)
        else:               # True
            x = self.conv1(x_l)   
            x = self.norm1(x)   
            x = self.relu(x)
        if self.model != "LIS":
            x = self.maxpool(x)      
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if self.model == "LIS":
                x = self.AdaD[i](x)
            x = res_layer(x)                # (104 152)  (52 76) (26 38) (13 19)
            # if i <=2:  
            #     if self.model == "Raw-adapter":
            #         x, ada = self.merge_blocks[i](x, ada, ratio=self.merge_ratio)
            if i in self.out_indices:   # self.out_indices (0,1,2,3)
                outs.append(x)
        return tuple(outs), loss_mat


    def mat_loss2(self, x_i, x_l, P):
        """
        Parameters:
        x_i: Iutput of Linear Component
        x_l: Output of Nonlinear Component
        P: Matrix to be optimized, shape [4, h, w, 3, 4]
        """
        b, c, h, w = x_i.shape
        x_i = x_i.view(b, c, -1) # Shape:[4 4 hw]
        # Reshape tensor
        x_l = x_l.view(b, 3, -1)  # Shape: [4, 3, hw]
        x_iT = x_i.permute(0, 2, 1)  # Shape: [4, hw, 4]
        # Perform batch matrix multiplication
        result1 = torch.bmm(x_l, x_iT)  # Shape: [4, 3, 4]
        result2 = torch.bmm(x_i, x_iT)  # Shape: [4, 4, 4]
        det = torch.linalg.det(result2) # Determinant
        if torch.any(det == 0):
            raise ValueError("Has zeor determinant")
        result2 = torch.linalg.inv(result2)
        A = torch.bmm(result1, result2)
        P = torch.mean(P, dim=(1, 2))
        # A shape: [4, 3, 4] 
        # P shape: [4, h, w, 3, 4]
        B, mh, mw = A.shape  
        P = P.view(B, mh, mw)
        # Calculate dot product
        dot_product = (A * P).sum(dim=2)  # Shape: [4, 3]

        # Calculate Norm
        norm1 = torch.norm(P, dim=-1) + 1e-5  # Shape: [4, 3]
        norm2 = torch.norm(A, dim=-1) + 1e-5 # Shape: [4, 3]

        # Calculate coscine similarity
        similarity = dot_product / (norm1 * norm2)
        loss_mat = (1 - similarity).sum(dim=-1).mean()

        return loss_mat * 0.05

    def mat_loss(self, x_i, x_l, P):
        b, c, h, w = x_i.shape
        x_i = x_i.view(b, c, -1) # Shape: [4 4 hw]
        # Reshape
        x_l = x_l.view(b, 3, -1)  # Shape: [4, 3, hw]
        x_iT = x_i.permute(0, 2, 1)  # Shape: [4, hw, 4]
        # Batch matrix multiply
        result1 = torch.bmm(x_l, x_iT)  # Shape: [4, 3, 4]
        result2 = torch.bmm(x_i, x_iT)  # Shape: [4, 4, 4]
        det = torch.linalg.det(result2) # Determinant
        if torch.any(det == 0):
            raise ValueError("Has zeor determinant")
        result2 = torch.linalg.inv(result2)
        A = torch.bmm(result1, result2)
        A = A.unsqueeze(1).unsqueeze(2).repeat(1, h, w, 1, 1)

        return cosine_similarity(P, A) * 0.01

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(Dark_ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
