# Copyright (c) OpenMMLab. All rights reserved.
from ast import Mod
import warnings
import os
import cv2
import torch
import torchvision
import torch.nn as nn
from typing import List, Tuple, Union
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from .FeaEnhancer.feat_enhancer import FeatEnHancer
from mmdet.registry import MODELS

from ..layers import ResLayer
from .LIS.CustomConv import *
# Input-level Adapter
from .LOD_Adapter.input_adapter import Input_level_Adapeter
from .RAW_Adapter.model_adapter import Model_level_Adapeter, Merge_block
# Model-level Adapter
from mmdet.models.backbones.LED_Adapter.utils import cam_process, color_transform, default_ISP, gamma_expansion, default_ISP_3c, cosine_similarity, apply_wb

from .LED_Adapter.noise_predictor import Params_Predictor,  LUT3D,  Color_Level_Process_v2, Matrix_Predictor_v2, Matrix_Predictor_v3, Color_Level_Process_v3, Weight_Merge, Matrix_Predictor_v1_5, Matrix_Predictor_v4, Color_Level_Process_v4, Matrix_Predictor_v5
from mmdet.models.backbones.LED_Adapter.model_adapter import Attention_Merge, Channel_Spatial_Adapter

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
class LOD_ResNet(BaseModule):

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
                 light_mode='normal',
                 w_lut=False,   # with or without 3DLUT
                 lut_dim=32,
                 k_size=3,  # LOD dataset, 9
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
                 ISP_version = 'v1'):
        super(LOD_ResNet, self).__init__(init_cfg)
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
        self.version = ISP_version       
        
        if self.version == 'v1':
            self.params_predictor = Params_Predictor(dim=64, attn_drop=0.2, proj_drop=0.2, head_drop=0.2)
            self.gamma = nn.Parameter(torch.FloatTensor([2.2]), requires_grad=True)
            self.color_mapper = LUT3D(n_colors=3, n_vertices=17, n_feats=64, n_ranks=3)
        elif self.version == 'v2':
            self.gamma = nn.Parameter(torch.FloatTensor([2.2]), requires_grad=True)
            self.params_predictor = Matrix_Predictor_v2(attn_drop=0.1, proj_drop=0.1, head_drop=0.2, data='LOD')
            self.nonlinear = Color_Level_Process_v3()
            # self.nonlinear2 = Color_Level_Process_v4()
        elif self.version == 'v3':
            self.params_predictor = Matrix_Predictor_v2(attn_drop=0.1, proj_drop=0.1, head_drop=0.2)
            self.nonlinear = Color_Level_Process_v3()
        elif self.version == 'v4':
            self.params_predictor = Matrix_Predictor_v4(attn_drop=0.1, proj_drop=0.1, head_drop=0.2)
            self.color_mapper = LUT3D(n_colors=3, n_vertices=17, n_feats=64, n_ranks=3)
            self.gamma = nn.Parameter(torch.FloatTensor([2.2]), requires_grad=True)
            self.nonlinear = Color_Level_Process_v3()


        self.w_lut = w_lut  # with or without implicit 3D LUT
        self.sigmoid = nn.Sigmoid()
        self.pre_encoder = Input_level_Adapeter(mode = light_mode, lut_dim = lut_dim, k_size=k_size, w_lut=self.w_lut)    
        
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
        
        self._make_stem_layer(in_channels, stem_channels)

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

    def forward(self, x, metainfo):
        B, C, H, W = x.shape
        x_i = x.clip(0,1)
        if self.version == 'v0':
            x_l, x_g = default_ISP(x_i, metainfo)
            loss_mat = torch.zeros_like(x_l)
        elif self.version == 'v2':
            mat = self.whittened_mat(metainfo)
            dmat, glo, local = self.params_predictor(x_i, mat)
            x_g, mat = color_transform(x_i, mat, dmat=dmat)
            # x_g = torch.stack([x_i[:,0,...],
            #     torch.mean(x_i[:, [1,3], ...], dim=1),
            #     x_i[:,2,...]], dim=1)
            x_g = x_g.clip(0, 1)
            # x_l = x_g
            x_l = self.nonlinear(x_g) 
            # x_l = self.nonlinear2(x_g)
            # x_l = self.nonlinear3(x_g)
            # loss_mat = self.mat_loss(x_i, x_l, mat)
            loss_mat = self.mat_loss4(x_i, x_l, mat)
            # loss_mat = torch.zeros_like(x_l)
        elif self.version == 'v3':
            x_l, x_g = default_ISP(x_i, metainfo)
            mat = self.whittened_mat(metainfo)
            dmat = self.params_predictor(x_i, mat) 
            zero = torch.zeros_like(mat)
            dx_g, dmat_ = color_transform(x_i, zero, dmat=dmat)
            x_g, mat = color_transform(x_i, mat, dmat=dmat)
            dx_l = self.nonlinear(dx_g) 
            x_l = (x_l + dx_l).clip(0, 1)
            # loss_mat = self.mat_loss(x_i, x_l-x_g, dmat_)
            loss_mat = self.mat_loss(x_i, x_l, mat)
        elif self.version == 'v4':
            x_l, x_g = default_ISP(x_i, metainfo)
            mat = self.whittened_mat(metainfo)
            dmat = self.params_predictor(x_i, mat)
            x_g, mat = color_transform(x_i, mat, dmat=dmat)
            dx_l = self.nonlinear(x_g) 
            # x_l = (x_g + dx_l).clip(0, 1)
            loss_mat = self.mat_loss(x_i, x_l, mat)
        elif self.version == 'v-1':
            x_l = torch.stack([x_i[:,0,...],
                torch.mean(x_i[:, [1,3], ...], dim=1),
                x_i[:,2,...]], dim=1)
            # x_l = x_l.detach()
            x_l = x_l.clip(0, 1)
            loss_mat = torch.zeros_like(x_l)
        x_l = self.pre_encoder(x_l) # Input-level Adapter
        x_l = x_l[-1]  # 4 3 416 608
        # x_l, c_map, spatial_map= self.augmentor([x_l, x_g, x_i])
        if self.model == "Enhancer":
            x_l = self.enhancer(x_l)
        elif self.model == "Raw-adapter":
            x = self.pre_encoder(x_l) # Input-level Adapter
            # ada = self.model_adapter([x[0], x[1], x[2], x[3]]) # 1 24 104 152
            x_l = x[-1]  # 4 3 416 608
        """Forward function."""
        if self.deep_stem:  # False
            x = self.stem(x_l)
        else:               # True
            x = self.conv1(x_l) # 4 64 208 304
            x = self.norm1(x)
            x = self.relu(x)
        if self.model != "LIS":
            x = self.maxpool(x)       # 1 64 208 304
        #print('input shape', x.shape)
        outs = [] 
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if self.model == "LIS":
                x = self.AdaD[i](x)
            x = res_layer(x)
            # if i <=2:  
            #     x, c_map, spatial_map = self.merge_blocks[i](x, c_map, spatial_map)
            if i in self.out_indices:   # self.out_indices (0,1,2,3)
                outs.append(x) # 形状和 merge 前一样
        # loss_mat = 0
        return tuple(outs), loss_mat
    
    def mat_loss4(self, x_i, x_l, P):
        b, c, h, w = x_i.shape
        x_i = x_i.view(b, c, -1) # [4 4 hw]
        # 重塑张量
        x_l = x_l.view(b, 3, -1)  # 大小为 [4, 3, hw]
        x_iT = x_i.permute(0, 2, 1)  # 大小为 [4, hw, 4]
        # 进行批量矩阵乘法
        result1 = torch.bmm(x_l, x_iT)  # 大小为 [4, 3, 4]
        result2 = torch.bmm(x_i, x_iT)  # 大小为 [4, 4, 4]
        det = torch.linalg.det(result2) # 求行列式
        if torch.any(det == 0):
            raise ValueError("Has zeor determinant")
        # inverse_tensor = []
        # for i in range(result2.shape[0]):
        #     if det[i] == 0:
        #         # 使用伪逆
        #         inverse_tensor.append(torch.linalg.pinv(result2[i]))
        #     else:
        #         # 使用正常的逆
        #         inverse_tensor.append(torch.linalg.inv(result2[i]))
        # result2 = torch.stack(inverse_tensor)
        result2 = torch.linalg.inv(result2)
        A = torch.bmm(result1, result2)
        P = torch.mean(P, dim=(1, 2))
        # 计算 tensorA 和 tensorB 在最后两个维度的余弦相似度
        # tensorA: [4, 3, 4] 
        # tensorB: [4, h, w, 3, 4]
        B, mh, mw = A.shape  
        P = P.view(B, mh, mw)
        # 首先计算点积
        dot_product = (A * P).sum(dim=2)  # 形状为 [4, 3]

        # 计算范数
        norm1 = torch.norm(P, dim=-1) + 1e-5  # 形状为 [4, 3]
        norm2 = torch.norm(A, dim=-1) + 1e-5 # 形状为 [4, 3]

        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)
        loss_mat = (1 - similarity).sum(dim=-1).mean()

        return loss_mat * 0.05

    def mat_loss(self, x_i, x_l, P):
        b, c, h, w = x_i.shape
        x_i = x_i.view(b, c, -1) # [4 4 hw]

        # 重塑张量
        x_l = x_l.view(b, 3, -1)  # 大小为 [4, 3, hw]
        x_iT = x_i.permute(0, 2, 1)  # 大小为 [4, hw, 4]

        # 进行批量矩阵乘法
        result1 = torch.bmm(x_l, x_iT)  # 大小为 [4, 3, 4]
        result2 = torch.bmm(x_i, x_iT)  # 大小为 [4, 4, 4]
        # det = torch.linalg.det(result2) # 求行列式
        # inverse_tensor = []
        # for i in range(result2.shape[0]):
        #     if det[i] == 0:
        #         # 使用伪逆
        #         inverse_tensor.append(torch.linalg.pinv(result2[i]))
        #     else:
        #         # 使用正常的逆
        #         inverse_tensor.append(torch.linalg.inv(result2[i]))
        # result2 = torch.stack(inverse_tensor)
        result2 = torch.linalg.pinv(result2)
        A = torch.bmm(result1, result2)
        A = A.unsqueeze(1).unsqueeze(2).repeat(1, h, w, 1, 1)

        return cosine_similarity(P, A) * 0.01

    def whittened_mat(self, metainfo, increment: List = None):
        """ 
        Args : 
            metainfo: Dictionary of parameters.
            increment: A list of increment for ccm r1 b1 (dccm, dr1, db1)
            weights: A list of weights for increments (wccm, wr1, wb1)
        Returns :
            enhanced_representation : enhanced aggregated feature representations of
            three feature maps from different scales.enhanced features in (B, 1, H, W)
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

    def learnable_ISP(self, x_i, metainfo):
        white_level = metainfo.pop('white_level')
        black_level = metainfo.pop('black_level')
        B, C, H, W = x_i.shape
        r1 = metainfo['wb'][:,0].unsqueeze(1)
        b1 = metainfo['wb'][:,2].unsqueeze(1)
        ccm = metainfo['ccm'].view(B, -1)
        q = torch.cat([r1, b1, ccm], dim=1)
        r1, b1, ccm = self.params_predictor(x_i, q)
        params = {}
        params['r1'] = r1
        params['b1'] = b1
        params['ccm'] = ccm
        metainfo = self.add_gain(params, metainfo)
        # remove darkshading
        x_g = cam_process(x_i, metainfo)
        x_l = self.learnable_process(x_g.permute(0, 2, 3, 1), **metainfo).permute(0,3,1,2)
        return x_l, x_g
    
    def add_gain(self, params, metainfo):
        # wb = metainfo['wb']
        # ccm = metainfo['ccm']
        metainfo['wb'][:,0] += params['r1'].squeeze() * 0.1
        metainfo['wb'][:,2] += params['b1'].squeeze() * 0.1
        # params['wb'] = wb
        metainfo['ccm'] += params['ccm'] * 0.1
        return metainfo
    
    def learnable_process(self, x, iso, wb, exp_time, ccm, scale=100.0):
        B, H, W, C = x.shape
        x = gamma_expansion(x, self.gamma)
        ccm = ccm.view(B, -1)
        level = iso * exp_time / scale
        params = torch.cat([level, wb[:,0].unsqueeze(1), 
                        wb[:,2].unsqueeze(1), ccm], dim=1).view(B, 1, 1, 3+9)
        params = self.sigmoid(params.expand(-1, H, W, -1))
        color_input = torch.cat([x, params], dim=3)
        out = self.color_mapper(color_input)
        return torch.clip(out, 0., 1.0)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(LOD_ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()