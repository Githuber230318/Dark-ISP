# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import yaml
import warnings
from typing import List, Tuple, Union
import numpy as np
import torch
from torch import Tensor

from mmdet.registry import MODELS
from collections import OrderedDict
from mmengine.optim import OptimWrapper
from mmengine.utils import is_list_of
from mmdet.structures import SampleList
from typing import Dict, Optional, Tuple, Union
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
from .two_stage import TwoStageDetector
from mmdet.models.backbones.LED_Adapter.utils import normlize, default_ISP, default_ISP_3c

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])

@MODELS.register_module()
class TwoStageRAWDetector(TwoStageDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 cri_pix: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor, 
            init_cfg=init_cfg)
        
        self.encoder_layer_grads = {}
        self.decoder_layer_grads = {}
        if cri_pix is not None:
            self.cri_pix = MODELS.build(cri_pix)

    def extract_feat(self, batch_inputs: Tensor,
                     metainfo: SampleList) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x, loss_mat = self.backbone(batch_inputs, metainfo)
        if self.with_neck:
            x = self.neck(x)
        if self.training:
            return x, loss_mat
        else:
            return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             gts: Tensor = None) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        # self.register_hooks()
        # process metadata batch
        device = batch_inputs.device
        metainfo = [batch_data_samples[i].metainfo for i in range(len(batch_data_samples))]
        metainfo = self.dict_to_tensor(batch_data_samples, device)

        # normalize
        B, _, _, _ = batch_inputs.shape
        ratio =  metainfo.pop('ratio').view(B,1,1,1)
        batch_inputs = normlize(batch_inputs, metainfo['white_level'], metainfo['black_level']) 
        batch_inputs = batch_inputs * ratio

        x, loss_mat = self.extract_feat(batch_inputs, metainfo)
    
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg) # 19 dao 30.ji
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        losses['mat_loss'] =[loss_mat]

        return losses
        

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # process metadata batch
        device = batch_inputs.device
        metainfo = [batch_data_samples[i].metainfo for i in range(len(batch_data_samples))]
        metainfo = self.dict_to_tensor(batch_data_samples, device)
        
         # normalize
        ratio =  metainfo.pop('ratio')
        B, _, _, _ = batch_inputs.shape
        batch_inputs = normlize(batch_inputs, metainfo['white_level'], metainfo['black_level']) * ratio.view(B,1,1,1)

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs, metainfo)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    def dict_to_tensor(self, metainfo, device):
        ccm = []
        wb = []
        iso = []
        white_level = []
        black_level = []
        exp_time = []
        ratio = []
        for i in range(len(metainfo)):
            ccm.append(metainfo[i].metainfo.pop('ccm'))
            wb.append(metainfo[i].metainfo.pop('wb'))
            iso.append(metainfo[i].metainfo.pop('iso'))
            white_level.append(metainfo[i].metainfo.pop('white_level'))
            black_level.append(metainfo[i].metainfo.pop('black_level'))
            exp_time.append(metainfo[i].metainfo.pop('exp_time'))
            ratio.append(metainfo[i].metainfo.pop('ratio'))
        ccm = torch.stack(ccm)
        wb = torch.stack(wb)
        white_level = torch.stack(white_level)
        black_level = torch.stack(black_level)
        iso = torch.stack(iso)
        exp_time = torch.stack(exp_time)
        ratio = torch.stack(ratio)
        params = {}
        params['ccm'] = ccm.to(device)
        params['wb'] = wb.to(device)
        params['iso'] = iso.to(device)
        params['white_level'] = white_level.to(device)
        params['black_level'] = black_level.to(device)
        params['exp_time'] = exp_time.to(device)
        params['ratio'] = ratio.to(device)

        return params

    def register_hooks(self):
        # 注册 hooks 来获取每层的输出和梯度
        for name, layer in self.backbone.denoiser.encoder.named_children():
            layer.register_backward_hook(self.get_encoder_grad_hook(name))
        for name, layer in self.backbone.denoiser.decoder.named_children():
            layer.register_backward_hook(self.get_decoder_grad_hook(name))

    def get_decoder_grad_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.decoder_layer_grads[name] = grad_output[0]  # 只取第一个梯度
        return hook
    
    def get_encoder_grad_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.encoder_layer_grads[name] = grad_output[0]  # 只取第一个梯度
        return hook

    # def train_step(self, data: Union[dict, tuple, list],
    #                optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
    #     """Implements the default model training process including
    #     preprocessing, model forward propagation, loss calculation,
    #     optimization, and back-propagation.

    #     During non-distributed training. If subclasses do not override the
    #     :meth:`train_step`, :class:`EpochBasedTrainLoop` or
    #     :class:`IterBasedTrainLoop` will call this method to update model
    #     parameters. The default parameter update process is as follows:

    #     1. Calls ``self.data_processor(data, training=False)`` to collect
    #        batch_inputs and corresponding data_samples(labels).
    #     2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
    #        loss
    #     3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
    #        backward and dict of loss tensor used to log messages.
    #     4. Calls ``optim_wrapper.update_params(loss)`` to update model.

    #     Args:
    #         data (dict or tuple or list): Data sampled from dataset.
    #         optim_wrapper (OptimWrapper): OptimWrapper instance
    #             used to update model parameters.

    #     Returns:
    #         Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
    #     """
    #     # Enable automatic mixed precision training context.
    #     with optim_wrapper.optim_context(self):
    #         data = self.data_preprocessor(data, True)
    #         detect_losses, denoise_loss = self._run_forward(data, mode='loss')  # type: ignore
    #     detect_losses, log_vars = self.parse_losses(detect_losses, denoise_loss)  # type: ignore
    #     # optim_wrapper.update_params(detect_losses+denoise_loss)
    #     optim_wrapper.update_twoloss(detect_losses, denoise_loss, 
    #                                 [self.backbone.denoiser.encoder, self.backbone.denoiser.decoder])
    #     return log_vars
    
    # def parse_losses(
    #     self, detect_losses: Dict[str, torch.Tensor],
    #     denoise_loss
    # ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    #     """Parses the raw outputs (losses) of the network.

    #     Args:
    #         losses (dict): Raw output of the network, which usually contain
    #             losses and other necessary information.

    #     Returns:
    #         tuple[Tensor, dict]: There are two elements. The first is the
    #         loss tensor passed to optim_wrapper which may be a weighted sum
    #         of all losses, and the second is log_vars which will be sent to
    #         the logger.
    #     """
    #     log_vars = []
    #     for loss_name, loss_value in detect_losses.items():
    #         if isinstance(loss_value, torch.Tensor):
    #             log_vars.append([loss_name, loss_value.mean()])
    #         elif is_list_of(loss_value, torch.Tensor):
    #             log_vars.append(
    #                 [loss_name,
    #                  sum(_loss.mean() for _loss in loss_value)])
    #         else:
    #             raise TypeError(
    #                 f'{loss_name} is not a tensor or list of tensors')

    #     loss = sum(value for key, value in log_vars if 'loss' in key)
    #     log_vars.append(['denoise_loss', denoise_loss])
    #     log_vars.insert(0, ['loss', loss])
    #     log_vars = OrderedDict(log_vars)  # type: ignore

    #     return loss, log_vars  # type: ignore