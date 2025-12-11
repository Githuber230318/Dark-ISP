from .single_stage import SingleStageDetector
from typing import List, Tuple, Union
import torch
import yaml
from copy import deepcopy
from torch import Tensor
#from torch.nn.functional import interpolate
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.backbones.Dark_modules.utils import normlize
from mmdet.models.detectors.noise_utils import VirtualNoisyPairGenerator
from .base import BaseDetector
from torch import nn
import torchvision
import os
from collections import OrderedDict

@MODELS.register_module()
class SingleStageRAWPartDetector(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList, 
             gts: Tensor = None) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # process metadata batch
        device = batch_inputs.device
        metainfo = [batch_data_samples[i].metainfo for i in range(len(batch_data_samples))]
        metainfo = self.dict_to_tensor(metainfo, device)

        if gts is None:
            x = self.extract_feat(batch_inputs, metainfo, gts)
            losses = self.bbox_head.loss(x, batch_data_samples)
        else:
            x, x_l = self.extract_feat(batch_inputs, metainfo, gts)
            # gts = normlize(gts, metainfo['white_level'], metainfo['black_level'])
            # x, dk_x, dk_gt, x_ds, gt_ds, x_i = self.extract_feat(batch_inputs, metainfo, gts)
            losses = self.bbox_head.loss(x, batch_data_samples)
            # l_iso = self.cri_pix(x_ds, gt_ds) 
            # losses['finetune_loss'] = [l_iso]
            l_dn = self.cri_pix(x_l, gts)
            losses['finetune_loss'] = [l_dn] 
            # l_dk = self.cri_pix(dk_x, dk_gt)
            # losses['finetune_loss'].append(l_dk)
        
        return losses

    def dict_to_tensor(self, metainfo, device):
        wb = []
        ccm = []
        for i in range(len(metainfo)):
            wb.append(metainfo[i]['wb'])
            ccm.append(metainfo[i]['ccm'])
        wb = torch.stack(wb)
        ccm = torch.stack(ccm)
        params = {}
        params['wb'] = wb.to(device)
        params['ccm'] = ccm.to(device)

        return params
    
    def extract_feat(self, batch_inputs: Tensor, 
                     metainfo: SampleList,
                     gts: Tensor = None,) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """

        x = self.backbone(batch_inputs, metainfo)
        if self.with_neck:
            x = self.neck(x)
        return x


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
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        
        # process metadata batch
        device = batch_inputs.device
        metainfo = [batch_data_samples[i].metainfo for i in range(len(batch_data_samples))]
        metainfo = self.dict_to_tensor(metainfo, device)

        x = self.extract_feat(batch_inputs, metainfo)

        #x, x_show = self.extract_feat(batch_inputs)
        
        # save_path = r'/data/unagi0/cui_data/light_dataset/LOD_BMVC2021/RAW_dark_raw_adapter'
        # print(x_show.shape)
        # x_show = F.interpolate(x_show, (832, 1216))
        # print(x_show.shape)
        # torchvision.utils.save_image(x_show, os.path.join(save_path, batch_data_samples[0].img_id + '.png'))
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def build_noise_g(self, opt):
        opt = deepcopy(opt)
        noise_g_class = eval(opt.pop('type'))
        return noise_g_class(opt, device='cuda')