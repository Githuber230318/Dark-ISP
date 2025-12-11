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
class SingleStageRAWDetector(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 cri_pix: OptConfigType = None,
                 noise_optpath = None,
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
        
        self.addnoise = False
        if cri_pix is not None:
            self.cri_pix = MODELS.build(cri_pix)
        if noise_optpath is not None:
            opt = yaml_load(noise_optpath)
            self.addnoise = True
            self.noise_g = self.build_noise_g(opt['noise_g'])
            # self.noise_g = self.noise_g

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList, 
             cur_epoch=None,
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

        # normalize
        B, _, _, _ = batch_inputs.shape
        ratio =  metainfo.pop('ratio').view(B,1,1,1)
        
        batch_inputs = normlize(batch_inputs, metainfo['white_level'], metainfo['black_level']) 
        if self.addnoise:
            with torch.no_grad():
                scale =  metainfo['white_level'] - metainfo['black_level']
                batch_inputs = self.noise_g(batch_inputs, scale, ratio)
                batch_inputs = torch.clip(batch_inputs, 0., 1)
        else:
            batch_inputs = batch_inputs  * ratio
    
        x, loss_mat = self.extract_feat(batch_inputs, metainfo, cur_epoch)
        losses = self.bbox_head.loss(x, batch_data_samples)
        losses['mat_loss'] =[loss_mat]
        
        return losses

    def dict_to_tensor(self, metainfo, device):
        iso = []
        wb = []
        white_level = []
        black_level = []
        ccm = []
        exp_time = []
        ratio = []
        for i in range(len(metainfo)):
            iso.append(metainfo[i]['iso'])
            wb.append(metainfo[i]['wb'])
            white_level.append(metainfo[i]['white_level'])
            black_level.append(metainfo[i]['black_level'])
            ccm.append(metainfo[i]['ccm'])
            exp_time.append(metainfo[i]['exp_time'])
            ratio.append(metainfo[i]['ratio'])
        iso = torch.stack(iso)
        wb = torch.stack(wb)
        white_level = torch.stack(white_level)
        black_level = torch.stack(black_level)
        ccm = torch.stack(ccm)
        exp_time = torch.stack(exp_time)
        ratio = torch.stack(ratio)
        params = {}
        params['iso'] = iso.to(device)
        params['wb'] = wb.to(device)
        params['white_level'] = white_level.to(device)
        params['black_level'] = black_level.to(device)
        params['ccm'] = ccm.to(device)
        params['exp_time'] = exp_time.to(device)
        params['ratio'] = ratio.to(device)

        return params
    
    def extract_feat(self, batch_inputs: Tensor, 
                     metainfo: SampleList,
                     cur_epoch=None,
                     batch_data_samples=None) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        #print(batch_inputs.shape)
        #save_path = r'/home/mil/cui/mmdetection/SHOW/normal/'
        #check(save_path,  batch_inputs)
        x, loss_mat = self.backbone(batch_inputs, metainfo, cur_epoch, batch_data_samples)
        if self.with_neck:
            x = self.neck(x)

        if self.training:
            return x, loss_mat
        else:
            return x
        #x_show = self.backbone.pre_encoder(batch_inputs)
        #print(x_show.shape)
        #return x, x_show

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

        # normalize
        B, _, _, _ = batch_inputs.shape
        ratio =  metainfo.pop('ratio').view(B,1,1,1)
        
        batch_inputs = normlize(batch_inputs, metainfo['white_level'], metainfo['black_level'])
        if self.addnoise:
            with torch.no_grad():
                scale =  metainfo['white_level'] - metainfo['black_level']
                batch_inputs = self.noise_g(batch_inputs, scale, ratio)
        else:
            # ratio = 1
            batch_inputs = batch_inputs * ratio

        x = self.extract_feat(batch_inputs, metainfo, batch_data_samples)

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
    
    def _forward(self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None,
            cur_epoch=None)-> Tuple[List[Tensor]]:
        # process metadata batch
        device = batch_inputs.device
        metainfo = [batch_data_samples[i].metainfo for i in range(len(batch_data_samples))]
        metainfo = self.dict_to_tensor(metainfo, device)

        # normalize
        B, _, _, _ = batch_inputs.shape
        ratio =  metainfo.pop('ratio').view(B,1,1,1)
        
        batch_inputs = normlize(batch_inputs, metainfo['white_level'], metainfo['black_level']) 
        if self.addnoise:
            with torch.no_grad():
                scale =  metainfo['white_level'] - metainfo['black_level']
                batch_inputs = self.noise_g(batch_inputs, scale, ratio)
                batch_inputs = torch.clip(batch_inputs, 0., 1)
        else:
            batch_inputs = batch_inputs  * ratio
    
        x = self.extract_feat(batch_inputs, metainfo)
        results = self.bbox_head.forward(x)
        return results
        