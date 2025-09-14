# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage_raw import SingleStageRAWDetector
from .single_stage import SingleStageDetector
from .single_stage_partraw import SingleStageRAWPartDetector

from torch import Tensor
from mmdet.structures import OptSampleList, SampleList

@MODELS.register_module()
class RAWRetinaNet(SingleStageRAWDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
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
            init_cfg=init_cfg,
            cri_pix=cri_pix,
            noise_optpath=noise_optpath)

@MODELS.register_module()
class RAWPartRetinaNet(SingleStageRAWPartDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
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
        

@MODELS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
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
        

@MODELS.register_module()
class RetinaNetNoiseInv(RetinaNet):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self, **args) -> None:
        super().__init__(**args)
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             gts: Tensor = None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        backbone_x, x = self.extract_feat(batch_inputs, return_backbone=True)

        losses = self.bbox_head.loss(x, batch_data_samples)

        return losses, backbone_x, x