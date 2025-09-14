# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional
import torch
import mmengine.fileio as fileio
import numpy as np
import os
import re
import exifread
import mmcv
import cv2
from PIL import Image
from .base import BaseTransform
from .builder import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 postfix: str = None,
                 *,
                 backend_args: Optional[dict] = None,
                 need_gt=False) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.postfix = postfix
        self.need_gt = need_gt

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        if self.postfix == 'npz':
            img, wb, ccm = self.depack_meta(filename)
            results['wb'] = wb
            results['ccm'] = ccm
        else:
            try:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(
                        self.file_client_args, filename)
                    img_bytes = file_client.get(filename)
                else:
                    img_bytes = fileio.get(
                        filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)
        if self.need_gt:
            match = re.search(r'(\d+)\.png$', filename)
            if match:
                number = int(match.group(1))  # 提取数字并转换为整数
                new_number = number - 1  # 计算新数字
                # 构造新的文件名
                new_filename = f"{new_number}.JPG"
                # 替换路径中的文件名
                gtname = filename.replace(f"{number}.png", new_filename)
                gtname = gtname.replace("RGB_dark", "RGB_normal")
            # 读取新的图片  
            img_bytes = fileio.get(gtname, backend_args=self.backend_args)
            gt = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend) 
            results['gt'] = gt  

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def depack_meta(self, meta_path, to_tensor=True):
        meta = np.load(meta_path, allow_pickle=True)
        im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
        wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
        ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
        meta.close()
        if to_tensor:
            wb = torch.from_numpy(wb).float().contiguous()
            ccm = torch.from_numpy(ccm).float().contiguous()

        return im, wb, ccm

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str

@TRANSFORMS.register_module()
class LoadRAWImageFromFile(LoadImageFromFile):
    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 postfix: str = None,
                 use_gt: bool = False,
                 use_rgb: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__(
            to_float32=to_float32,
            color_type=color_type,
            imdecode_backend=imdecode_backend,
            file_client_args=file_client_args,
            ignore_empty=ignore_empty,
            backend_args=backend_args)
        self.postfix = postfix
        self.use_gt = use_gt
        self.use_rgb = use_rgb
    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']

        try:
            img, gt, rgb, iso, exp_time, black_level, white_level, \
               wb, ccm, raw_pattern, ratio = self.depack_meta(filename, postfix=self.postfix)
        
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        # clean_name = results['clean_path']
        # gt, gt_iso, gt_exp_time, _, _, g_wb, g_ccm, _ = self.depack_meta(clean_name, postfix=self.postfix)
        # gt = gt.astype(np.float32)
        # results['ratio'] = (gt_iso * gt_exp_time)/ (iso * exp_time)
        # results['gt'] = np.transpose(gt, (1, 2, 0)) 
        # wb = g_wb
        # ccm = g_ccm
        if self.use_rgb:
            results['gt'] = rgb
        # img = np.transpose(img, (1, 2, 0)) 
        if self.use_gt:
            results['gt'] = gt
        results['img'] = img
        results['black_level'] = black_level
        results['white_level'] = white_level
        results['wb'] = wb
        results['ccm'] = ccm
        results['iso'] = iso
        results['ratio'] = ratio
        results['exp_time'] = exp_time
        results['raw_pattern'] = raw_pattern
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def pack_raw_bayer(self, raw: np.ndarray, raw_pattern: np.ndarray):
        #pack Bayer image to 4 channels
        R = np.where(raw_pattern==0)
        G1 = np.where(raw_pattern==1)
        B = np.where(raw_pattern==2)
        G2 = np.where(raw_pattern==3)

        raw = raw.astype(np.uint16)
        H, W = raw.shape
        if H % 2 == 1:
            raw = raw[:-1]
        if W % 2 == 1:
            raw = raw[:, :-1]
        out = np.stack((raw[R[0][0]::2,  R[1][0]::2], #RGBG
                        raw[G1[0][0]::2, G1[1][0]::2],
                        raw[B[0][0]::2,  B[1][0]::2],
                        raw[G2[0][0]::2, G2[1][0]::2]), axis=0).astype(np.uint16)
        return out

    def depack_meta(self, meta_path, postfix='npz', to_tensor=True):
        if postfix == 'npz':
            meta = np.load(meta_path, allow_pickle=True)
            black_level = np.ascontiguousarray(meta['black_level'].copy().astype('float32'))
            white_level = np.ascontiguousarray(meta['white_level'].copy().astype('float32'))
            im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
            wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
            ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
            iso = np.ascontiguousarray(meta['iso'].copy().astype('float32'))
            ratio = np.ascontiguousarray(meta['ratio'].copy().astype('float32'))
            exp_time = np.ascontiguousarray(meta['exp_time'].copy().astype('float32'))
            raw_pattern = meta['raw_pattern'].copy()
            if self.use_gt:
                gt = np.ascontiguousarray(meta['gt'].copy().astype('float32'))
            else:
                gt = None
            if self.use_rgb:
                rgb = np.ascontiguousarray(meta['rgb'].copy().astype('float32'))
            else:
                rgb = None
            meta.close()
        elif postfix == None:
            import rawpy
            ## using rawpy
            raw = rawpy.imread(meta_path)
            iso, exp_time = self.metainfo(meta_path)
            raw_vis = raw.raw_image_visible.copy()
            raw_pattern = raw.raw_pattern
            wb = np.array(raw.camera_whitebalance, dtype='float32').copy()
            wb /= wb[1]
            ccm = np.array(raw.rgb_camera_matrix[:3, :3],
                            dtype='float32').copy()
            black_level = np.array(raw.black_level_per_channel,
                                    dtype='float32').reshape(4, 1, 1)
            white_level = np.array(raw.camera_white_level_per_channel,
                                    dtype='float32').reshape(4, 1, 1)
            im = self.pack_raw_bayer(raw_vis, raw_pattern).astype('float32')
            raw.close()
       

        if to_tensor:
            # im = torch.from_numpy(im).float().contiguous()
            black_level = torch.from_numpy(black_level).float().contiguous()
            white_level = torch.from_numpy(white_level).float().contiguous()
            wb = torch.from_numpy(wb).float().contiguous()
            ccm = torch.from_numpy(ccm).float().contiguous()
            iso = torch.tensor(iso, dtype=torch.float)
            exp_time = torch.tensor(exp_time, dtype=torch.float)
            ratio = torch.tensor(ratio, dtype=torch.float)

        return im, gt, rgb, iso, exp_time,\
               black_level, white_level, \
               wb, ccm, raw_pattern, ratio

    def metainfo(self, rawpath):
        with open(rawpath, 'rb') as f:
            tags = exifread.process_file(f)
            _, suffix = os.path.splitext(os.path.basename(rawpath))

            if suffix == '.dng':
                expo = eval(str(tags['Image ExposureTime']))
                iso = eval(str(tags['Image ISOSpeedRatings']))
            else:
                expo = eval(str(tags['EXIF ExposureTime']))
                iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
        return iso, expo

@TRANSFORMS.register_module()
class LoadAnnotations(BaseTransform):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in key point detection.
                # Can only load the format of [x1, y1, v1,…, xn, yn, vn]. v[i]
                # means the visibility of this keypoint. n must be equal to the
                # number of keypoint categories.
                'keypoints': [x1, y1, v1, ..., xn, yn, vn]
                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in np.float32
            'gt_bboxes': np.ndarray(N, 4)
             # In np.int64 type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # with (x, y, v) order, in np.float32 type.
            'gt_keypoints': np.ndarray(N, NK, 3)
        }

    Required Keys:

    - instances

      - bbox (optional)
      - bbox_label
      - keypoints (optional)

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int64)
    - gt_seg_map (np.uint8)
    - gt_keypoints (np.float32)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        with_keypoints (bool): Whether to parse and load the keypoints
            annotation. Defaults to False.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(
        self,
        with_bbox: bool = True,
        with_label: bool = True,
        with_seg: bool = False,
        with_keypoints: bool = False,
        imdecode_backend: str = 'cv2',
        file_client_args: Optional[dict] = None,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        for instance in results['instances']:
            gt_bboxes.append(instance['bbox'])
        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape(-1, 4)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results['instances']:
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client_args is not None:
            file_client = fileio.FileClient.infer_client(
                self.file_client_args, results['seg_map_path'])
            img_bytes = file_client.get(results['seg_map_path'])
        else:
            img_bytes = fileio.get(
                results['seg_map_path'], backend_args=self.backend_args)

        results['gt_seg_map'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        gt_keypoints = []
        for instance in results['instances']:
            gt_keypoints.append(instance['keypoints'])
        results['gt_keypoints'] = np.array(gt_keypoints, np.float32).reshape(
            (len(gt_keypoints), -1, 3))

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str
