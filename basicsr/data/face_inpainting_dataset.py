import cv2
import math
import numpy as np
import os.path as osp
import torch
import random
from PIL import Image
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder, create_mask, get_image_paths, celebAHQ_masks_to_faceParser_mask_detailed
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.matlab_functions import imresize


@DATASET_REGISTRY.register()
class FaceInpaintingDataset(data.Dataset):

    def __init__(self, opt):
        super(FaceInpaintingDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mask_folder = opt['dataroot_mask'] if 'dataroot_mask' in opt else None
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.input_size = opt['img_size']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = get_image_paths(self.gt_folder)
            self.mask_paths = get_image_paths(self.mask_folder)

        # degradations
        self.downsample_list = opt['downsample_list']
        self.cond_norm = opt['cond_norm']

        logger = get_root_logger()


    def _get_hole(self, source, target, index):
         # calculate the hole map
        source_bg_mask = np.logical_or(source == 4, source == 0)
        source_bg_mask = np.logical_or(source_bg_mask, source == 8)
        source_bg_mask = np.logical_or(source_bg_mask, source == 7)
        source_bg_mask = np.logical_or(source_bg_mask, source == 11)
        source_face_mask = np.logical_not(source_bg_mask)

        target_bg_mask = np.logical_or(target == 4, target == 0)
        target_bg_mask = np.logical_or(target_bg_mask, target == 8)
        target_bg_mask = np.logical_or(target_bg_mask, target == 7)
        target_bg_mask = np.logical_or(target_bg_mask, target == 11)
        target_face_mask = np.logical_not(target_bg_mask)

        face_overlap_mask = np.logical_and(source_face_mask, target_face_mask)
        hole_mask = np.logical_xor(face_overlap_mask, target_face_mask)

        return hole_mask.astype(float)


    def _get_mask(self, index):
        mask1_path = self.mask_paths[index]
        mask2_path = self.mask_paths[(index + 1) % len(self.mask_paths)]
        """
        mask1_bytes = self.file_client.get(mask1_path)
        mask1 = imfrombytes(mask1_bytes, flag='grayscale', float32=True)
        mask2_bytes = self.file_client.get(mask2_path)
        mask2 = imfrombytes(mask2_bytes, flag='grayscale', float32=True)
        """
        mask1 = Image.open(mask1_path).convert('L')
        mask2 = Image.open(mask2_path).convert('L')

        mask1 = celebAHQ_masks_to_faceParser_mask_detailed(mask1)
        mask2 = celebAHQ_masks_to_faceParser_mask_detailed(mask2)

        return self._get_hole(mask1, mask2, index)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        mask = self._get_mask(index)

        # random horizontal flip
        [img_gt, mask], status = augment([img_gt, mask], hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        
        img_gt = cv2.resize(img_gt, (self.input_size, self.input_size))
        mask = cv2.resize(mask, (self.input_size, self.input_size))
        mask = (mask > 0)
        if len(mask.shape) < 3:
            mask = mask[:, :, None] 

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, mask = img2tensor([img_gt, mask], bgr2rgb=True, float32=True)
        img_lq = img_gt * (1 - mask)
        img_lq = torch.cat([img_lq, mask], dim=0)

        # round and clip
        # img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        in_size = torch.sum(mask) / (self.input_size * self.input_size)
        cond = torch.from_numpy(np.array([in_size], dtype=np.float32))

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'cond': cond, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
