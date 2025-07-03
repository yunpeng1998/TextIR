import cv2
import math
import numpy as np
import os.path as osp
import torch
import random
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder, create_mask, get_image_paths
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.matlab_functions import imresize


@DATASET_REGISTRY.register()
class FFHQ_control_Dataset(data.Dataset):

    def __init__(self, opt):
        super(FFHQ_control_Dataset, self).__init__()
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

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        if self.mask_paths:
            if np.random.rand() < 0.8:
                idx = random.randint(0, len(self.mask_paths) - 1)
                mask_path = self.mask_paths[idx]
                mask_bytes = self.file_client.get(mask_path)
                mask = imfrombytes(mask_bytes, flag='grayscale', float32=True)
            else:
                mask = create_mask(width=self.input_size, 
                                    height=self.input_size, 
                                    mask_width=self.input_size // 2,
                                    mask_height=self.input_size // 2)
        else:
            mask = create_mask(width=self.input_size,
                                height=self.input_size,
                                d_x=self.input_size // 16,
                                d_y=self.input_size // 16)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
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

        # in_size = scale / self.cond_norm
        # cond = torch.from_numpy(np.array([in_size], dtype=np.float32))

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
