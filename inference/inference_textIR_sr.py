import argparse
import cv2
import math
import glob
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize
from torchvision import utils
import json
import kornia.color as color
import clip
import random

import sys
sys.path.append("/apdcephfs_cq2/share_1290939/branchwang/projects/GCFSR")

from basicsr.data import degradations as degradations
from basicsr.archs.gcfsr_arch import ColorizationArch, SuperResolutionArch, GCFSR
from basicsr.utils.matlab_functions import imresize


def generate(args, cond, img_lq, text_feature, g_ema, device, imgname):

    if args.task == 'inpainting':
        mask = img_lq[:, [3], :, :]

    # scale1, scale2 = None, None
    with torch.no_grad():
        output, _, = g_ema(img_lq, text_feature, cond)
    
        output.clamp_(0, 1.)

        if args.task == 'inpainting':
            output = cond * (1 - mask) + output * mask
        # output = color.lab_to_rgb(torch.cat([img_lq, output], dim=1))

        output = output.data.squeeze().float().cpu().numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        cv2.imwrite(os.path.join(args.output, imgname + '.png'), output)

        if args.task == 'inpainting':
            img_masked = img_lq[:, :3, :, :].data.squeeze().float().cpu().numpy()
            img_masked = np.transpose(img_masked[[2, 1, 0], :, :], (1, 2, 0))
            img_masked = (img_masked * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, imgname + '_masked.png'), img_masked)

    return # scale1.squeeze().cpu().data.numpy(), scale2.squeeze().cpu().data.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sr')
    parser.add_argument('--mask', type=str, default='/apdcephfs_cq2/share_1290939/branchwang/data/mask/testing_mask')
    parser.add_argument('--model_path', 
                        type=str, 
                        default='/apdcephfs_cq2/share_1290939/branchwang/data/experiments/clip_restor/experiments/colorization_ImageNet_256/models/net_g_425000.pth')
    parser.add_argument('--input', 
                        type=str, 
                        default='/apdcephfs/share_1290939/0_public_datasets/face_datasets/CelebAMask-HQ/CelebA-HQ-img', 
                        help='input test image folder')
    parser.add_argument('--output', type=str, default='/apdcephfs_cq2/share_1290939/branchwang/projects/GCFSR/inference/results/coco', help='output folder')
    parser.add_argument('--json', 
                        type=str, 
                        default='/apdcephfs_cq2/share_1290939/branchwang/projects/GCFSR/options/selected_val.json', 
                        help='output folder')
    parser.add_argument('--scale', type=int, default=32, help='input size')
    parser.add_argument('--text', type=str, default='/apdcephfs_cq2/share_1290939/branchwang/data/celeba-caption')

    args = parser.parse_args()
    
    # set up model
    if args.task == 'inpainting':
        mask_files = sorted(glob.glob(args.mask + '/*png'))
        mask_type = 6

        args.model_path = '/apdcephfs_cq2/share_1290939/branchwang/data/experiments/clip_restor/experiments/inpainting_ffhq256_norm/models/net_g_300000.pth'
        args.output = '/apdcephfs_cq2/share_1290939/branchwang/projects/GCFSR/inference/results/inpainting'

        model = GCFSR(out_size=256)
    elif args.task == 'sr':
        # args.model_path = '//apdcephfs_cq2/share_1290939/branchwang/data/experiments/clip_restor/experiments/sr_ffhq1024/models/net_g_110000.pth'
        args.model_path = '//apdcephfs_cq2/share_1290939/branchwang/data/experiments/clip_restor/experiments/sr_ffhq512_clip_loss_norm/models/net_g_540000.pth'
        scale = 32
        
        args.output = '/apdcephfs_cq2/share_1290939/branchwang/projects/GCFSR/inference/results/sr_512/' + 'x' + str(scale)
        model = SuperResolutionArch(out_size=512)
    else:
        pass


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output, exist_ok=True)

    model.load_state_dict(torch.load(args.model_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(device)

    clip_model, _ = clip.load("/apdcephfs_cq2/share_1290939/branchwang/pretrained_models/clip/ViT-B-32.pt")
    clip_model.to(device)

    img_files = glob.glob(args.input + '/*.jpg')[:500]
    # text_files = glob.glob(args.text + '/*')

    random.seed(10)

    # s1 = []
    # s2 = []

    for img_path in img_files:
        img_name = os.path.basename(img_path).split('.')[0]
        print('Testing', img_name)

        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if args.task == 'inpainting':
            img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.
        else:
            img = cv2.resize(img, (1024, 1024)).astype(np.float32) / 255.

        ###### numpy to tensor, BGR to RGB
        img_lq = cv2.resize(img, (1024 // scale, 1024 // scale))
        img_lq = cv2.resize(img, (1024, 1024))

        img_gt = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_gt = img_gt.unsqueeze(0).to(device)
        img_lq = torch.from_numpy(np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_lq = img_lq.unsqueeze(0).to(device)

        if args.task == 'inpainting':
            index = random.randint((mask_type - 1) * 2000, mask_type * 2000 - 1)
            mask = cv2.imread(mask_files[index])
            mask = cv2.resize(mask, (256, 256))
            mask = (mask > 0)

            if len(mask.shape) < 3:
                mask = mask[:, :, None]

            mask = torch.from_numpy(np.transpose(mask[:, :, [0]], (2, 0, 1))).float()
            mask = mask.unsqueeze(0).to(device)

            img_lq = img_gt * (1 - mask)
            img_lq = torch.cat([img_lq, mask], dim=1)
        else:
            in_size = scale / 64
            cond = torch.from_numpy(np.array([in_size], dtype=np.float32)).to(device)

        text_file = args.text + '/' + str(int(img_name)) + '.txt'
        
        f = open(text_file, 'r')
        sentence_list = f.readlines()

        f.close()


        text_token = clip.tokenize(sentence_list).to(device)
        text_feature = clip_model.encode_text(text_token).detach().float()
        text_feature = text_feature.mean(dim=0, keepdim=True)
        text_feature /= torch.linalg.norm(text_feature, dim=-1, keepdim=True)

        generate(args, cond, img_lq, text_feature, model, device, img_name)
        # s1.append(alpha1)
        # s2.append(alpha2)

    # np.save(os.path.join(args.output, str(scale) + '_gen.npy'), np.mean(s1, axis=0))
    # np.save(os.path.join(args.output, str(scale) + '_enc.npy'), np.mean(s2, axis=0))
        
if __name__ == '__main__':
    main()
