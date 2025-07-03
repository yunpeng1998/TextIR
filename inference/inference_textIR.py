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

import sys
sys.path.append("/apdcephfs_cq2/share_1290939/branchwang/projects/GCFSR")

from basicsr.data import degradations as degradations
from basicsr.archs.gcfsr_arch import ColorizationArch
from basicsr.utils.matlab_functions import imresize


def generate(args, img_lq, cond, g_ema, device, imgname):

    with torch.no_grad():
        output, _ = g_ema(img_lq, cond)
        output = color.lab_to_rgb(torch.cat([img_lq, output], dim=1))

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        cv2.imwrite(os.path.join(args.output, imgname), output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='inpainting')
    parser.add_argument('--model_path', 
                        type=str, 
                        default='/apdcephfs_cq2/share_1290939/branchwang/data/experiments/clip_restor/experiments/colorization_ImageNet_256/models/net_g_425000.pth')
    parser.add_argument('--input', 
                        type=str, 
                        default='/apdcephfs/share_1290939/0_public_datasets/coco/val2017', 
                        help='input test image folder')
    parser.add_argument('--output', type=str, default='/apdcephfs_cq2/share_1290939/branchwang/projects/GCFSR/inference/results/coco', help='output folder')
    parser.add_argument('--json', 
                        type=str, 
                        default='/apdcephfs_cq2/share_1290939/branchwang/projects/GCFSR/options/selected_val.json', 
                        help='output folder')
    parser.add_argument('--scale', type=int, default=32, help='input size')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output, exist_ok=True)
    
    # set up model
    model = ColorizationArch(out_size=256)
    model.load_state_dict(torch.load(args.model_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(device)

    clip_model, _ = clip.load("/apdcephfs_cq2/share_1290939/branchwang/pretrained_models/ViT-B-32.pt")
    clip_model.to(device)

    test_info = json.load(open(args.json, 'r', encoding='utf-8'))

    for img_name, sentence_list in test_info.items():
        # for i, sentence in enumerate(sentence_list):

        img_path = os.path.join(args.input, img_name)
        print('Testing', img_name)

        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.

        ###### numpy to tensor, BGR to RGB
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        
        img_lab = color.rgb_to_lab(img)
        img_lq = img_lab[:, [0], :, :]
        # cond = torch.from_numpy(np.array([in_size], dtype=np.float32)).unsqueeze(0).to(device) 

        text_token = clip.tokenize(sentence_list).to(device)
        text_feature = clip_model.encode_text(text_token).detach().float()
        text_feature = text_feature.mean(dim=0, keepdim=True)
        text_feature /= torch.linalg.norm(text_feature, dim=-1, keepdim=True)

        generate(args, img_lq, text_feature, model, device, img_name)
        
if __name__ == '__main__':
    main()
