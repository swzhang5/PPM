import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from models import build_model
import os
import warnings

warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader


def get_image_list(root_path, sub_path):

    img_map = {}
    img_list = []
    # loads the image/gt pairs
    #for _, train_list in enumerate(images_path):
    train_list = sub_path.strip()
    with open(os.path.join(root_path, train_list)) as fin:
        for line in fin:
            if len(line) < 2: 
                continue
            line = line.strip().split()
            img_map[os.path.join(root_path, line[0].strip())] = \
                        os.path.join(root_path, line[1].strip())
    img_list = sorted(list(img_map.keys()))

    return img_list, img_map

def zeropadding(img):
    b,c,h,w = img.shape
    # import pdb
    # pdb.set_trace()
    block = 128
    new_h = ((h - 1) // block + 1) * block
    new_w = ((w - 1) // block + 1) * block
    dtype = img.dtype
    device = img.device
    pad_img = torch.zeros([b,c,new_h,new_w], dtype=dtype, device=device)
    # import pdb
    # pdb.set_trace()
    pad_img[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(img)
    return pad_img
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images, img_map = get_image_list(args.root_path, 'shanghai_tech_part_a_test.list')
    # images, img_map = get_image_list(args.root_path, 'test.list')
    
    list_file = []
    with torch.no_grad():
        maes = []
        mses = []
        for img_path in images:
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            # load the images
            img = cv2.imread(img_path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            width, height = img.size

            # pre-proccessing
            img = transform(img)
            # img_scale = transform(img_scale_raw)
            samples = torch.Tensor(img).unsqueeze(0)
            
            samples = samples.to(device)
            
            samples = zeropadding(samples)

            

            output, outputs  = model(samples)
    
            outputs_scores_2 = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]#[1,49152,2]
            p = torch.nn.functional.softmax(outputs['pred_logits'], -1)[0]#[49152,2]
            entropy = -torch.sum(p * torch.log2(p), -1)
            outputs_points_2 = outputs['pred_points'][0]

            threshold = 0.5
            # filter the predictions
            points = outputs_points_2[outputs_scores_2 > threshold]
            predict_cnt = int((outputs_scores_2 > threshold).sum())

            gt_path = img_map[img_path]

            gt_points = []
            with open(gt_path) as f_label:
                for line in f_label:
                    x = float(line.strip().split(' ')[0])
                    y = float(line.strip().split(' ')[1])
                    gt_points.append([x, y])
            gt_cnt = int(len(gt_points))

            mae = abs(predict_cnt - gt_cnt)
            mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
            maes.append(float(mae))
            mses.append(float(mse))


            
         
        mae = np.mean(maes)
        mse = np.sqrt(np.mean(mses))
        print(mae)
        print(mse)   



def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')
    parser.add_argument("--root_path", help="Root path for SHA_Crowd")
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
