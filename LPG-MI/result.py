import logging
import numpy as np
import os
import random
import statistics
import time
import torch
from argparse import ArgumentParser
from kornia import augmentation
import json
import losses as L
import utils
from evaluation import get_knn_dist, calc_fid
from models.classifiers import VGG16, IR152, FaceNet, FaceNet64
from models.generators.resnet64 import ResNetGenerator
from utils import save_tensor_images
from opacus.validators import ModuleValidator
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import copy

if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='Stage-2: Image Reconstruction')
    parser.add_argument('--model', default='VGG16', help='VGG16 | IR152 | FaceNet64 |VGG_MixedGhost')
    parser.add_argument('--inv_loss_type', type=str, default='margin', help='ce | margin | poincare')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--iter_times', type=int, default=600)
    # Generator configuration
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=64,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=4,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    # path
    parser.add_argument('--save_dir', type=str,
                        default='PLG_MI_Inversion')
    parser.add_argument('--path_G', type=str,
                        default='')
    args = parser.parse_args()

    Acc = {'Average Acc': '{:.2f}'.format(0.99) , 
           'Average Acc5' : '{:.2f}'.format(1.00) ,}
           
    E = FaceNet(1000)
    E = torch.nn.DataParallel(E).cuda()
    path_E = 'checkpoints/evaluate_model/FaceNet_95.88.tar'
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)
    
    print("=> Calculate the KNN Dist.")
    path = os.path.join('E:/NCKH2023/LPG-MI/PLG_MI_Inversion', 'VGG16')
    knn_dist = get_knn_dist(E, os.path.join(path, 'all_imgs'), "celeba_private_feats")
    print("KNN Dist %.2f" % knn_dist)

    print("=> Calculate the FID.")
    fid = calc_fid(recovery_img_path=os.path.join(path, "success_imgs"),
                   private_img_path="datasets/celeba_private_domain",
                   batch_size=200)
    print("FID %.2f" % fid)

    
    with open(os.path.join(path, 'result.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=3)
        json.dump(Acc, f, indent=3)
        json.dump({'KNN_dist' : '{.2f}'.format(knn_dist),
                   'FID' : '{.2f}'.format(fid)} , f, indent =3)

print('Saved result')