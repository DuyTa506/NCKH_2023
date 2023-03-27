import logging
import numpy as np
import os
import random
import statistics
import time
import torch
from argparse import ArgumentParser
from kornia import augmentation
from baselines import utils
import losses as L
from evaluation import get_knn_dist, calc_fid
from models.classifiers import VGG16, IR152, FaceNet, FaceNet64
from models.generators.resnet64 import ResNetGenerator
from utils import save_tensor_images
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gan_checkpoint = 'E:/NCKH2023/LPG-MI/GAN_Checkpoints/improved_celeba_G_facenet.tar'
G = ResNetGenerator(
        64, 128, 4,
        num_classes=1000, distribution='normal'
    )

gen_ckpt = torch.load(gan_checkpoint, map_location='cpu')
utils.load_state_dict(G, gen_ckpt)
