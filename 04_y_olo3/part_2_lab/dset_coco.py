
from __future__ import division
import time
import os
import os.path as osp
import numpy as np
import cv2
import pickle as pkl
import pandas as pd
import random
from copy import copy, deepcopy
import matplotlib.pyplot as plt
# apt install libgl1-mesa-glx
from custom_coco import CustomCoco, calculate_APs,CIOU_xywh_torch

import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms

from util import *

import albumentations as A
from torch.utils.data import Subset

from custom_coco import CustomCoco


# import torchvision.datasets as dset
# path2data='/root/Datasets/coco/images/train2014'
# path2json='/root/Datasets/coco/annotations/instances_train2014.json'
# coco_train = dset.CocoDetection(root = path2data,
#                                 annFile = path2json)


# gpu = "1"
# device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
def ret_train_val():
    img_size = 608 
    train_transform = A.Compose([
        #A.SmallestMaxSize(256),
        A.Resize(img_size, img_size),
        A.RandomCrop(width=224, height=224),
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
    )

    eval_transform = A.Compose([
        A.Resize(img_size, img_size),
        #A.SmallestMaxSize(256),
        A.CenterCrop(width=224, height=224),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
    )

    print('Load Dataset')
    path2data_train="/root/Datasets/coco/images/train2014/"
    path2json_train="/root/Datasets/coco/annotations/instances_train2014.json"

    path2data_val="/root/Datasets/coco/images/val2014"
    path2json_val="/root/Datasets/coco/annotations/instances_val2014.json"

    BATCH_SIZE = 10
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_dataset = Subset(CustomCoco(root = path2data_train,
                                    annFile = path2json_train, transform=train_transform), list(range(0,20)))
    val_dataset = Subset(CustomCoco(root = path2data_val,
                                    annFile = path2json_val, transform=eval_transform), list(range(0,20)))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=1, collate_fn=collate_fn)

    return train_dataloader,val_dataloader