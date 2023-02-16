from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
import dset_coco
import torch.optim as optim
import math
from custom_coco import CustomCoco, calculate_APs, CIOU_xywh_torch
from copy import copy, deepcopy
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

from util import *

import albumentations as A

images = "cocoimages"
batch_size = 4
confidence = 0.5
nms_thesh = 0.4
start = 0
CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 80
classes = load_classes("data/coco.names")

#Set up the neural network

print("Loading network.....")
model = Darknet("cfg/yolov4.cfg")

# Edit Convo Layer 114
# Here we need to edit this layer because previously the input channel to this was set as 1024 but actually this layer needs to accept the input from the concatenation of four 512-channel layers so I need to modify this layer to have input channel of 2048
model.module_list[114].conv_114 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

model.load_weights_("csdarknet53-omega_final.weights",True)
#model.load_weights("yolov4.weights")
print("Network successfully loaded")

model.net_info["height"] = 608
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

def CIOU_xywh_torch(boxes1,boxes2):
    '''
    cal CIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''
    # cx cy w h->xyxy
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    # (x2 minus x1 = width)  * (y2 - y1 = height)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # upper left of the intersection region (x,y)
    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])

    # bottom right of the intersection region (x,y)
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # if there is overlapping we will get (w,h) else set to (0,0) because it could be negative if no overlapping
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = 1.0 * inter_area / union_area
    # cal outer boxes
    outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    outer = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
    outer_diagonal_line = torch.pow(outer[..., 0], 2) + torch.pow(outer[..., 1], 2)

    # cal center distance
    # center x center y
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5

    # euclidean distance
    # x1-x2 square 
    center_dis = torch.pow(boxes1_center[...,0]-boxes2_center[...,0], 2) +\
                 torch.pow(boxes1_center[...,1]-boxes2_center[...,1], 2)

    # cal penalty term
    # cal width,height
    boxes1_size = torch.max(boxes1[..., 2:] - boxes1[..., :2], torch.zeros_like(inter_right_down))
    boxes2_size = torch.max(boxes2[..., 2:] - boxes2[..., :2], torch.zeros_like(inter_right_down))
    v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan((boxes1_size[...,0]/torch.clamp(boxes1_size[...,1],min = 1e-6))) -
            torch.atan((boxes2_size[..., 0] / torch.clamp(boxes2_size[..., 1],min = 1e-6))), 2)

    alpha = v / (1-ious+v)

    #cal ciou
    cious = ious - (center_dis / outer_diagonal_line + alpha*v)

    return cious

import torch
import numpy as np
import torch.nn as nn
from util import *
import math

def run_training(model, optimizer, dataloader, device, img_size, n_epoch, every_n_batch, every_n_epoch, ckpt_dir):
    losses = None
    for epoch_i in range(n_epoch):
        running_loss = 0.0
        for inputs, labels, bboxes in dataloader:
            inputs = Variable(torch.from_numpy(np.array(inputs)).squeeze(1).permute(0,3,1,2).float(),requires_grad=True)
            inputs = inputs.to(device)
            labels = Variable(torch.stack(labels),requires_grad=True)
            labels = labels.to(device)
            #print(inputs.shape)

            running_corrects = 0

            # zero the parameter gradients
            # it uses for update training weights
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs, True)

                pred_xywh = outputs[..., 0:4] / img_size
                pred_conf = outputs[..., 4:5]
                pred_cls = outputs[..., 5:]

                label_xywh = labels[..., :4] / img_size

                label_obj_mask = labels[..., 4:5]
                label_noobj_mask = (1.0 - label_obj_mask)
                lambda_coord = 0.001
                lambda_noobj = 0.05
                label_cls = labels[..., 5:]
                loss = nn.MSELoss()
                loss_bce = nn.BCELoss()

                loss_coord = lambda_coord * label_obj_mask * loss(input=pred_xywh, target=label_xywh)
                loss_conf = (label_obj_mask * loss_bce(input=pred_conf, target=label_obj_mask)) + \
                            (lambda_noobj * label_noobj_mask * loss_bce(input=pred_conf, target=label_obj_mask))
                loss_cls = label_obj_mask * loss_bce(input=pred_cls, target=label_cls)

                loss_coord = torch.sum(loss_coord)
                loss_conf = torch.sum(loss_conf)
                loss_cls = torch.sum(loss_cls)

                # print(pred_xywh.shape, label_xywh.shape)

                ciou = CIOU_xywh_torch(pred_xywh, label_xywh)
                # print(ciou.shape)
                ciou = ciou.unsqueeze(-1)
                # print(ciou.shape)
                # print(label_obj_mask.shape)
                loss_ciou = torch.sum(label_obj_mask * (1.0 - ciou))
                # print(loss_coord)
                loss =  loss_ciou +  loss_conf + loss_cls
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print('Running loss')
                # print(loss_coord, loss_conf, loss_cls)
        epoch_loss = running_loss / 750
        print(epoch_loss)
        print('End Epoch')

def iou_xywh_numpy(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    # print(boxes1, boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def CIOU_xywh_torch(boxes1,boxes2):
    '''
    cal CIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''
    # cx cy w h->xyxy
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    # (x2 minus x1 = width)  * (y2 - y1 = height)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # upper left of the intersection region (x,y)
    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])

    # bottom right of the intersection region (x,y)
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # if there is overlapping we will get (w,h) else set to (0,0) because it could be negative if no overlapping
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = 1.0 * inter_area / union_area

    # cal outer boxes
    outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    outer = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
    outer_diagonal_line = torch.pow(outer[..., 0], 2) + torch.pow(outer[..., 1], 2)

    # cal center distance
    # center x center y
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5

    # euclidean distance
    # x1-x2 square 
    center_dis = torch.pow(boxes1_center[...,0]-boxes2_center[...,0], 2) +\
                 torch.pow(boxes1_center[...,1]-boxes2_center[...,1], 2)

    # cal penalty term
    # cal width,height
    boxes1_size = torch.max(boxes1[..., 2:] - boxes1[..., :2], torch.zeros_like(inter_right_down))
    boxes2_size = torch.max(boxes2[..., 2:] - boxes2[..., :2], torch.zeros_like(inter_right_down))
    v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan((boxes1_size[...,0]/torch.clamp(boxes1_size[...,1],min = 1e-6))) -
            torch.atan((boxes2_size[..., 0] / torch.clamp(boxes2_size[..., 1],min = 1e-6))), 2)

    alpha = v / (1-ious+v)

    #cal ciou
    cious = ious - (center_dis / outer_diagonal_line + alpha*v)

    return cious

train_dataloader,val_dataloader = dset_coco.ret_train_val()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)



n_epoch = 5
img_size = 608
save_every_batch = False
save_every_epoch = True
ckpt_dir = "../checkpoints"


run_training(model, optimizer, train_dataloader, device,
                 img_size,
                 n_epoch,
                 save_every_batch,
                 save_every_epoch,
                 ckpt_dir)



















# Set the model in evaluation mode

# model.eval()

# read_dir = time.time()

# # Detection phase

# try:
#     imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
# except NotADirectoryError:
#     imlist = []
#     imlist.append(osp.join(osp.realpath('.'), images))
# except FileNotFoundError:
#     print ("No file or directory with the name {}".format(images))
#     exit()
    
# if not os.path.exists("des"):
#     os.makedirs("des")

# load_batch = time.time()
# loaded_ims = [cv2.imread(x) for x in imlist]

# im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
# im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
# im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)


# leftover = 0
# if (len(im_dim_list) % batch_size):
#     leftover = 1

# if batch_size != 1:
#     num_batches = len(imlist) // batch_size + leftover            
#     im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
#                         len(im_batches))]))  for i in range(num_batches)]  

# write = 0


# if CUDA:
#     im_dim_list = im_dim_list.cuda()
    
# start_det_loop = time.time()
# for i, batch in enumerate(im_batches):
#     # Load the image 
#     start = time.time()
#     if CUDA:
#         batch = batch.cuda()
#     with torch.no_grad():
#         prediction = model(Variable(batch), CUDA)

#     prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

#     end = time.time()

#     if type(prediction) == int:

#         for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
#             im_id = i*batch_size + im_num
#             print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
#             print("{0:20s} {1:s}".format("Objects Detected:", ""))
#             print("----------------------------------------------------------")
#         continue

#     prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

#     if not write:                      #If we have't initialised output
#         output = prediction  
#         write = 1
#     else:
#         output = torch.cat((output,prediction))

#     for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
#         im_id = i*batch_size + im_num
#         objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
#         print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
#         print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
#         print("----------------------------------------------------------")

#     if CUDA:
#         torch.cuda.synchronize()        
# try:
#     output
# except NameError:
#     print ("No detections were made")
#     exit()

# im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

# scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)

# output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
# output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

# output[:,1:5] /= scaling_factor

# for i in range(output.shape[0]):
#     output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
#     output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
# output_recast = time.time()
# class_load = time.time()
# colors = [[255, 0, 0], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255]]

# draw = time.time()


# def write(x, results):
#     c1 = tuple(x[1:3].int().cpu().numpy())
#     c2 = tuple(x[3:5].int().cpu().numpy())
#     img = results[int(x[0])]
#     cls = int(x[-1])
#     color = random.choice(colors)
#     label = "{0}".format(classes[cls])
#     cv2.rectangle(img, c1, c2,color, 1)
#     t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
#     c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
#     cv2.rectangle(img, c1, c2,color, -1)
#     cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
#     return img


# list(map(lambda x: write(x, loaded_ims), output))

# det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format("des",x.split("/")[-1]))

# list(map(cv2.imwrite, det_names, loaded_ims))

# end = time.time()

# print("SUMMARY")
# print("----------------------------------------------------------")
# print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
# print()
# print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
# print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
# print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
# print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
# print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
# print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
# print("----------------------------------------------------------")


# torch.cuda.empty_cache()
