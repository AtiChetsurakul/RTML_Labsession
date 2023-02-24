# Import libraries
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import copy
from copy import deepcopy
import numpy as np
from torchvision.transforms.transforms import RandomCrop
from model_calling import *
from model_training_ import train_model

# Prepare the dataset
# Set device to GPU or CPU

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Allow augmentation transform for training set, no augementation for val/test set

# Resize to 256
train_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Resize to 224

eval_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Download CIFAR-10 and split into training, validation, and test sets.
# The copy of the training dataset after the split allows us to keep
# the same training/validation split of the original training set but
# apply different transforms to the training set and validation set.

full_train_dataset = torchvision.datasets.CIFAR10(root='/root/data_keep', train=True,
                                                  download=True)

train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_dataset, [40000, 10000])
train_dataset.dataset = copy(full_train_dataset)
train_dataset.dataset.transform = train_preprocess
val_dataset.dataset.transform = eval_preprocess

test_dataset = torchvision.datasets.CIFAR10(root='/root/data_keep', train=False,
                                            download=True, transform=eval_preprocess)

BATCH_SIZE = 10
NUM_WORKERS = 2

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=NUM_WORKERS)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}


resnet = ResSENet18().to(device)
# count number of params


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'number of trainable parameters: {count_parameters(resnet)}')

# Optimizer and loss function
criterion = nn.CrossEntropyLoss().to(device)
params_to_update = resnet.parameters()
# Now we'll use Adam optimization
# optimizer = optim.SGD(params_to_update, lr=0.01, weight_decay = 1e-4, momentum = 0.9)
optimizer = optim.Adam(params_to_update, lr=0.01)
best_model, val_acc_history, loss_acc_history = train_model(
    resnet, dataloaders, criterion, optimizer, device, 10, 'Result/SEresnet18_bestsofar')


np.save('Result/val_acc_history_SEresnet18_25.npy', np.array(val_acc_history))
np.save('Result/val_loss_history_SEresnet18_25.npy', np.array(loss_acc_history))
