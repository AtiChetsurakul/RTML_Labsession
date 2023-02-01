from model_calling import *
import torchvision.datasets as dataset_
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
from model_training_ import train_model
from model_calling import *
import pickle
import torchvision.utils as vutils
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


# mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

dataset = dataset_.ImageFolder(root='/root/data_keep/muffin',
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

dataset_test = dataset_.ImageFolder(root='/root/data_keep/muff_test',
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
test_dataloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=4, shuffle=True)


folds = 6
skf = StratifiedKFold(n_splits=folds, shuffle=True)


models = []


def make_model(ResSENet18):
    model = ResSENet18()
    model.load_state_dict(torch.load(
        '/root/keep_lab/RTML_Labsession/03_resnet/Res_pyProj/Result/SEresnet18_bestsofar.pth'))
    model.linear = nn.Linear(512, 2)
    model.eval()
    return model


n_models = 6

for i in np.arange(n_models):
    # fig, ax = plt.subplots(1, 2, sharex=True, figsize=(20, 5))
    model_acc = []
    model_loss = []
    for fold, (train_index, val_index) in enumerate(skf.split(dataset, dataset.targets)):
        # print('********************* Fold {}/{} ******************** '.format(fold, 8 - 1),
        #   file=open(f"SE_muff_chi_model{i}.txt", "a"))
        batch_size = 4
        train = torch.utils.data.Subset(dataset, train_index)
        val = torch.utils.data.Subset(dataset, val_index)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                                   shuffle=True, num_workers=0,
                                                   pin_memory=False)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=False)

        dataloaders = {'train': train_loader, 'val': val_loader}
        dataloaders = {'train': train_loader, 'val': val_loader}

        model = make_model(ResSENet18)
        model.to(device)
        dataloaders = {'train': train_loader, 'val': val_loader}
        criterion = nn.CrossEntropyLoss().to(device)
        if i <= 3:
            optimizer = optim.Adam(model.parameters(), lr=0.005 + 0.01*i)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=0.0005, momentum=(1-(9*10**float(2-i))))

        _, val_acc_history, loss_acc_history = train_model(model, dataloaders,
                                                           criterion, optimizer, device,
                                                           25, f'Result/SE_muff_chi_model{i}')
        model_acc.append(torch.Tensor(val_acc_history).detach().numpy())
        model_loss.append(torch.Tensor(loss_acc_history).detach().numpy())
        with open(f'Result/SE_muff_acc{i}.atikeep', 'wb') as pic:
            pickle.dump((model_acc, model_loss), pic)
        # np.save(f'Result/SE_muff_chi_acc{i}.npy',
            # np.array(torch.Tensor(val_acc_history).detach().numpy()))
        # np.save(f'Result/SE_muff_chi_acc{i}.npy', np.array(
            # torch.Tensor(loss_acc_history).detach().numpy()))
