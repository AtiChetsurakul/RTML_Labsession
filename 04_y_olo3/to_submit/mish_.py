import torch.nn as nn
import torch
import torch.nn.functional as F 

class Mish(nn.Module):
     def __init__(self):
         super().__init__()

     def forward(self, x):
         return x * torch.tanh(F.softplus(x))

class LinearActivation(nn.Module):
     def __init__(self):
         super().__init__()

     def forward(self, x):
         return x