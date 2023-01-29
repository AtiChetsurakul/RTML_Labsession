import torch.nn as nn
import torch.nn.functional as F
from module_bottle_basBlo import BasicBlock
from module_bottle_basBlo import BottleneckBlock
from module_ResSE import ResidualSEBasicBlock
from module_Resnet import ResNet


def ResNet18(num_classes=10):
    '''
    '''
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet50(num_classes=10):
    '''
    '''
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=10):
    '''
    '''
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes)


def ResSENet18(num_classes=10):
    return ResNet(ResidualSEBasicBlock, [2, 2, 2, 2], num_classes)
