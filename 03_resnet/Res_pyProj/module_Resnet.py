import torch.nn as nn
import torch.nn.functional as F
from module_bottle_basBlo import BasicBlock
from module_bottle_basBlo import BottleneckBlock
from module_ResSE import ResidualSEBasicBlock


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()

        self.is_debug = False

        self.in_planes = 64
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # FC layer = 1 layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.EXPANSION, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.EXPANSION
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.is_debug:
            print(f'conv1: {out.shape}')
        out = self.maxpool(out)
        if self.is_debug:
            print(f'max pool: {out.shape}')
        out = self.layer1(out)
        if self.is_debug:
            print(f'conv2_x: {out.shape}')
        out = self.layer2(out)
        if self.is_debug:
            print(f'conv3_x: {out.shape}')
        out = self.layer3(out)
        if self.is_debug:
            print(f'conv4_x: {out.shape}')
        out = self.layer4(out)
        if self.is_debug:
            print(f'conv5_x: {out.shape}')
        out = self.avgpool(out)
        if self.is_debug:
            print(f'avg pool: {out.shape}')
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
