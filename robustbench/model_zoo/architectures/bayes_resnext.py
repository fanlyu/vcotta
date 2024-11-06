import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .BBBLayers import BBBConv2d, BBBLinear

class Bayes_ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
                 inplanes, 
                 planes, 
                 cardinality, 
                 base_width, 
                 stride=1, 
                 downsample=None):
        super(Bayes_ResNeXtBottleneck, self).__init__()

        dim = int(math.floor(planes * (base_width / 64.0)))

        self.conv_reduce = BBBConv2d(
            inplanes, 
            dim * cardinality, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False)
        self.bn_reduce = nn.BatchNorm2d(dim * cardinality)

        self.conv_conv = BBBConv2d(
            dim * cardinality, 
            dim * cardinality, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            groups=cardinality,
              bias=False)
        self.bn = nn.BatchNorm2d(dim * cardinality)

        self.conv_expand = BBBConv2d(
            dim * cardinality, 
            planes * 4,
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)

        self.downsample = downsample

    def forward(self, x, sample):
        residual = x

        bottleneck = self.conv_reduce(x, sample)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck, sample)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck, sample)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + bottleneck, inplace=True)

class Bayes_CifarResNeXt(nn.Module):
    def __init__(self, block, depth, cardinality, base_width, num_classes):
        super(Bayes_CifarResNeXt, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes

        self.conv_1_3x3 = BBBConv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = BBBLinear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BBBConv2d(
                    self.inplanes, 
                    planes * block.expansion, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        

        layers = []
        for i in range(blocks):
            layers.append(
                block(self.inplanes, planes, self.cardinality, self.base_width, stride if i == 0 else 1, downsample if i == 0 else None))
            self.inplanes = planes * block.expansion

        return nn.ModuleList(layers)

    def forward(self, x, sample):#
        x = self.conv_1_3x3(x, sample)
        x = F.relu(self.bn_1(x), inplace=True)

        for layer in self.stage_1: ####sample
            x = layer(x, sample)
        for layer in self.stage_2:
            x = layer(x, sample)
        for layer in self.stage_3:
            x = layer(x, sample)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def compute_kl_with_prior(self, prior_model):
        kl = 0
        for (nm, module), (_, prior_module) in zip(self.named_modules(), prior_model.named_modules()):
            if isinstance(module, (BBBConv2d, BBBLinear)):
                kl += module.compute_kl_with_prior(prior_module)
        return kl

       
