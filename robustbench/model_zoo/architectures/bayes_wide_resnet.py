import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BBBLayers import BBBConv2d, BBBLinear


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = BBBConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = BBBConv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and BBBConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x, sample):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x, sample=sample)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out, sample=sample)
        return torch.add(x if self.equalInOut else self.convShortcut(x, sample=sample), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        # return nn.Sequential(*layers)
        return nn.ModuleList(layers)
        

    def forward(self, x, sample):
        output = x
        for layer in self.layer:
            output = layer(output, sample=sample)
            # output = layer(output)
        return output


class Bayes_WideResNet(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
        super(Bayes_WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = BBBConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = BBBLinear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, sample): 
        out = self.conv1(x, sample=sample) 
        out = self.block1(out, sample=sample)
        out = self.block2(out, sample=sample)
        out = self.block3(out, sample=sample)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out, sample=sample)
        return out

    def compute_kl_with_prior(self, prior_model):
        kl = 0
        for (nm, module), (_, prior_module) in zip(self.named_modules(), prior_model.named_modules()):
            if isinstance(module, (BBBConv2d, BBBLinear)):
                kl += module.compute_kl_with_prior(prior_module)
        return kl