#!/usr/bin/env python3
# code from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# We modify it to use SynchronizedBatchNorm2d

import torch.nn as nn
from lib.nn import SynchronizedBatchNorm2d
import math
import logging
import torch

logger = logging.getLogger(__name__)

__all__ = ['PTResNet', 'ptresnet18', 'ptresnet50', 'ptresnet101']

rgb_model_local_paths = {
    'resnet50': 'path to resnet50',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = SynchronizedBatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def bn_freezing_wrapper_in_place(model, freeze_until=None):
    if freeze_until is None:
        return
    model_class = type(model)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(model_class, self).train(mode)
        if mode:
            for m in self.modules():
                m.train()
        else:
            for m in self.modules():
                m.eval()
            return self

        print("Freezing Mean/Var of BatchNorm2D.")
        print("Freezing Weight/Bias of BatchNorm2D.")
        assert freeze_until.startswith('layer')
        model.bn1.eval()
        model.bn1.weight.requires_grad = False
        model.bn1.bias.requires_grad = False
        done_freeze = False
        for idx in range(1, 5):
            layername = 'layer%d' % (idx)
            if layername == freeze_until:
                done_freeze = True
            layermod = getattr(model, layername)
            for m in layermod.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    print('Setting to eval', m)
            if done_freeze:
                break
        return self
    funcType = type(model.train)
    model.train = funcType(train, model)
    print("BN Freezing wrapper called.")


class PTResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                zero_init_residual=False, freeze_until=None):
        super(PTResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        if num_classes is not None:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        if freeze_until is not None:
            if freeze_until.startswith('layer'):
                self.conv1.weight.requires_grad = False
                done_freeze = False
                for idx in range(1, 5):
                    layername = 'layer%d' % (idx)
                    if layername == freeze_until:
                        done_freeze = True
                    layermod = getattr(self, layername)
                    print("Freezing %s" % (layername))
                    for m in layermod.parameters():
                        m.requires_grad = False
                    if done_freeze:
                        break
            else:
                raise NotImplementedError()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                SynchronizedBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


def ptresnet10(pretrained=False, **kwargs):
    """Constructs a PTResNet-10 model.
    https://github.com/cvjena/cnn-models/blob/master/PTResNet_preact/resnet_preact.py

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PTResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model


def ptresnet18(pretrained=False, **kwargs):
    """Constructs a PTResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PTResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_pretrained_model('resnet18'))
    return model


def ptresnet34(pretrained=False, **kwargs):
    """Constructs a PTResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PTResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_pretrained_model('resnet34'))
    return model


def ptresnet50(pretrained=False, freeze_until=None, **kwargs):
    """Constructs a PTResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PTResNet(Bottleneck, [3, 4, 6, 3], freeze_until=freeze_until,
                    **kwargs)
    if pretrained:
        model.load_state_dict(load_pretrained_model('resnet50'))
    return model


def ptresnet101(pretrained=False, **kwargs):
    """Constructs a PTResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PTResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_pretrained_model('resnet101'))
    return model
