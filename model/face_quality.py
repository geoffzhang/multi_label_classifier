#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 00:08:31 2020

@author: geoff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_bn(inp, oup, kernel_size, stride, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class FaceQuality(nn.Module):
    def __init__(self, width_mult=0.25, dropout=0.4, phase='train'):
        super(FaceQuality, self).__init__()
        input_channels = 64
        output_num = 5
        divisor = 8
        
        
        inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 5, 2],
                [2, 128, 1, 2],
                [4, 128, 6, 1],
                [2, 16, 1, 1],
            ]
        
        # build first and second layer
        features = []
        input_channels = _make_divisible(input_channels*width_mult, divisor)
        features.append(conv_bn(3, input_channels, kernel_size=3, stride=2, padding=1, groups=1))
        # features.append(conv_bn(input_channels, input_channels, kernel_size=3, stride=1, padding=1, groups=1))
        features.append(conv_bn(input_channels, input_channels, kernel_size=3, stride=1, padding=1, groups=input_channels))

        # build inverted residual layer
        for t, c, n, s in inverted_residual_setting:
            output_channels = _make_divisible(c*width_mult, divisor)
            for i in range(n):
                stride = s if i==0 else 1
                features.append(InvertedResidual(input_channels, output_channels, stride, t, nn.BatchNorm2d))
                input_channels = output_channels
        
        # build last several layers
        output_channels = _make_divisible(32*width_mult, divisor)
        features.append(conv_bn(input_channels, output_channels, kernel_size=3, stride=2, padding=1, groups=1))
        
        input_channels = output_channels
        output_channels = _make_divisible(128*width_mult, divisor)
        features.append(conv_bn(input_channels, output_channels, kernel_size=7, stride=1, padding=0, groups=1))
        
        self.features = nn.Sequential(*features)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        last_channel = _make_divisible(16*width_mult, 8) \
                     + _make_divisible(32*width_mult, 8) \
                     + _make_divisible(128*width_mult, 8)  
        
        self.classifier = nn.Sequential(
            # nn.Dropout(p=dropout),  
            nn.Linear(last_channel, output_num),
            )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        x = self.features[0:-2](x)
        # print(x.size())
        
        s1 = self.avg_pool(x)
        # print(x.size())
        
        x = self.features[-2](x)    
        s2 = self.avg_pool(x)
        
        x = self.features[-1](x)
        # print(x.size())
        
        s3 = x
        
        x = torch.cat([s1, s2, s3], axis=1)
        # print(x.size())
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        x = torch.sigmoid(x)
        return x
    
if __name__=='__main__':
    net = FaceQuality(width_mult=0.25)
    input = torch.randn(1,3,112,112)
    net.eval()
    
    net(input)
    
    torch.save(net.state_dict(), "net.pth")