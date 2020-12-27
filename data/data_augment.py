#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 22:43:29 2020

@author: geoff
"""

import torchvision.transforms as transforms

train_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([112,112]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([112,112]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])