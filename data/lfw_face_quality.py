#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 22:19:03 2020

@author: geoff
"""

import torch
import torch.utils.data as data

import os
import cv2

from data.data_augment import train_transformer

class LFWFaceQuality(data.Dataset):
    def __init__(self, datasets_dir, list_path, transform=None):
        self.imgs_path = []
        self.datasets_dir = datasets_dir
        self.transform = transform
        
        with open(list_path, 'r') as f:
            self.imgs_path = f.read().split()
        
        
    def __len__(self):
        return len(self.imgs_path)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.datasets_dir, self.imgs_path[index])
        img = cv2.imread(img_path)
        label = int(img_path.split('/')[-2])
        target = torch.zeros(5)
        target[label] = 1
        
        # cv2.imshow('img', img)
        # cv2.waitKey(1)
        
        # print(len(img))
        if img is None:
            print('-----------', img_path)
            
        if self.transform is not None:
            img = self.transform(img)
        
        # img = img.unsqueeze(axis=0)
        
        return img, target
        
if __name__=='__main__':
    datasets_dir = '/home/geoff/data/face_quality/train_val'
    train_list_path = os.path.join(datasets_dir, 'train.txt')
    dataset = LFWFaceQuality(datasets_dir, train_list_path, train_transformer)
    
    data_loader_train = data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    for i, batch in enumerate(data_loader_train):
        img, target = batch
        print(i, target)
        break
