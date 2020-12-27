#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:59:15 2020

@author: geoff
"""

import argparse
import os

import torch
import torch.utils.data as data
import torch.optim as optim

from data.lfw_face_quality import LFWFaceQuality
from model.face_quality import FaceQuality
from data.data_augment import train_transformer, val_transformer
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='face_quality')
parser.add_argument('--datasets_dir', default='/home/geoff/data/face_quality/train_val', type=str)
parser.add_argument('--train_list_path', default='/home/geoff/data/face_quality/train_val/train.txt', type=str)
parser.add_argument('--val_list_path', default='/home/geoff/data/face_quality/train_val/val.txt', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--moment', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--num_epoch', default=100, type=float)
parser.add_argument('--tensorboard', default='log', type=str)
parser.add_argument('--save_model_path', default='./weights/', type=str)
parser.add_argument('--resume_net', default='./weights/face_quality_min_loss.pth', type=str)

args = parser.parse_args()

# set deivce
device = torch.device('cuda: {}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
print('device: ', device)
# load data
train_data = LFWFaceQuality(args.datasets_dir, args.train_list_path, train_transformer)
val_data = LFWFaceQuality(args.datasets_dir, args.val_list_path, val_transformer)

# data loader
train_data_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)
val_data_loader = data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)

# load model
net = FaceQuality(width_mult=0.25, phase='train')
net = net.to(device)
net.train()

if args.resume_net is not None and os.path.exists(args.resume_net):
    state_dict = torch.load(args.resume_net)
    net.load_state_dict(state_dict)

# loss function
critersion = torch.nn.BCELoss(reduction='mean')

# optimer
optimizer = torch.optim.SGD(net.parameters(), args.lr, args.moment, args.weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)

# train
writer = SummaryWriter(args.tensorboard)

count = 0
iter_total = len(train_data_loader)
min_loss = 1e6
for epoch in range(args.num_epoch):
    for i, batch in enumerate(train_data_loader):
        count = count+1
        optimizer.zero_grad()
 
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        preds = net(inputs)
        loss = critersion(preds, targets)
        
        if min_loss < loss.item() and i%100 == 0:
            min_loss = loss.item()
            torch.save(net.state_dict(), args.save_model_path+'face_quality_min_loss.pth')
        print('epoch:{}/{} iter:{}/{} loss:{:.4f}'.format(epoch, args.num_epoch, i, iter_total, loss.item()))
 
        writer.add_scalars('face_quality', {'loss':loss.item()}, count)
        loss.backward()
        optimizer.step()
        # loss.update()
        
        
    lr_scheduler.step()
writer.close()
    






