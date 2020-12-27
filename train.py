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

def main():
    # load data
    train_data = LFWFaceQuality(args.datasets_dir, args.train_list_path, train_transformer)
    val_data = LFWFaceQuality(args.datasets_dir, args.val_list_path, val_transformer)
    
    # data loader
    train_data_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)
    val_data_loader = data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)
    
    # load model
    net = FaceQuality(width_mult=0.25, phase='train')
    net = net.to(device)
    
    
    if args.resume_net is not None and os.path.exists(args.resume_net):
        state_dict = torch.load(args.resume_net)
        net.load_state_dict(state_dict)
        print('load resume from {}'.format(args.resume_net))
    
    # loss function
    criterion = torch.nn.BCELoss(reduction='mean')
    
    # optimer
    optimizer = torch.optim.SGD(net.parameters(), args.lr, args.moment, args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    # train
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.num_epoch):
        # train(net, train_data_loader, criterion, optimizer, writer, epoch)
        # lr_scheduler.step()
        
        print('begin validation...')
        prob = validation(net, val_data_loader)
        print('epoch: {}, prob: {:.2f}%'.format(epoch, prob*100))
    
    writer.close()

def train(net, train_data_loader, criterion, optimizer, writer, epoch):
    net.train()
    iter_total = len(train_data_loader)
    for i, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
 
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        preds = net(inputs)
        loss = criterion(preds, targets)
        
        if i%100 == 0:
            torch.save(net.state_dict(), args.save_model_path+'face_quality_min_loss.pth')

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('epoch:{}/{} iter:{}/{} loss:{:.4f} lr:{:4f}'.format\
              (epoch, args.num_epoch, i, iter_total, loss.item(), current_lr)) 
        writer.add_scalars('face_quality', {'loss':loss.item()}, epoch*iter_total+i)

        loss.backward()
        optimizer.step()

def validation(net, val_data_loader):
    net.eval()
    num_pred_correct = 0
    num_all_sample = 0
    for i, batch in enumerate(val_data_loader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            preds = net(inputs)
            
            mask_preds = torch.max(preds, dim=1)[1]
            mask_targets = torch.max(targets, dim=1)[1]
            
            num = torch.eq(mask_preds, mask_targets).sum()
            num_pred_correct += num
            num_all_sample += inputs.size(0)
    
    prob = num_pred_correct * 1.0 / num_all_sample
    
    return prob

if __name__=='__main__':
    main()






