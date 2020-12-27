from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

import sys
sys.path.append("/home/geoff/workspace/github_mine/pytorch_backbone/datasets/face_quality")
from generate_data import *

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='/home/geoff/data/face_quality/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.4, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
#    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = testset_folder + "image_path.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    rotation = 1
    for i, img_name in enumerate(test_dataset):
        image_path = os.path.join(testset_folder, img_name)
        
        save_name = image_path.split('/')[-1]
        print("{}, path: {}".format(i, save_name))
        
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        
        if rotation==0:
            img_raw = make_angle_data(img_raw)
        
        img = np.float32(img_raw)
        

        # testing scale
        target_size = 640
        max_size = 640
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
#        print(img.size())
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        bboxs = dets

        idx_max = -1
        area_max = 0
        if len(bboxs)==0:
            continue
            
        for i, box in enumerate(bboxs):
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            if w*h > area_max:
                area_max = w*h
                idx_max = i
        
        
        x1 = int(bboxs[idx_max][0])
        y1 = int(bboxs[idx_max][1])
        x2 = int(bboxs[idx_max][2])
        y2 = int(bboxs[idx_max][3])
        w = x2 - x1
        h = y2 - y1
        x1 -= int((h-w)/2)
        w = h
        x2 = x1 + w
        
        x1_tmp = max(x1, 0)
        w = h = w - (x1_tmp-x1)
        x1 = x1_tmp
        
        x2_tmp = min(x2, im_width)
        w = h = w-(x2-x2_tmp)
        x2 = x2_tmp
        
        y1_tmp = max(y1, 0)
        w = h = h -(y1_tmp - y1)
        y1 = y1_tmp
        
        y2_tmp = min(y2, im_height)
        w = h = h - (y2 - y2_tmp)
        y2 = y2_tmp
        print(x1,y1,x2,y2)
        
        landmark = bboxs[idx_max][5:15]
        landmark[0::2] = landmark[0::2] - x1
        landmark[1::2] = landmark[1::2] - y1
##        cv2.rectangle(img_raw, (x1, y1), (x2, y2), (255,0,0),2)
##        cv2.circle(img_crop, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)

#        
        img_crop = img_raw[y1:y1+h, x1:x1+w,:]
#        
        #
        img_blur = make_blur_data(img_crop)
        
        #
        img_light = make_light_data(img_crop)
        
        #
        img_occlusion = make_occlusion_data(img_crop, landmark)
        
        if args.save_image != True:
            crop_dir = os.path.join(testset_folder,"4")
            if not os.path.isdir(crop_dir):
                os.makedirs(crop_dir)
            cv2.imwrite(os.path.join(crop_dir, save_name), img_crop)
        
        # save images
        if args.save_image:
            # create folder
            crop_dir = os.path.join(testset_folder,"0")
            if not os.path.isdir(crop_dir):
                os.makedirs(crop_dir)
            blur_dir = os.path.join(testset_folder,"1")
            if not os.path.isdir(blur_dir):
                os.makedirs(blur_dir)
            light_dir = os.path.join(testset_folder,"2")
            if not os.path.isdir(light_dir):
                os.makedirs(light_dir)
            occlusion_dir = os.path.join(testset_folder,"3")
            if not os.path.isdir(occlusion_dir):
                os.makedirs(occlusion_dir)
            
            #save
            cv2.imwrite(os.path.join(blur_dir, save_name), img_blur)
            cv2.imwrite(os.path.join(light_dir, save_name), img_light)
            cv2.imwrite(os.path.join(occlusion_dir, save_name), img_occlusion)
            cv2.imwrite(os.path.join(crop_dir, save_name), img_crop)
            
#        name = "name.jpg"
#        cv2.imshow("img_blur", img_blur)
#        cv2.imshow("img_light", img_light)
#        cv2.imshow("img_occlusion", img_occlusion)
#        cv2.imshow("crop", img_crop)
#        cv2.imwrite(name, img_raw)
#        cv2.waitKey(10)
            
            
            

