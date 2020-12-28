# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
import os

def make_light_data(img):
    
    def _convert(img, alpha=1, beta=0):
        tmp = img.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        img[:] = tmp
    
    img_t = img.copy()
        
    # brightness distortion
    if random.randrange(2) or 1:
        beta = random.choice([random.uniform(-150, -64), random.uniform(64, 150)])
        _convert(img_t, beta=beta)
        # print("beta = {:.2f}".format(beta))
        
    #contrast distortion
    if random.randrange(2) or 1:
        alpha = random.choice([random.uniform(1, 2), random.uniform(0.5, 0.1)])
        _convert(img_t, alpha=alpha)
        # print("alpha = {:.2f}".format(alpha))
            
    return img_t

def make_blur_data(img):
    img_t = img.copy()
    kernel_size = random.choice([5,7,9,11,13,15,17])
    sigmaX = random.uniform(1, 5)
    img_t = cv2.GaussianBlur(img_t, ksize=(kernel_size, kernel_size), sigmaX=sigmaX)
    # print("kernel_size = {}, sigmaX = {:.2f}".format(kernel_size, sigmaX))

    return img_t
    
def make_occlusion_data(img, landmark):
    img_t = img.copy()
    # print("landmark: ", len(landmark))
    h, w = img.shape[0], img.shape[1]
    eye_left = (int(landmark[0]), int(landmark[1]))
    eye_right = (int(landmark[2]), int(landmark[3]))
    nose = (int(landmark[4]), int(landmark[5]))
    mouth = (int((landmark[6]+landmark[8])/2), int((landmark[7]+landmark[9])/2))
    
    eye_w = int(random.uniform(0.2, 0.3) * w/2)
    eye_h = int(random.uniform(0.1, 0.2) * h/2)
    
    nose_w = int(random.uniform(0.15, 0.25) * w/2)
    nose_h = int(random.uniform(0.3, 0.4) * h/2)
    
    mouth_w = int(random.uniform(0.35, 0.5) * w/2)
    mouth_h = int(random.uniform(0.2, 0.3) * h/2)
    
    part = random.choice([0,1,2,3])
    if part==0:
        img_t[eye_left[1]-eye_h:eye_left[1]+eye_h, eye_left[0]-eye_w:eye_left[0]+eye_w,:] = 0
    
    if part==1:
        img_t[eye_right[1]-eye_h:eye_right[1]+eye_h, eye_right[0]-eye_w:eye_right[0]+eye_w,:] = 0
    
    if part==2:
        img_t[nose[1]-nose_h:nose[1]+nose_h, nose[0]-nose_w:nose[0]+nose_w,:] = 0
    
    if part==3: 
        img_t[mouth[1]-mouth_h:mouth[1]+mouth_h, mouth[0]-mouth_w:mouth[0]+mouth_w,:] = 0
    
    
    
    
    return img_t

def make_angle_data(img):
    img_t = img.copy()
    center = (int(img.shape[1]/2), int(img.shape[0]/2))
    dsize = (img.shape[1], img.shape[0])
    angle = random.choice([random.uniform(20, 60),random.uniform(-60, -20)])
    T = cv2.getRotationMatrix2D(center, angle, 1)
    img_t = cv2.warpAffine(img_t, T, dsize)
    # mask = np.where(img_t==0)
    # img_t[mask] = 128
    
    print("angle: {:2f}".format(angle))
    
    return img_t
        
        
def main():
    
    datasets_dir = "/home/geoff/data/face_quality/"
    label_path = os.path.join(datasets_dir, "image_path.txt")
    images_path = []
    with open(label_path, "r") as f:
        images_path = f.readlines()
        
        
    for idx, image_path in enumerate(images_path):
        # read img
        img_path = os.path.join(datasets_dir, image_path.split('\n')[0])
        print(img_path)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        print("image.shape: ", img.shape)
        # img = cv2.resize(img, (640, 480))
        
        # solve image
        img_light = make_light_data(img)
        img_blur = make_blur_data(img)
        img_angle = make_angle_data(img)
        
        # cv2.imshow("img", img)
        cv2.imshow("img_angle", img_angle)
        cv2.imshow("img_blur", img_blur)
        # cv2.imshow("light", img_light)
        cv2.waitKey(0)
    
    
if __name__=="__main__":
    main()
    
