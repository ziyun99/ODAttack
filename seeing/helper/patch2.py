import math
import numpy as np
import os
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable


def create_patch_mask(in_features, position):
    width = in_features.size(2)
    height = in_features.size(3)
    patch_mask = torch.zeros([1,3, width, height])

    p_w = position[0,2]
    p_h = position[0,3]
    p_x_s=position[0,0]
    p_y_s=position[0,1]
    patch_mask[:, :, p_x_s:p_x_s+p_w+1, p_y_s:p_y_s+p_h+1] = 1

    return patch_mask


def create_img_mask(in_features, patch_mask):
    mask = torch.ones([1, 3, in_features.size(2), in_features.size(3)])
    img_mask = mask - patch_mask

    return img_mask


# add a patch to the original image
def add_patch(in_features, my_patch, position, CUDA=True):
    # in_features: [1,3,416,416]

    patch_mask = create_patch_mask(in_features,position)
    img_mask = create_img_mask(in_features, patch_mask)

    if CUDA:
        patch_mask = Variable(patch_mask.cuda(), requires_grad=False)
        img_mask = Variable(img_mask.cuda(), requires_grad=False)

    patch_mask = Variable(patch_mask, requires_grad=False)
    img_mask = Variable(img_mask, requires_grad=False)

    #with_patch= in_features * img_mask + my_patch * patch_mask
    with_patch= in_features + torch.mul(my_patch , patch_mask)
    return with_patch



def shear_transform(img,shear_range1,shear_range2):
    rows,cols,ch = img.shape
    print(cols)

    # pt1 = 5+shear_range1*np.random.uniform()-shear_range1/2
    # pt2 = 20+shear_range2*np.random.uniform()-shear_range2/2
    #first1
    # pts1 = np.float32([[5,5],[20,5],[5,20]])
    # pt1 = 5+shear_range1
    # pt2 = 20+shear_range2
    # pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    #second

    pts1 = np.float32([[10,10],[10,20],[20,10]])
    pt1 = 10+shear_range1
    pt2 = 10+shear_range2
    pts2 = np.float32([[10,10],[pt1,20],[20,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(img,shear_M,(cols,rows))


def translation_transform(img,trans_range):
    rows,cols,ch = img.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    return cv2.warpAffine(img,Trans_M,(cols,rows))

def gamma_transform(img):
    gamma = random.uniform(0.4, 1.6)
    img = (img)**(gamma)
    return (img)

def addweight_transform(img1,img2):
    h, w, ch = img1.shape
    # zero_matrix
    src2 = np.zeros([h, w, ch], img1.dtype)
    a=random.uniform(0.8, 1.4)
    g=random.uniform(-0.2, 0.2)
    img_out1 = cv2.addWeighted(img1, a, src2, 1 - a, g)  #  #g=brightness, a=contrast
    img_out2 = cv2.addWeighted(img2, a, src2, 1 - a, g)  #  #g=brightness, a=contrast
    return (img_out1,img_out2)

def rotate_transform(img, angle):
    rows,cols,ch=img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    print(rotation_matrix.shape)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows))

def blur_transform(img):
    k = random.randrange(1,5,2)
    return cv2.GaussianBlur(img,(k,k),0)

def transform(img,position,patch=True,torch=True):
    if torch:
        img=torch2cv(img)

    if patch:
        img_all=img
        p_w = position[0,2]
        p_h = position[0,3]
        p_x_s=position[0,0]
        p_y_s=position[0,1]
    #    img_patch=img[:, :, p_x_s:p_x_s+p_w+1, p_y_s:p_y_s+p_h+1]
        img_patch=img[ p_x_s:p_x_s+p_w+1, p_y_s:p_y_s+p_h+1,:]
        img=img_patch
    r = np.random.randint(2, size=1)

    print(r)
    #print(r)
    if(r==0):
        img = blur_transform(img)
    if(r==1):
        img =img# gamma_transform(img)
    if(r==2):
        img = rotate_transform(img,random.randint(-5,5))
    if(r==3):
        img = translation_transform(img,10)   #suit for image not for patch
    if(r==4):
        shear_range=6 #180--0
        x1=shear_range*np.random.uniform()-shear_range/2
        y1=shear_range*np.random.uniform()-shear_range/2
        img=shear_transform(img,x1,y1)
    if patch:
        #img_all[ :,:, p_x_s:p_x_s+p_w+1, p_y_s:p_y_s+p_h+1]=img
        img_all[ p_x_s:p_x_s+p_w+1, p_y_s:p_y_s+p_h+1,:]=img
        img=img_all
    if torch:
        img=cv2torch(img)

    return img


def shear_transform_mulltiple(img,shear_range1,shear_range2):
    rows,cols,ch = img.shape
    print(cols)

    # pt1 = 5+shear_range1*np.random.uniform()-shear_range1/2
    # pt2 = 20+shear_range2*np.random.uniform()-shear_range2/2
    #first1
    # pts1 = np.float32([[5,5],[20,5],[5,20]])
    # pt1 = 5+shear_range1
    # pt2 = 20+shear_range2
    # pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    #second

    pts1 = np.float32([[10,10],[10,20],[20,10]])
    pt1 = 10+shear_range1
    pt2 = 10+shear_range2
    pts2 = np.float32([[10,10],[pt1,20],[20,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(img,shear_M,(cols,rows))


def translation_transform_multiple(img1,img2,c_x,c_y,trans_range,width=100):
    rows,cols,ch = img1.shape
    tr_x=np.random.randint(-trans_range,trans_range, size=1)
    tr_y=np.random.randint(-trans_range,trans_range, size=1)
    if c_x+tr_x>(cols-width):
        tr_x=-tr_x
    if c_x+tr_x<width:
        tr_x=-tr_x
    if c_y+tr_y>(rows-width):
        tr_y=-tr_y
    if c_y+tr_y<width:
        tr_y=-tr_y
   # tr_x = trans_range*np.random.uniform()-trans_range/2
   # tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_y],[0,1,tr_x]])
    img1=cv2.warpAffine(img1,Trans_M,(cols,rows))
    img2=cv2.warpAffine(img2,Trans_M,(cols,rows))
    return img1,img2,c_x+tr_x,c_y+tr_y

#translate patch to the img's center position
def translation_center_multiple(img1,img2,c_x,c_y):
    if torch:
        img1=torch2cv(img1)
        img2=torch2cv(img2)
    rows,cols,ch = img1.shape
    tr_x=int(416/2-c_x)
    tr_y=int(416/2-c_y)

    Trans_M = np.float32([[1,0,tr_y],[0,1,tr_x]])
    img1=cv2.warpAffine(img1,Trans_M,(cols,rows))
    img2=cv2.warpAffine(img2,Trans_M,(cols,rows))
    if torch:
         img1=cv2torch(img1)
         img2=cv2torch(img2)
    return img1,img2,c_x+tr_x,c_y+tr_y

def rotate_transform_multiple(rotate_x,rotate_y,img1,img2, angle):
    rows,cols,ch=img1.shape
    #rotate_x=cols/2
    #rotate_y=rows/2
    rotation_matrix = cv2.getRotationMatrix2D((rotate_y, rotate_x),angle, 1)
    img1=cv2.warpAffine(img1, rotation_matrix, (cols, rows))
    img2=cv2.warpAffine(img2, rotation_matrix, (cols, rows))
    return img1,img2



def transform_multiple(img1,img2,position,c_x,c_y,patch=False,torch=True):
    if torch:
        img1=torch2cv(img1)
        img2=torch2cv(img2)
    p_w = position[0,2]
    p_h = position[0,3]
    p_x_s=position[0,0]
    p_y_s=position[0,1]
    if patch:
        img_all=img1

    #    img_patch=img[:, :, p_x_s:p_x_s+p_w+1, p_y_s:p_y_s+p_h+1]
        img_patch=img1[ p_x_s:p_x_s+p_w+1, p_y_s:p_y_s+p_h+1,:]
        img=img_patch
    rotate_x=p_x_s+p_w/2
    rotate_y=p_y_s+p_h/2

    r = np.random.randint(2, size=1)
    r=3
#    print(r)
    #print(r)
    if(r==0):
        img1 = blur_transform(img1)
    if(r==1):
        img1 =gamma_transform(img1)
    if(r==2):
        img1,img2 = rotate_transform_multiple(rotate_x,rotate_y,img1,img2,random.randint(-5,5))
    if(r==3):
        img1,img2,c_x,c_y= translation_transform_multiple(img1,img2,c_x,c_y,10)   #suit for image not for patch
    if(r==4):
        shear_range=6 #180--0
        x1=shear_range*np.random.uniform()-shear_range/2
        y1=shear_range*np.random.uniform()-shear_range/2
        img1,img2=shear_transform(img1,img2,x1,y1)
    if(r==5):
        img1,img2=resize_transform_multiple(img1,img2)
    if patch:
        #img_all[ :,:, p_x_s:p_x_s+p_w+1, p_y_s:p_y_s+p_h+1]=img
        img_all[ p_x_s:p_x_s+p_w+1, p_y_s:p_y_s+p_h+1,:]=img1
        img=img_all
    if torch:
        img1=cv2torch(img1)
        img2=cv2torch(img2)
    return img1,img2,c_x,c_y



def perspective_transform(img,angle):
    w,h=img.shape[0:2]
    w2=(0.113*w/(45*45))*angle*angle
    h2=(h/420)*angle

    org = np.array([[0,0],
                    [w,0],
                    [0,h],
                    [w,h]], np.float32)

    dst = np.array([[0+w2,0-h2],
                    [w-w2,0+h2],
                    [0+w2,h+h2],
                    [w-w2,h-h2]], np.float32)

    warpR = cv2.getPerspectiveTransform(org, dst)

    result = cv2.warpPerspective(img, warpR, (h,w))

    return result,org,dst

def inverse_perspective_transform(img,org,dst):
    w,h=img.shape[0:2]

    warpR2 = cv2.getPerspectiveTransform(dst, org)

    result2 = cv2.warpPerspective(img, warpR2, (h,w))

    return result2


def perspective_transform_multiple(img,torch=True,set_angle=100):
    if torch:
        img=torch2cv2(img)
    if set_angle==100:
        angle=random.randint(-60,60)
    else:
          angle=set_angle
    img,org,dst =perspective_transform(img,angle)
    if torch:
        img=cv2torch2(img)
    return img,org,dst,angle

def gamma_transform_multiple(img,torch=True):
    if torch:
        img=torch2cv2(img)
    img =gamma_transform(img)
    if torch:
        img=cv2torch2(img)

    return img

def addweight_transform_multiple(img1,img2,torch=True):
    if torch:
        img1=torch2cv2(img1)
        img2=torch2cv2(img2)

    img1,img2 =addweight_transform(img1,img2)
  
    if torch:
        img1=cv2torch2(img1)
        img2=cv2torch2(img2)
    return img1,img2

def torch2cv(img_transform):
    img_transform=img_transform[0,:,:,:].cpu().data.numpy()
    img_transform=img_transform.transpose(1,2,0)
    return img_transform

def cv2torch(img_transform):
    img_transform=img_transform.transpose(2,0,1)
    img=torch.from_numpy(img_transform).unsqueeze(0)
    return img.cuda()

def torch2cv2(img_transform):
    img_transform=img_transform.cpu().data.numpy()
    img_transform=img_transform.transpose(1,2,0)
    return img_transform

def cv2torch2(img_transform):
    img_transform=img_transform.transpose(2,0,1)
    img=torch.from_numpy(img_transform)
    return img.cuda()

def resize_transform(img, ratio):
    rows, cols, ch = img.shape
    img = cv2.resize(img, (round(rows * ratio), round(cols * ratio)), cv2.INTER_CUBIC)
    return img


def rescale_transform_multiple(img1,img2,ratio=0,torch=True):
    if torch:
        img1=torch2cv(img1)
        img2=torch2cv(img2)
    if ratio==0:
       ratio=random.uniform(0.2,1.2)
  #  img1,img2=translation_center_multiple(img1,img2,c_x,c_y)
    img1,img2=resize_transform_multiple(img1,img2,ratio)
    if torch:
        img1=cv2torch(img1)
        img2=cv2torch(img2)
    return img1,img2,ratio

def resize_transform_multiple(img1,img2,ratio):
  #  print(ratio)
    img_w, img_h = img1.shape[1], img1.shape[0]
  #  print(img_w,img_h)
    new_w = int(img_w *ratio)
    new_h = int(img_h * ratio)
  #  print(img_w,img_h,new_h,new_w)
    w=416
    h=416
    resized_image1 = cv2.resize(img1, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    resized_image2 = cv2.resize(img2, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    canvas1 = np.zeros((w, h, 3), dtype=np.float32)
    canvas2 = np.zeros((w, h, 3), dtype=np.float32)
    if ratio>1:
        canvas1=resized_image1[(new_h-h)//2:(new_h-h)//2 + h,(new_w-w)//2:(new_w-w)//2 + w, :]
        canvas2=resized_image2[(new_h-h)//2:(new_h-h)//2 + h,(new_w-w)//2:(new_w-w)//2 + w, :]
    else:
        canvas1[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,:]=resized_image1
        canvas2[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w, :]=resized_image2
    return canvas1,canvas2

def amplify_size(img,c_x,c_y,w,h,ind=416):
    if torch:
        img=torch2cv(img)
    patch=np.zeros(shape=(3,w,h),dtype=np.float32)
    #patch=img[:,c_x-w/2:c_x+w/2,c_y-w/2:c_y+w/2]
    print(c_x-w/2)
    print(c_x+w/2)
    start_x=int(c_x-w/2)
    end_x=int(c_x+w/2)
    start_y=int(c_y-h/2)
    end_y=int(c_y+h/2)
    print(start_x)
    patch=img[start_x:end_x,start_y:end_y,:]
    print(patch.shape)
    patch_am=cv2.resize(patch, (ind,ind), interpolation = cv2.INTER_CUBIC)
    if torch:
        patch_am=cv2torch(patch_am)
    return patch_am
