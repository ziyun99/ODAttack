from __future__ import division
__author__ = 'mooncaptain'

import time
import math
import argparse
from scipy import io
import os
import os.path as osp
import random
import pickle as pkl
import itertools
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

from helper.util import *
from helper.util_creation import get_loss_disappear,get_loss_creation_select, get_ind,get_loss_creation,get_loss_smoothness,get_map_bounding,get_random_img_ori,get_loss_median
from helper.patch import perspective_transform_multiple,inverse_perspective_transform, translation_center_multiple,gamma_transform_multiple, amplify_size,shear_transform,rotate_transform,translation_transform,gamma_transform,blur_transform,transform,transform_multiple,rescale_transform_multiple,add_patch
from helper.preprocess import prep_image, inp_to_image
from helper.darknet import Darknet


def get_test_input(input_dim, CUDA):
    img = cv2.imread("det/stop_sign.jpg")
   # img = cv2.imread("det/adv_stop_sign.png")
    img = cv2.resize(img, (input_dim, input_dim))

    img_ =  img.transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_

def get_test_input_ori(input_dim, CUDA):
    img = cv2.imread("det/stop_sign.jpg")
    #img = cv2.imread("det/adv_stop_sign.png")
    img = cv2.resize(img, (input_dim, input_dim))

    img_ =  img.transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_


def write_archor(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[7])
    label = "{0}".format(classes[cls])
    colors = pkl.load(open("helper/pallete", "rb"))
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img




def get_loss_color(img_adv,map_bounding,map_patch1,map_patch2,map_patch3):
#    print(img_adv.shape)
#    print(map_bounding.shape)
    img_adv=img_adv[0,:,:,:].cuda()
    map_bounding=map_bounding[0,:,:,:]
    map_patch1=map_patch1[0,:,:,:]
    map_patch2=map_patch2[0,:,:,:]
    map_patch3=map_patch3[0,:,:,:]
    img_patch=img_adv*map_bounding
    red_bool=(img_patch[0,:,:]>img_patch[1,:,:]) * (img_patch[0,:,:]>img_patch[2,:,:])#[Blue, Green, Red] for original image, here it is inverse
    green_bool=(img_patch[1,:,:]>img_patch[0,:,:]) * (img_patch[1,:,:]>img_patch[2,:,:])#
    blue_bool=(img_patch[2,:,:]>img_patch[1,:,:]) * (img_patch[2,:,:]>img_patch[0,:,:])#
    #grey
    img_patch_gen=torch.max(img_patch,0)#generalization
    img_patch_gen=torch.unsqueeze(img_patch_gen[0],0)
    img_patch_gen=torch.cat((img_patch_gen,img_patch_gen,img_patch_gen), 0)
    img_patch_gen=img_patch/(img_patch_gen+0.001)


    ra1=abs((img_patch_gen[0,:,:]+0.001)/(img_patch_gen[1,:,:]+0.001)-1)
    ra2=abs((img_patch_gen[0,:,:]+0.001)/(img_patch_gen[2,:,:]+0.001)-1)
    grey_bool=torch.clamp((ra1<0.1)+(ra2<0.1),0,1)
   # print(grey_bool)
    #loss for patch1

    #sum of red, grey
    sum_color=torch.sum((map_patch1[1,:,:]+map_patch1[2,:,:])*(1+-1*torch.clamp((grey_bool+red_bool),0,1).float()))
    loss_1=torch.sum(sum_color)
   # print(torch.clamp((grey_bool+red_bool),0,1).int())
    #sum of green, grey
    sum_color=torch.sum((map_patch2[0,:,:]+map_patch2[2,:,:])*(1+-1*torch.clamp((grey_bool+green_bool),0,1).float()))
    loss_2=torch.sum(sum_color)

    #sum of blue, grey
    sum_color=torch.sum((map_patch3[1,:,:]+map_patch3[0,:,:])*(1+-1*torch.clamp((grey_bool+blue_bool),0,1).float()))
    loss_3=torch.sum(sum_color)
    loss=loss_1+loss_2+loss_3
    return loss


#get_loss_saturation
def get_loss_saturation(img_adv,map_bounding,map_patch1,map_patch2,map_patch3):
    img_adv=img_adv[0,:,:,:].cuda()
    map_bounding=map_bounding[0,0,:,:].cuda()
    m_max=torch.max(img_adv,0)[0]
    m_min=torch.min(img_adv,0)[0]
    saturation=(m_max+0.01-m_min)/(m_max+0.01)
    s_loss=0
    s_loss1=0
    s_loss2=0
    s_loss3=0
    #loss for all patches
    # thres=0.8  #0.9
    # sa_mask=(saturation > thres).float()
    # saturation_loss=saturation*sa_mask*map_bounding
   #loss for patch1
    thres=0.8  #0.9
    sa_mask=(saturation > thres).float()
    saturation_loss=saturation*sa_mask*map_patch1
    s_loss1=torch.sum(torch.sum(saturation_loss))
   #loss for patch2
    thres=0.8  #0.9
    sa_mask=(saturation > thres).float()
    saturation_loss=saturation*sa_mask*map_patch2
    s_loss2=torch.sum(torch.sum(saturation_loss))
   #loss for patch3
    thres=0.8  #0.9
    sa_mask=(saturation > thres).float()
    saturation_loss=saturation*sa_mask*map_patch3
    s_loss3=torch.sum(torch.sum(saturation_loss))

    loss=s_loss1+s_loss2+s_loss3
    return loss,s_loss1,s_loss2,s_loss3



#get_loss_disappear
def get_loss(prediction,adv_label,ori_index,img_adv,map_bounding,map_patch1,map_patch2,map_patch3):
    ind_nz = get_ind(prediction, ori_index)
    loss_disappear= get_loss_disappear(prediction,ind_nz,adv_label)
#    print('loss_creation:',loss_creation)
 #   print('loss_index',loss_index)
    #loss_smoothness=get_loss_smoothness(img_adv,map_bounding)
    loss_smoothness=0#get_loss_median(img_adv,map_bounding)
    loss_saturation,loss_patch1,loss_patch2,loss_patch3=get_loss_saturation(img_adv,map_bounding,map_patch1,map_patch2,map_patch3)
    loss_color=get_loss_color(img_adv,map_bounding,map_patch1,map_patch2,map_patch3)
    loss=loss_disappear+loss_saturation*(1/5000)#loss_color*(1/10000)#+loss_saturation*(1/5000)#+loss_smoothness*(1/1000)#100*100..20000  loss_smoothness*(1/10000)+
    return  loss,loss_disappear,loss_saturation,ind_nz,loss_patch1,loss_patch2,loss_patch3,loss_color


def save_img(img,i):
    img_save = inp_to_image(img)
    cv2.imwrite('det_record/yue_record_'+str(i)+'.png', img_save)


def save_img2(img,i):
    img_save = inp_to_image(img)
    cv2.imwrite('disappear_0.1_0.3_0.5_love_satu_0.8/yue_record_'+str(i)+'.png', img_save)



def get_character_input(input_dim, character_path):
    img = cv2.imread(character_path)
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_out.cuda()

def get_stop_patch(input_dim=201):
    images='stop/'
    img = cv2.imread(images+"stop1.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    stop_ori=img_out.cuda()

    width=input_dim
    height=input_dim
    map_ori=torch.ones(1,3,width,height).cuda()
#get pattern image
    map_pattern_ori=torch.ones(1,3,width,height).cuda()
    map_pattern1=torch.ones(1,3,width,height).cuda()
    map_pattern2=torch.ones(1,3,width,height).cuda()
    map_pattern3=torch.ones(1,3,width,height).cuda()
#the first patch pattern
    pattern_path="apple.jpg"
    lenth=int(width*max(0.5-0.296,0.316-0.03))
    A=get_character_input(lenth,images+pattern_path)
    map_pattern_ori[0,:,int(height*0.03):int(height*0.03)+lenth,int(width*0.2):int(width*0.2)+lenth]=A
#    pattern_path="butter6.jpg"
#    expand=1.2
#    lenth=int(expand*width*max(0.5-0.296,0.316-0.03))
#    A=get_character_input(lenth,images+pattern_path)
#    map_pattern_ori[0,:,int(height*0.03):int(height*0.03)+lenth,int(width*0.2):int(width*0.2)+lenth]=A
    map_pattern1[0,:,int(height*0.03):int(height*0.03)+lenth,int(width*0.2):int(width*0.2)+lenth]=A

#the second patch pattern
    pattern_path="butterfly.jpg"
    lenth=int(width*max(0.796-0.531,0.316-0.133))
    B=get_character_input(lenth,images+pattern_path)
    map_pattern_ori[0,:,int(height*0.133):int(height*0.133)+lenth,int(width*0.531):int(width*0.531)+lenth]=B
    map_pattern2[0,:,int(height*0.133):int(height*0.133)+lenth,int(width*0.531):int(width*0.531)+lenth]=B
#the third patch pattern
    pattern_path="love.jpg"
    lenth=int(width*max(0.918-0.673,2*(0.52-0.265)))
    C=get_character_input(lenth,images+pattern_path)
    map_pattern_ori[0,:,int(height*0.6):width,int(width*0.265):int(width*0.265)+lenth]=C[0,:,0:width-int(height*0.6),:]
#    pattern_path="butter8.jpg"
#    lenth=int(width*max(0.918-0.673,2*(0.52-0.265)))
#    C=get_character_input(lenth,images+pattern_path)
#    map_pattern_ori[0,:,int(height*0.65):width,int(width*0.265):int(width*0.265)+lenth]=C[0,:,0:width-int(height*0.65),:]
    map_pattern3[0,:,int(height*0.65):width,int(width*0.265):int(width*0.265)+lenth]=C[0,:,0:width-int(height*0.65),:]
    #map_ori is the mask of the three patches
    map_pattern1=map_ori*((map_pattern1<0.5).float())
    map_pattern2=map_ori*((map_pattern2<0.5).float())
    map_pattern3=map_ori*((map_pattern3<0.5).float())
    map_ori=map_ori*((map_pattern_ori<0.5).float())


    #map_stop is the mask of the stop
    map_stop=-(map_ori-1)

    patch_stop=stop_ori*map_ori
    return stop_ori[0,:,:,:],map_ori[0,:,:,:],map_stop[0,:,:,:],patch_stop[0,:,:,:],map_pattern1[0,:,:,:], map_pattern2[0,:,:,:], map_pattern3[0,:,:,:] #output:original stop, map mask for patch, map mask for stop, four patch




def get_adv_episilon(img_ori,adv_label,first_index,ori_index):
    start_x = 200
    start_y = 100
    width = 201
    height = 201
    radius=(width-1)/2

    map_ori=torch.zeros(1,3,416,416).cuda()
    map_resize=torch.zeros(1,3,416,416).cuda()# the map of all three patches
    map_patch1_resize_full=torch.zeros(1,3,416,416).cuda() #the map of the first patch
    map_patch2_resize_full=torch.zeros(1,3,416,416).cuda()
    map_patch3_resize_full=torch.zeros(1,3,416,416).cuda()
    img_input=torch.zeros(1,3,416,416).cuda()
    ori_stop_perspective=torch.zeros(3,width,width).cuda()
    x_c=250
    y_c=150
   # print(patch.shape)
    original_stop,map_4_patches,map_4_stop,patch_four,map_patch1, map_patch2, map_patch3=get_stop_patch( input_dim=201)
    patch_fix=original_stop
   # patch_fix=torch.from_numpy(patch_fix).cuda().float()
   # patch_fix=img_ori[ 0,:, start_x:start_x+width,start_y:start_y+height]
    patch_transform=img_ori[ 0,:, start_x:start_x+width,start_y:start_y+height]

   # print(patch_fix.size)
    img_ori=get_test_input_ori(inp_dim, CUDA)
    
 #   img_ori = Variable(img_ori, requires_grad=True)
    nsteps=50000
    loss_dis=1
    ratio=1
    ratio_turn=1
    loss=1
    snsteps=0
    perspective=0
    for i in range(nsteps):
#patch_transform, map_bounding_transform
#patch_fix can only be changed by translation and adv_episilon
#first step's out=patch_transform
        if i%3==0:
            patch_transform[:,:,:]=gamma_transform_multiple(patch_fix[:,:,:])
# gamma_transform_multiple(patch_fix)
        else:
            patch_transform[:,:,:] = patch_fix[:,:,:]
#perspective transform
        if i%4==0:
            perspective=1
            patch_transform,org,dst,angle=perspective_transform_multiple(patch_transform)
            ori_stop_perspective,org_abandon,dst_abandon,angle_abandon=perspective_transform_multiple(original_stop,True,angle)
        else:
            ori_stop_perspective[:,:,:]=original_stop[:,:,:]

#random background
        if i%50==0:
            img_ori=get_random_img_ori(imlist).cuda()
 #       save_img(patch_transform,i)
 #       save_img2(patch_transform,i)
#third_step's output=img_input, input=patch_transform
        snsteps=snsteps+1
        if (snsteps > 50 or loss_dis<0.1):
             snsteps=0
           #it is more difficulty to fool the YOLO for stop sign(original size is 201*201) when the ratio=0.2, so we increase the cycles of ratio in(0.1,0.3)
             if(ratio_turn==1):
                 ratio = random.uniform(0.1, 1)
                 ratio_turn=0
             else:
                 ratio=random.uniform(0.1,0.3)
                 ratio_turn=1
             x_c=random.randint(10+int(ratio*radius),400-int(ratio*radius))
             y_c=random.randint(10+int(ratio*radius),400-int(ratio*radius))
        width_r=math.ceil(ratio*width)
        height_r=math.ceil(ratio*width)
        if(width_r%2==0):
             width_r=width_r+1
        if(height_r%2==0):
             height_r=height_r+1
#patch_resize=resize(stop+patch)
        patch_resize=cv2.resize(patch_transform.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)
#ori_stop_resize=resize(original stop) just for the stop_4 to get four corners
        ori_stop_resize=cv2.resize(ori_stop_perspective.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)
#map_4_patches_resize=resize(map_4_patches) just for the calculation of patches's saturation
        map_4_patches_resize=cv2.resize(map_4_patches.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)
#map_patch1_resize=resize(map_patch1 just for the calculation of patch1's saturation
        map_patch1_resize=cv2.resize(map_patch1.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)
#map_patch2_resize=resize(map_patch1 just for the calculation of patch1's saturation
        map_patch2_resize=cv2.resize(map_patch2.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)
#map_patch3_resize=resize(map_patch1 just for the calculation of patch1's saturation
        map_patch3_resize=cv2.resize(map_patch3.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)
#map_character_resize=resize(4_characters) just for the adding of four characters
#        map_character_resize=cv2.resize(map_character_ori.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)
        patch_resize=torch.from_numpy(patch_resize.transpose(2,0,1)).cuda()
        ori_stop_resize=torch.from_numpy(ori_stop_resize.transpose(2,0,1)).cuda()
        map_4_patches_resize=torch.from_numpy(map_4_patches_resize.transpose(2,0,1)).cuda()
        map_patch1_resize=torch.from_numpy(map_patch1_resize.transpose(2,0,1)).cuda()
        map_patch2_resize=torch.from_numpy(map_patch2_resize.transpose(2,0,1)).cuda()
        map_patch3_resize=torch.from_numpy(map_patch3_resize.transpose(2,0,1)).cuda()
 #       map_character_resize=torch.from_numpy(map_character_resize.transpose(2,0,1)).cuda()
 #       save_img2(patch_resize,i)
        map_resize[:,:,:,:]=map_ori[:,:,:,:]
        start_x=int(x_c-(width_r-1)/2)
        end_x=int(x_c+(width_r-1)/2+1)
        start_y=int(y_c-(height_r-1)/2)
        end_y=int(y_c+(height_r-1)/2+1)
 #       print(start_x,end_x,start_y,end_y)
#map_resize just for suturation calculation,map_patch1_resize_full for the suturation calculation of patch1
        map_resize[0,:,start_x:end_x,start_y:end_y]=map_4_patches_resize
        map_patch1_resize_full[0,:,start_x:end_x,start_y:end_y]=map_patch1_resize
        map_patch2_resize_full[0,:,start_x:end_x,start_y:end_y]=map_patch2_resize
        map_patch3_resize_full[0,:,start_x:end_x,start_y:end_y]=map_patch3_resize
#        print('map_resize:',torch.sum(torch.sum(map_resize)))
#        print("size_w:",width_r,"size_h:",height_r)
        img_input[:,:,:,:]=img_ori[:,:,:,:]

#     get four corners of stop
        stop_4=torch.sum(ori_stop_resize[:,:,:],0)
        stop_4=(stop_4<0.1).float().unsqueeze(0)
        stop_4=torch.cat((stop_4,stop_4,stop_4),0)
#        img_input[0,:,start_x:end_x,start_y:end_y]=torch.clamp((patch_resize+map_character_resize),0,1)+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
        img_input[0,:,start_x:end_x,start_y:end_y]=patch_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
     # img_input = torch.mul(img_ori, 1 - map_resize) + torch.mul(map_resize, patch_resize)
        if(i%101==0):       
               save_img2(img_input,i)
#fourth_step input=img_input
      #  print(img_ori.type,map_bounding.type,patch.type)
        input1=Variable(img_input, requires_grad=True)
        #forward
        rn_noise=torch.from_numpy(np.random.uniform(-0.1,0.1,size=(1,3,416,416))).float().cuda()
        prediction = model(torch.clamp(input1+rn_noise,0,1), CUDA)
        loss, loss_dis, loss_satu, ind_nz,loss_patch1,loss_patch2,loss_patch3,loss_color= get_loss(prediction, adv_label,ori_index,input1,map_resize,map_patch1_resize_full,map_patch1_resize_full,map_patch1_resize_full)
        if i%10==0:
                print("i:",i)
                print("loss_disappear:",loss_dis)
                print('loss_satu:',loss_satu)
                print('ind_nz:',ind_nz)
                print('ratio:',ratio)		
                print('loss_patch1:',loss_patch1)
                print('loss_patch1:',loss_patch2)
                print('loss_patch1:',loss_patch3)
                print('loss_color:',loss_color)
        loss.backward()
        input_grad=input1.grad.data
       # input_grad = torch.sign(input_grad)
        epsilon = 0.1/(int(i/5000)+1)
#inverse_rescale
        grad_resize = input_grad[0,:,start_x:end_x,start_y:end_y]
        grad_resize=cv2.resize(grad_resize.cpu().numpy().transpose(1,2,0),(width,height),cv2.INTER_CUBIC)
        #inverse_perspective_transform
        if(perspective==1):
           perspective=0
           grad_resize=inverse_perspective_transform(grad_resize,org,dst)
        grad_resize=torch.from_numpy(grad_resize.transpose(2,0,1)).cuda()
     #   print(patch_fix.shape)
    #    print(grad_resize.shape)

# add epsilon
        grad_4_patches=grad_resize*map_4_patches
        epsilon_4_patches=epsilon*torch.sign(grad_4_patches)
       # num_patches=torch.sum(map_4_patches)
      #  adv_epsilon=epsilon_4_patches/(torch.sum(torch.abs(epsilon_4_patches))/num_patches)
       # print(torch.sum(torch.abs(epsilon_4_patches)))
       # print(num_patches)
        patch_four=patch_four-epsilon_4_patches*map_4_patches
        patch_four = torch.clamp(patch_four, 0, 1)
        patch_fix = original_stop*map_4_stop+patch_four#+map_character_ori
        patch_fix=torch.clamp(patch_fix,0,1)
        if(i%100==0):
            save_img2(patch_fix,i)


    output = input1
    return output,output,patch_fix


def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest = 'images', help =
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help =
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)

    return parser.parse_args()


if __name__ ==  '__main__':
    print("1")
    args = arg_parse()

    scales = args.scales
    print(args)

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()
    torch.cuda.set_device(0)
    num_classes = 80
    classes = load_classes('data/coco.names')
    #load val_data list
    images="./val2014_2"
    try:
       imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
       imlist = []
       imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
       print ("No file or directory with the name {}".format(images))
       exit()
    print(len(imlist))
    #Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()
    #Set the model in evaluation mode
    model.eval()

    img_ori=get_test_input(inp_dim, CUDA)
    print("img_ori.shape:",img_ori.shape)

#save_ori.png
    # img_draw=img_ori[0].numpy()
    # img_draw=(img_draw*255).transpose(1,2,0)
    # cv2.imwrite('det/1.png', img_draw)

    prediction_dog = model(img_ori, CUDA)  # feature_archor_output


    ori_index = 11
    ind_nz_out = get_ind(prediction_dog, ori_index)  # 1=bicycle,11=stop_sign,9=traffic_light
    first_index=ind_nz_out[0]

    adv_label = [1 if i == 11 else 0 for i in range(80)] #stop sign=11,traffic_light=9
    adv_label=np.array(adv_label)
    adv_label=np.reshape(adv_label,(1,80))
    adv_label=torch.from_numpy(adv_label).float()


 #   output=get_adv_opt(img_ori,adv_label,first_index,ori_index)
    output_adv,patch_adv,patch_adv2 = get_adv_episilon(img_ori, adv_label, first_index, ori_index)
    prediction = model(output_adv , CUDA)
    output = write_results(prediction.data, confidence, num_classes, nms = True, nms_conf = nms_thesh)#final_output,[:,8] 8=input_img_num,xc,yc,width,height,confidence,confidence_class,class
    #print(output)
    #save adv
    img_draw=inp_to_image(output_adv)
    patch_adv=inp_to_image(patch_adv)
    patch_adv2=inp_to_image(patch_adv2)
 #   img_draw=output_adv[0].data.cpu().numpy()
#    img_draw=(img_draw*255).transpose(1,2,0)
    cv2.imwrite('det_big/yue_adv6.png', img_draw)
    cv2.imwrite('det_big/patch_adv.png',patch_adv)
    cv2.imwrite('det_big/patch_adv2.png',patch_adv2)
#  plot_adv_box
    img_draw= cv2.imread("det_big/yue_adv6.png")
   # cv2.rectangle(img_draw, (59,95), (313,308),(1,1,1), 1)
    out_save=list(map(lambda x: write_archor(x,img_draw), output))
    cv2.imwrite('det_big/yue_adv_box6.png', img_draw)