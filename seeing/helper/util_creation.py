from __future__ import division
__author__ = 'mooncaptain'

import time
import argparse
from scipy import io
import os
import os.path as osp
import numpy as np
import cv2
import pandas as pd
import random
import pickle as pkl
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

from helper.util import *
from helper.patch import transform,transform_multiple
from helper.preprocess import prep_image, inp_to_image
from helper.darknet import Darknet


num_classes = 80
confidence=0.5


def get_character_mask(character_path, inp_dim=100):
    img = cv2.imread(character_path)
    img = cv2.resize(img, (inp_dim, inp_dim))  #the size should be smaller than patch
    x=np.sum(img.astype(float), axis=2)
    x_bool=x>15   #each color thres is 5. r+g+b<15 indicates black color
    mask=np.clip(x*x_bool,0,1)
    mask=np.expand_dims(mask,axis=0)
    mask=np.concatenate((mask,mask,mask), axis=0)  #expand mask to a 3-dim array
    mask=1-mask   #the character is set to 0, the backgroud is set to 1
    img_ = img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    mask = torch.from_numpy(mask).float()
    return img_.cuda(), mask.cuda()


def get_loss_character(img_adv,map_bounding):
#    print(img_adv.shape)
#    print(map_bounding.shape)
    img_patch=img_adv*map_bounding
    red_bool=(img_patch[0,:,:]>img_patch[1,:,:]) * (img_patch[0,:,:]>img_patch[2,:,:])#[Blue, Green, Red]
    #grey
    img_patch_gen=torch.max(img_patch,0)#generalization
    img_patch_gen=torch.unsqueeze(img_patch_gen[0],0)
    img_patch_gen=torch.cat((img_patch_gen,img_patch_gen,img_patch_gen), 0)
    img_patch_gen=img_patch/(img_patch_gen+0.001)


    ra1=abs((img_patch_gen[0,:,:]+0.001)/(img_patch_gen[1,:,:]+0.001)-1)
    ra2=abs((img_patch_gen[0,:,:]+0.001)/(img_patch_gen[2,:,:]+0.001)-1)
    grey_bool=(ra1<0.1)+(ra2<0.1)
    #sum of red, grey
    sum_color=torch.sum(map_bounding[0,:,:]*(grey_bool+red_bool).float())
    sum_patch=torch.sum(map_bounding[0,:,:])
    loss=1-sum_color/sum_patch
    return loss


def get_sp_frames_success_rate(frame_index,sp_frames=100):
    length=frame_index[0]
    rate_line=np.zeros(shape=[1, length-sp_frames])
    sum_n=sum(frame_index[1:1+sp_frames])
    success_rate=sum_n/sp_frames
    rate_line[0,0]=success_rate
    for i in range(length-sp_frames-1):
        temp_sum=sum_n+frame_index[i+sp_frames+2]-frame_index[i+1]
        temp_rate=temp_sum/sp_frames
        rate_line[0,i+1]=temp_rate
        if success_rate<temp_rate:
           success_rate=temp_rate
    return success_rate, rate_line

def get_all_frames_success_rate(frame_index):
    length=frame_index[0]
    success_rate=sum(frame_index[1:length+1])/length
    return success_rate



def get_grid_cell_26(c_x,c_y):
    ind=507+(np.floor(c_x/16)*26+np.floor(c_y/16)+1)*3-1
    return ind

def get_grid_cell_13(c_x,c_y):
    ind=(np.floor(c_x/32)*13+np.floor(c_y/32)+1)*3-1
    return ind

def get_grid_cell_52(c_x,c_y):
    ind=2535+(np.floor(c_x/8)*52+np.floor(c_y/8)+1)*3-1
    return ind

def get_specified_ind(x_c,y_c,grid_size=26):
    ind=np.zeros(3)
    if grid_size==26:
       ind[0]=get_grid_cell_26(x_c,y_c)
    if grid_size==13:
       ind[0]=get_grid_cell_13(x_c,y_c)
    if grid_size==0:
       ind[0]=get_grid_cell_26(x_c,y_c)
       ind[1]=get_grid_cell_13(x_c,y_c)
       ind[2]=get_grid_cell_52(x_c,y_c)
    return ind.astype(int)


def get_loss_creation(prediction,adv_label,x_c,y_c,CUDA):#creation
    ind_nz=get_specified_ind(x_c,y_c)

    for i in range(len(ind_nz)):
        if i==0:  #improve the adv_probability of the first index
            loss_predict=(1-prediction[0,ind_nz[i],4])
            if CUDA:
                loss_class=torch.sum((prediction[0,ind_nz[i],5: 5 + num_classes]-adv_label.cuda())**2)
            else:
                loss_class=torch.sum((prediction[0,ind_nz[i],5: 5 + num_classes]-adv_label)**2)
            loss=(loss_predict+loss_class)#prediction_error+class_error,
            #print("loss_predict_loss3:",loss_predict)
        else:   #reduce the confidence of other index
           # loss=loss+prediction[0,ind_nz[i],4]
             loss=loss+(1-prediction[0,ind_nz[i],4])+torch.sum((prediction[0,ind_nz[i],5: 5 + num_classes]-adv_label.cuda())**2)
    return loss

def get_loss_creation_select(prediction,adv_label,x_c,y_c,CUDA):#creation
    ind_nz=get_specified_ind(x_c,y_c,grid_size=0)
    loss_id=torch.zeros(len(ind_nz)).cuda()
    for i in range(len(ind_nz)):
            loss_id[i]=(1-prediction[0,ind_nz[i],4])+torch.sum((prediction[0,ind_nz[i],5: 5 + num_classes]-adv_label.cuda())**2)
    loss_out=torch.min(loss_id,0)
#    loss=loss_out[0]
    loss=loss_id[2]
    loss_index=loss_out[1]
    return loss,loss_index



def get_loss_disappear(prediction,ind_nz,adv_label):#

    for i in range(len(ind_nz)):

        if i==0:  #improve the adv_probability of the first index
            loss_predict=(prediction[0,ind_nz[i],4])
           # loss_class=torch.sum((prediction[0,ind_nz[i],5: 5 + num_classes]-adv_label)**2)
            loss=(loss_predict)#prediction_error+class_error,+loss_class
        else:   #reduce the confidence of other index
            loss=loss+prediction[0,ind_nz[i],4]

    return loss

def get_loss_smoothness(img_adv,map_bounding):
    img_adv=img_adv[0,:,:,:].cuda()
    map_bounding=map_bounding[0,:,:,:].cuda()
    ch,rows,cols = img_adv.shape
    img_adv_i=torch.zeros(ch,rows,cols).cuda()
    img_adv_i[:,1:rows-1,:]=img_adv[:,0:rows-2,:]
    img_adv_j=torch.zeros(ch,rows,cols).cuda()
    img_adv_j[:,:,1:cols-1]=img_adv[:,:,0:cols-2]
    loss=torch.sum(torch.sum(torch.abs(torch.mul(img_adv-img_adv_i,map_bounding))))+torch.sum(torch.sum(torch.abs(torch.mul(img_adv-img_adv_j,map_bounding))))
    #loss=0
    #for i in range(rows):
    #    for j in range(cols):
            #print(map_bounding.shape)
    #        if map_bounding[0,i,j]!=0:
     #           loss=loss+torch.sum(torch.abs(img_adv[:,i,j]-img_adv[:,i+1,j]))+torch.sum(torch.abs(img_adv[:,i,j]-img_adv[:,i,j+1]))
    print("loss_smoothness:",loss)
    return  loss

#get median smoothness loss
def get_loss_median(img_adv,map_bounding):
    img_adv=img_adv[0,:,:,:].cuda()
    map_bounding=map_bounding[0,:,:,:].cuda()
    img_lis=torch.zeros(9,3,416,416).cuda()
    img_lis[4,:,:,:]=img_adv
    img_lis[0,:,1:416,1:416]=img_adv[:,0:415,0:415]
    img_lis[1,:,:,1:416]=img_adv[:,:,0:415]
    img_lis[2,:,0:415,1:416]=img_adv[:,1:416,1:416]
    img_lis[3,:,1:416,:]=img_adv[:,0:415,:]
    img_lis[5,:,0:415,:]=img_adv[:,1:416,:]
    img_lis[6,:,1:416,0:415]=img_adv[:,0:415,1:416]
    img_lis[7,:,:,0:415]=img_adv[:,:,1:416]
    img_lis[8,:,0:415,0:415]=img_adv[:,1:416,1:416]
    #get 3*3 slide_window median matrix
    img_median=torch.median(img_lis,0)
    img_median=img_median[0]
    loss=torch.sum(torch.sum(torch.abs(torch.mul(img_adv-img_median,map_bounding))))
    #print("loss_medien_smoothness:",loss)
    return loss

def get_loss_saturation(img_adv,map_bounding, device):
    img_adv=img_adv[0,:,:,:].cuda()
    map_bounding=map_bounding[0,0,:,:].cuda()
    m_max=torch.max(img_adv,0)[0]
    m_min=torch.min(img_adv,0)[0]
    saturation=(m_max+0.01-m_min)/(m_max+0.01)
    thres=0.8  #0.9
    sa_mask=(saturation > thres).float()
    saturation=saturation*sa_mask*map_bounding
    loss=torch.sum(torch.sum(saturation))
    return loss

def get_ind(prediction,ori_index):
    prediction=prediction.cpu()
    conf_mask = ((prediction[:,:,4] > confidence)).float().unsqueeze(2)#confidence>0.5
    # print("conf_mask")
    max_a,max_b = torch.max(prediction[:,:,5:5+ num_classes],2)
    # print(max_b)
    conf_mask2 = (max_b == ori_index).float().unsqueeze(2)#1=bicycle,11=stop_sign,9=traffic_light
    prediction = prediction*conf_mask2
    prediction = prediction*conf_mask
    ind_nz = torch.nonzero(prediction[0,:,4])
    # print('ind_nz.shape:',ind_nz.shape)
    if ind_nz.shape[0]==0:
        return [0]
    else:
        ind_out=np.zeros(shape=[3, ind_nz.shape[0]])
        ind_out[0,:]=ind_nz[:,0] #index of (confidence>0.5&&label==target_label)
        ind_out[1,:]=max_b[:,ind_nz[:,0]] # label of index
        ind_out[2,:]=prediction.data[0,ind_nz[:,0],4] #confidence of index
        ind_out=ind_out[:,(-ind_out[2,:]).argsort()] # sort index based on confidence
        ind_nz=ind_out[0,:].astype(np.int32)
        return ind_nz

def get_ind2(prediction,ori_index):
    conf_mask = ((prediction[:,:,4] > confidence)).float().unsqueeze(2)#confidence>0.5
    max_a,max_b=torch.max(prediction[:,:,5:5+ num_classes],2)
    conf_mask2 = (max_b == ori_index).float().unsqueeze(2)#1=bicycle,11=stop_sign,9=traffic_light
    prediction = prediction*conf_mask2
    prediction = prediction*conf_mask
    ind_nz = torch.nonzero(prediction[0,:,4])
    if ind_nz.shape[0]==0:
        return 0
    else:
        return 1
    
def get_map_bounding(start_x,start_y,width,height):
    position=np.zeros(shape=(1,4),dtype=np.int32)
    position[0,0]=start_x   #x_start
    position[0,1]=start_y   #y_start
    position[0,2]=width   #width
    position[0,3]=height   #height
    #patch = Variable(torch.zeros(1, 3, 416, 416))#, requires_grad=True
    map_bounding=torch.zeros(1,3,416,416)
    map_bounding[:,:,start_x:start_x+width,start_y:start_y+height]=1
    return map_bounding,position

def get_random_img_ori(imlist):
    item=random.randint(0,len(imlist)-1)#
    inp_dim=416
    img_process,img_ori,inp=prep_image(imlist[item],inp_dim)#imlist[item]
    return img_process

def get_random_stop_ori(imlist):
    item=random.randint(0,len(imlist)-1)#
    inp_dim=201
    img_process,img_ori,inp=prep_image(imlist[item],inp_dim)#imlist[item]
    return img_process
