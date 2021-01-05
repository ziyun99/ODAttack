
from __future__ import division
__author__ = 'mooncaptain'

import kornia
import time
import math
import argparse
from scipy import io
import os
import os.path as osp
import random
import pickle as pkl
import numpy as np
import cv2
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data as Data
from torch import autograd, optim
import torch.nn.functional as F

from helper.util import *
from helper.util_creation import get_loss_disappear,get_loss_creation_select, get_ind,get_ind2,get_loss_creation,get_loss_smoothness,get_map_bounding,get_random_img_ori,get_loss_median,get_loss_saturation,get_random_stop_ori
from helper.patch import add_patch
from helper.patch2 import  addweight_transform_multiple, perspective_transform_multiple,inverse_perspective_transform, translation_center_multiple,gamma_transform_multiple, amplify_size,shear_transform,rotate_transform,translation_transform,gamma_transform,blur_transform,transform,transform_multiple,rescale_transform_multiple
from helper.preprocess import prep_image, inp_to_image
from helper.darknet4 import Darknet

from tensorboardX import SummaryWriter
import subprocess

class Seeing():
    def __init__(self, config, args):
        super(Seeing, self).__init__()
        self.config = config
        self.load_images()
        self.load_model()

    def load_images(self):
        print("Load target images for AE generation, eg. stop sign")
        try:
            self.imlist_stop = [osp.join(osp.realpath('.'), self.config.stop_dir, img) for img in os.listdir(self.config.stop_dir) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] == '.jpg']
        except NotADirectoryError:
            self.imlist_stop = []
            self.imlist_stop.append(osp.join(osp.realpath('.'), self.config.stop_dir))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format(self.config.stop_dir))
            exit()

        print("Load background images for AE generation, eg. road")
        try:
            self.imlist_back = [osp.join(osp.realpath('.'), self.config.bg_dir, img) for img in os.listdir(self.config.bg_dir) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg' or os.path.splitext(img)[1] =='.BMP']
        except NotADirectoryError:
            self.imlist_back = []
            self.imlist_back.append(osp.join(osp.realpath('.'), self.config.bg_dir))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format(self.config.bg_dir))
            exit()

        print("Load background images for AE generation, eg. road")
        try:
            self.imlist_back_test = [osp.join(osp.realpath('.'), self.config.bg_test_dir, img) for img in os.listdir(self.config.bg_test_dir) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg' or os.path.splitext(img)[1] =='.BMP']
        except NotADirectoryError:
            self.imlist_back_test = []
            self.imlist_back_test.append(osp.join(osp.realpath('.'), self.config.bg_test_dir))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format(self.config.bg_test_dir))
            exit()

        print(self.config.stop_dir, ": ", len(self.imlist_stop), "stop imgs")
        print(self.config.bg_dir, ": ", len(self.imlist_back), "background imgs")
        print(self.config.bg_test_dir, ": ", len(self.imlist_back_test), "background imgs")

    def load_model(self):
        #Set up the neural network
        print("Loading network.....")
        self.model = Darknet(self.config.cfg)
        self.model.load_weights(self.config.weights)
        print("Network successfully loaded")
        
        self.model.net_info["height"] = self.config.reso
        self.inp_dim = int(self.model.net_info["height"])
        
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        self.model.to(self.config.device)
        self.model.eval() #Set the model in evaluation mode

        self.classes = load_classes(self.config.classes)
        self.num_classes = len(self.classes)

    def init_tensorboard(self):
        # subprocess.Popen(['tensorboard', '--host 0.0.0.0 --port 8080 --logdir={self.config.logdir}'])
        time_str = time.strftime("%Y%m%d-%H%M%S")
        if self.config.name is not None:
            return SummaryWriter(f'{self.config.logdir}/{time_str}_{self.config.name}')
        else:
            return SummaryWriter(f'{self.config.logdir}/{time_str}')
        
    def attack(self):
        adv_label = [1 if i == self.config.ori_index else 0 for i in range(self.num_classes)] #stop sign=11,traffic_light=9
        adv_label = np.array(adv_label)
        adv_label = np.reshape(adv_label,(1,self.num_classes))
        adv_label = torch.from_numpy(adv_label).float()

        patch_adv, input1 = self.get_adv_episilon(adv_label)

        # self.save_img_i(patch_adv, os.path.join(self.config.out_path, 'final/adv_stop'))
        # self.det_and_save_img_i(input1, os.path.join(self.config.out_path, 'final/adv_img'))

        print("Done and exit")
                          
    def get_adv_episilon(self, adv_label):
        self.writer = self.init_tensorboard()
  
        text = "Experiment Name: " + self.config.name + " | fir: " + str(self.config.fir_flag) + " " + str(self.config.fir_p) + " | dist: " + str(self.config.dist_flag) + " " + str(self.config.dist_p) + " | tv: " + str(self.config.tv_flag) + " " + str(self.config.tv_p) + " | nps: " + str(self.config.nps_flag) + " " + str(self.config.nps_p) + " | satur: " + str(self.config.satur_flag) + " " + str(self.config.satur_p)
        self.writer.add_text('experiment_configuration', text, 0) 
        self.writer.add_text('experiment_configuration', str(self.__dict__), 0) 
        self.writer.add_text('experiment_configuration', str(self.config.__dict__), 0) 

        #img_ori
        img_ori= self.get_test_input(self.inp_dim, self.config.path_img_ori)

        #ori_stop
        if self.config.from_mask:
            original_stop,map_4_patches,map_4_stop,patch_four=self.get_stop_patch_from_mask(self.config.patch_dim)
        else:
            original_stop,map_4_patches,map_4_stop,patch_four=self.get_stop_patch(self.config.patch_dim)
        original_stop0 = original_stop
        
        #pole
        ori_pole = self.get_pole(self.config.path_pole)

        if self.config.optimizer == "adam":
            patch_four=Variable(patch_four, requires_grad=True)

            optimizer = optim.Adam([patch_four], lr=0.03, amsgrad=True)
            scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
            scheduler = scheduler_factory(optimizer)
            if self.config.debug: print("patch_four", patch_four.requires_grad)
            
        printability_array = self.get_printability_array().to(self.config.device)
        
        init_time = time.time()
        suc_rate = 0
        try: 
            for i in range(self.config.nepochs):
                epoch_start = time.time()
                                    
                epoch_total_loss = 0
                epoch_det_loss = 0
                epoch_fir_loss = 0
                epoch_dist_loss = 0
                epoch_tv_loss = 0
                epoch_nps_loss = 0
                epoch_satur_loss = 0
                    
                original_stop = get_random_stop_ori(self.imlist_stop).to(self.config.device)
                original_stop = original_stop[0, :, :, :]
                for j in range(self.config.batch_size):
                    print("\nEpoch: {}, step: {}".format(i, j))
                    patch_f = patch_four * map_4_patches
                    patch_fix = original_stop * map_4_stop + patch_f
                    patch_fix = torch.clamp(patch_fix, 0, 1)
                    start_x = 200
                    start_y = 100
                    width = 201
                    height = 201
                    width_pole=201
                    height_pole=27
                    perspective=0
                    #map
                    map_ori=torch.zeros(1,3,416,416)
                    map_resize=torch.zeros(1,3,416,416)
                    #img_input
                    img_input=torch.zeros(1,3,416,416).to(self.config.device)
                    img_input2=torch.zeros(1,3,416,416).to(self.config.device)
                    #perspective
                    ori_stop_perspective=torch.zeros(3,201,201).to(self.config.device)
                    ori_stop2=torch.zeros(3,201,201).to(self.config.device)
                    ori_stop2_pers=torch.zeros(3,201,201).to(self.config.device)
                    #patch
                    patch_transform=img_ori[0,:, start_x:start_x+width,start_y:start_y+height]
                    patch_pers_transform=img_ori[0,:, start_x:start_x+width,start_y:start_y+height]
                    patch_pole_transform=torch.zeros(3,27,201).to(self.config.device)

                    k = random.randint(0,4)
                    addweight_transform = "False"
                    if k % 3 == 0:
                        addweight_transform = "True"
                        # brightness and contrast transformation
                        ch, h, w = patch_fix.shape #[3,201,201]
                        src2 = torch.zeros([ch, h, w], dtype = patch_fix.dtype).to(self.config.device)
                        a = random.uniform(0.8, 1.4)
                        g = random.uniform(-0.2, 0.2)
                        patch_transform = a*patch_fix + (1-a)*src2 + g
                        ori_stop2 = a*original_stop + (1-a)*src2 + g
                        patch_transform = torch.clamp(patch_transform, 0.000001, 0.999999)
                        ori_stop2 = torch.clamp(ori_stop2, 0.000001, 0.999999)
                        patch_pole_transform[:,:,:] = ori_pole[:,:,:]
                    else:
                        patch_transform[:,:,:] = patch_fix[:,:,:]
                        patch_pole_transform[:,:,:] = ori_pole[:,:,:]
                        ori_stop2[:,:,:] = original_stop[:,:,:]
                    if self.config.debug: print("patch_transform", patch_transform.requires_grad)
                    
                    if k % 2 == 0:
                        #perspective transform
                        angle = random.randint(-60,60)
                        w,h = patch_transform.shape[1:]
                        print(w, h)
                        w2 = (0.113*w/(45*45))*angle*angle
                        h2 = (h/420)*angle
                        org = torch.tensor([[[0,0],
                                        [w,0],
                                        [0,h],
                                        [w,h]]], dtype = torch.float32)

                        dst = torch.tensor([[[0+w2,0-h2],
                                        [w-w2,0+h2],
                                        [0+w2,h+h2],
                                        [w-w2,h-h2]]], dtype = torch.float32)

                        # compute perspective transform
                        M = kornia.get_perspective_transform(org, dst).to(self.config.device)

                        # warp the original image by the found transform
                        patch_pers_transform = kornia.warp_perspective(patch_transform.unsqueeze(0), M, dsize=(h, w)).squeeze()
                        ori_stop_perspective = kornia.warp_perspective(original_stop.unsqueeze(0), M, dsize=(h, w)).squeeze()
                        ori_stop2_pers = kornia.warp_perspective(ori_stop2.unsqueeze(0), M, dsize=(h, w)).squeeze()

                    else:
                        patch_pers_transform = patch_transform
                        ori_stop_perspective[:,:,:] = original_stop[:,:,:]
                        ori_stop2_pers[:,:,:] = ori_stop2[:,:,:]
                    if self.config.debug: print("patch_pers_transform", patch_pers_transform.requires_grad)

                
                    img_ori = get_random_img_ori(self.imlist_back).to(self.config.device)
                    ratio = random.uniform(0.1, 0.5)
                    x_c = random.randint(99,400-int(ratio*(100+201)))# x_c=random.randint(10+int(ratio*radius),400-int(ratio*radius))
                    y_c = random.randint(208-25,300)#y_c=random.randint(10+int(ratio*radius),400-int(ratio*radius))
                    width_r = math.ceil(ratio*width)
                    height_r = math.ceil(ratio*width)
                    width_pole_r = math.ceil(ratio*width_pole)
                    height_pole_r = math.ceil(ratio*height_pole)
                    if(width_r % 2 == 0):
                        width_r = width_r + 1
                    if(height_r % 2 == 0):
                        height_r = height_r + 1
                    if(width_pole_r % 2 == 0):
                        width_pole_r = width_pole_r + 1
                    if(height_r % 2 == 0):
                        width_pole_r = width_pole_r + 1
                            
                    patch_resize = F.interpolate(patch_pers_transform.unsqueeze(0), (width_r,height_r)).squeeze()
                    if self.config.debug: print("patch_resize", patch_resize.requires_grad)

                    patch_pole_resize = F.interpolate(patch_pole_transform.unsqueeze(0), (width_pole_r,height_pole_r)).squeeze()
                    ori_stop_resize = F.interpolate(ori_stop_perspective.unsqueeze(0), (width_r,height_r)).squeeze()
                    ori_stop2_resize = F.interpolate(ori_stop2_pers.unsqueeze(0), (width_r,height_r)).squeeze()
                    map_4_patches_resize = F.interpolate(map_4_patches.unsqueeze(0), (width_r,height_r)).squeeze()


                    map_resize[:,:,:,:]=map_ori[:,:,:,:]
                    start_x=int(x_c-(width_r-1)/2)
                    end_x=int(x_c+(width_r-1)/2+1)
                    start_y=int(y_c-(height_r-1)/2)
                    end_y=int(y_c+(height_r-1)/2+1)
                    start_pole_y=int(y_c-(height_pole_r-1)/2)
                    end_pole_y=int(y_c+(height_pole_r-1)/2+1)
                    start_pole_x=int(x_c+(width_r-1)/2+1)
                    end_pole_x=int(x_c+(width_r-1)/2+width_pole_r+1)
                    # print(start_x,end_x,start_y,end_y)

                    # map_resize just for saturation calculation
                    map_resize[0,:,start_x:end_x,start_y:end_y]=map_4_patches_resize
                    img_input[:,:,:,:]=img_ori[:,:,:,:]
                    img_input2[:,:,:,:]=img_ori[:,:,:,:]

                    # get four corners of stop
                    stop_4=torch.sum(ori_stop_resize[:,:,:],0)
                    stop_4=(stop_4<0.1).float().unsqueeze(0)
                    stop_4=torch.cat((stop_4,stop_4,stop_4),0)
                    # img_input[0,:,start_x:end_x,start_y:end_y]=torch.clamp((patch_resize+map_character_resize),0,1)+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
                    # adv_stop_img = patch_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
                    img_input[0,:,start_x:end_x,start_y:end_y] = patch_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
                    img_input[0,:,start_pole_x:end_pole_x,start_pole_y:end_pole_y]=patch_pole_resize

                    # ori_stop_img = ori_stop2_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4                 
                    img_input2[0,:,start_x:end_x,start_y:end_y] = ori_stop2_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
                    img_input2[0,:,start_pole_x:end_pole_x,start_pole_y:end_pole_y]=patch_pole_resize
                    
                    if self.config.optimizer == "fgsm":
                        input1=Variable(img_input, requires_grad=True)
                    else:
                        input1=img_input
                    input2=Variable(img_input2, requires_grad=False)
                    if self.config.debug: print("input1", input1.requires_grad)

                    #forward
                    rn_noise=torch.from_numpy(np.random.uniform(-0.1,0.1,size=(1,3,416,416))).float().to(self.config.device)
                    prediction,feature_out = self.model(torch.clamp(input1+rn_noise,0,1), self.config.CUDA)
                    prediction_target,feature_target = self.model(torch.clamp(input2+rn_noise,0,1), self.config.CUDA)
                    det_loss, loss_dis, satur_loss, ind_nz= self.get_loss(self.config.device, prediction, adv_label, self.config.ori_index, input1, map_resize)
                    fir_loss = self.get_feature_dist(start_x,end_x,start_y,end_y,feature_out,feature_target) 
                    dist_loss = torch.dist(patch_fix, original_stop0, 2)
                    
                    tvcomp1 = torch.sum(torch.abs(patch_four[:, :, 1:] - patch_four[:, :, :-1]+0.000001),0)
                    tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
                    tvcomp2 = torch.sum(torch.abs(patch_four[:, 1:, :] - patch_four[:, :-1, :]+0.000001),0)
                    tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
                    tv = tvcomp1 + tvcomp2
                    tv_loss =  tv/torch.numel(patch_four)
                    
                    # calculate euclidian distance between colors in patch and colors in printability_array 
                    # square root of sum of squared difference
                    color_dist = (patch_four - printability_array+0.000001)
                    color_dist = color_dist ** 2
                    color_dist = torch.sum(color_dist, 1)+0.000001
                    color_dist = torch.sqrt(color_dist)
                    # only work with the min distance
                    color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
                    # calculate the nps by summing over all pixels
                    nps_score = torch.sum(color_dist_prod,0)
                    nps_score = torch.sum(nps_score,0)
                    nps_loss =  nps_score/torch.numel(patch_four)
                    
                    
                    print('det_loss:', det_loss)
                    print('fir_loss:', fir_loss)
                    print('dist_loss:', dist_loss)
                    print('tv_loss:', tv_loss)
                    print('nps_loss:', nps_loss)
                    print('satur_loss:', satur_loss)
                    
                    loss = det_loss
                    
                    if self.config.fir_flag:
                        adap = self.config.fir_p*float(det_loss.data/(1/fir_loss.data))
                        loss = loss + adap*(1/fir_loss)

                    if self.config.dist_flag:
                        adap = self.config.dist_p*float(det_loss.data/dist_loss.data)
                        loss = loss + adap*(dist_loss)

                    if self.config.tv_flag:
                        adap = self.config.tv_p*float(det_loss.data/tv_loss.data)
                        loss = loss + adap*(tv_loss)

                    if self.config.nps_flag:
                        adap = self.config.nps_p*float(det_loss.data/nps_loss.data)
                        loss = loss + adap*(nps_loss)

                    if self.config.satur_flag:
                        adap = self.config.satur_p*float(det_loss.data/satur_loss.data)
                        loss = loss + adap*(satur_loss)

                    if self.config.debug: print("loss", loss.requires_grad)
                    if self.config.debug: print("patch_four", patch_four.requires_grad)
                    print("backward")
                    loss.backward()
                    # print("After loss.backward(): patch_four grad data: ", patch_four.grad.data)

                    if self.config.optimizer == "fgsm" and self.config.batch_variation:
                        input_grad = input1.grad.data
                        # print("input_grad", input_grad)
                        # input_grad = torch.sign(input_grad)

                        #inverse_rescale
                        grad_resize1 = input_grad[0,:,start_x:end_x,start_y:end_y]
                        grad_resize1 = cv2.resize(grad_resize1.cpu().numpy().transpose(1,2,0),(width,height),cv2.INTER_CUBIC)
                        if(perspective == 1):
                           perspective = 0
                           grad_resize1 = inverse_perspective_transform(grad_resize1,org,dst)
                        grad_resize1 = torch.from_numpy(grad_resize1.transpose(2,0,1)).to(self.config.device)
                        if j==0:
                           grad_resize = grad_resize1          
                        else:
                           grad_resize += grad_resize1

                    if self.config.optimizer == "fgsm" and not self.config.batch_variation:
                        input_grad = input1.grad.data
                        # print("input_grad", input_grad)
                        # input_grad = torch.sign(input_grad)

                        #inverse_rescale
                        grad_resize1 = input_grad[0,:,start_x:end_x,start_y:end_y]
                        grad_resize1 = cv2.resize(grad_resize1.cpu().numpy().transpose(1,2,0),(width,height),cv2.INTER_CUBIC)
                        if(perspective == 1):
                           perspective = 0
                           grad_resize1 = inverse_perspective_transform(grad_resize1,org,dst)
                        grad_resize1 = torch.from_numpy(grad_resize1.transpose(2,0,1)).to(self.config.device)
                        grad_resize = grad_resize1          

                        grad_sum = torch.sum(grad_resize)
                        if grad_sum == 0:
                            print("WARNING: ZERO GRADIENT")
                            return
                        print("GRADIENT sum:", grad_sum)

                        # add epsilon
                        epsilon = 0.05 / (math.floor(i/100) + 1)
                        grad_4_patches = grad_resize * map_4_patches
                        # print(grad_resize)
                        # print(torch.sign(grad_4_patches))
                        epsilon_4_patches = epsilon * torch.sign(grad_4_patches) #FGSM attack
                        patch_four = patch_four - epsilon_4_patches * map_4_patches
                        patch_four = torch.clamp(patch_four, 0, 1)
                        patch_f = patch_four
                        
                    elif self.config.optimizer == "adam" and not self.config.batch_variation:
                    # no batch variation
                        grad_sum = torch.sum(patch_four.grad.data)
                        if grad_sum == 0:
                            print("WARNING: ZERO GRADIENT")
                            return
                        else: 
                            print("GRADIENT sum", grad_sum)
                        optimizer.step()
                        # print("After optimizer.step(): patch_four grad data: ", patch_four.grad.data)
                        optimizer.zero_grad()
                        # print("After zero_grad: patch_four grad data: ", patch_four.grad.data)
                        # patch_four=patch_four*map_4_patches
                        patch_four.data.clamp_(0,1)       #keep patch in image range
                    
                    epoch_total_loss += loss
                    epoch_det_loss += det_loss
                    epoch_fir_loss += fir_loss
                    epoch_dist_loss += dist_loss
                    epoch_tv_loss += tv_loss
                    epoch_nps_loss += nps_loss
                    epoch_satur_loss += satur_loss

                    # if (j + 1) % 5 == 0:
                    #     iteration = self.config.batch_size * i + j
                    #     self.writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/fir_loss', fir_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/dist_loss', dist_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/satur_loss', satur_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('misc/epoch', i, iteration)

#                     del img_input, img_input2, ori_stop_perspective, ori_stop2, ori_stop2_pers, patch_transform, patch_pers_transform, patch_pole_transform, input2

                avg_total_loss = epoch_total_loss/self.config.batch_size
                avg_det_loss = epoch_det_loss/self.config.batch_size
                avg_fir_loss = epoch_fir_loss/self.config.batch_size
                avg_dist_loss = epoch_dist_loss/self.config.batch_size
                avg_tv_loss = epoch_tv_loss/self.config.batch_size
                avg_nps_loss = epoch_nps_loss/self.config.batch_size
                avg_satur_loss = epoch_satur_loss/self.config.batch_size
            
                print('\nDone epoch, i:', i)
                print('avg_total_loss:',avg_total_loss)
                
                if self.config.optimizer == "fgsm" and self.config.batch_variation:
                    grad_resize = grad_resize/self.config.batch_size  #need to average the gradient?
                    avg_grad_sum = torch.sum(grad_resize)
                    print("AVERAGE GRADIENT, sum:", avg_grad_sum)

                    # add epsilon
                    epsilon = 0.05 / (math.floor(i/100) + 1)
                    grad_4_patches = grad_resize * map_4_patches
                    # print(grad_resize)
                    # print(torch.sign(grad_4_patches))
                    epsilon_4_patches = epsilon * torch.sign(grad_4_patches) #FGSM attack
                    patch_four = patch_four - epsilon_4_patches * map_4_patches
                    patch_four = torch.clamp(patch_four, 0, 1)
                    patch_f = patch_four

                    #random original _stop
                    original_stop = get_random_stop_ori(self.imlist_stop).to(self.config.device)
                    original_stop = original_stop[0, :, :, :]
                    patch_fix = original_stop * map_4_stop + patch_four
                    patch_fix = torch.clamp(patch_fix, 0, 1)
            
                elif self.config.optimizer == "adam" and self.config.batch_variation:
                # batch variation update #need to average the grad_data in patch_four?
                    print("Before averaging the gradients: ")
                    grad_sum = torch.sum(patch_four.grad.data)           
                    if grad_sum == 0:
                        print("WARNING: ZERO GRADIENT")
                        return
                    else: 
                        print("ACCUMULATED GRADIENT", grad_sum)

                    patch_four.grad.data /= self.config.batch_size
                    print("After averaging the gradients: ")
                    avg_grad_sum = torch.sum(patch_four.grad.data)           
                    if avg_grad_sum == 0:
                        print("WARNING: ZERO GRADIENT")
                        return
                    else: 
                        print("AVERAGE GRADIENT, sum:", avg_grad_sum)

                    optimizer.step()
                    print("After optimizer.step()")
                    grad_sum = torch.sum(patch_four.grad.data)
                    print("VALID GRADIENT", grad_sum)

                    optimizer.zero_grad()
                    print("After zero_grad:")
                    grad_sum = torch.sum(patch_four.grad.data)
                    if grad_sum == 0:
                        print("ZERO GRADIENT")
                    else: 
                        print("NON-ZERO GRADIENT", grad_sum)
                    # patch_four=patch_four*map_4_patches
                    patch_four.data.clamp_(0,1)       #keep patch in image range


                    # epsilon = optimizer.param_groups[0]['lr']
                    if i%20 == 0:
                        # print("New learning rate: ")
                        scheduler.step(avg_total_loss)
                        # print(optimizer.param_groups[0]['lr'])                
                    

                epoch_end = time.time()
                t = epoch_end - epoch_start
                print("One epoch time taken: ", t)
                
               
                test_start_time = time.time()
                if self.config.run_test and (i + 1) % self.config.test_interval == 0:                  
                    suc_rate, suc_step, total_frames = self.yolo_test(self.config.ntests, copy.deepcopy(patch_four))
                    print("suc_rate, suc_step, total_frames", suc_rate, suc_step, total_frames)
                test_end_time = time.time()
                init_time += test_end_time - test_start_time
                    
                torch.cuda.empty_cache()
                
                if (i + 1) % self.config.save_interval == 0:
                    curr_time = time.time()
                    accumulated_time = (curr_time - init_time)/60
                    # iteration = self.config.batch_size * (i+1)
                    self.writer.add_scalar('loss/total_loss', avg_total_loss.detach().cpu().numpy(), i)
                    self.writer.add_scalar('loss/det_loss', avg_det_loss.detach().cpu().numpy(), i)
                    self.writer.add_scalar('loss/fir_loss', avg_fir_loss.detach().cpu().numpy(), i) 
                    self.writer.add_scalar('loss/dist_loss', avg_dist_loss.detach().cpu().numpy(), i)
                    self.writer.add_scalar('loss/tv_loss', avg_tv_loss.detach().cpu().numpy(), i)
                    self.writer.add_scalar('loss/nps_loss', avg_nps_loss.detach().cpu().numpy(), i)
                    self.writer.add_scalar('loss/satur_loss', avg_satur_loss.detach().cpu().numpy(), i)

                    # self.writer.add_scalar('misc/learning_rate', epsilon, i)
                    self.writer.add_scalar('misc/duration', round(accumulated_time, 3), i)
                    self.writer.add_scalar('misc/suc_rate', round(suc_rate, 3), i)
                    self.writer.add_image('adv_stop', patch_fix.squeeze(), i)
                    self.writer.add_image('patch', patch_f.squeeze(), i)
                    self.writer.add_image('adv_img', input1.squeeze(), i)
                    

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            return patch_fix, input1

        return patch_fix, input1
        #return patch_fix,output_adv

    def get_test_input(self, input_dim, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (input_dim, input_dim))

        img_ =  img.transpose((2,0,1))
        img_ = img_[np.newaxis,:,:,:]/255.0
        img_ = torch.from_numpy(img_).float()
        img_ = Variable(img_)
        img_ = img_.to(self.config.device)
        return img_

    def get_stop_patch(self, input_dim):
        img = cv2.imread(self.config.path_ori_stop)
        img = cv2.resize(img, (input_dim, input_dim))
        img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
        img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        stop_ori=img_out.to(self.config.device)

        width=input_dim
        height=input_dim
        map_ori=torch.zeros(1,3,width,height).to(self.config.device)
        #get character image
        #   map_character_ori=torch.zeros(1,3,width,height).to(self.config.device)
        
        #  control the ratio of the patch on stop sign
        if self.config.patch_ratio == 0.10:
            map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.444)]=1#rec_90  #0.10
            map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.444)]=1
        elif self.config.patch_ratio == 0.20:
            map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.58)]=1#rec_70  #0.20
            map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.58)]=1
        elif self.config.patch_ratio == 0.25:
            map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.674)]=1#rec_90   #0.25
            map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.674)]=1
        elif self.config.patch_ratio == 0.30:
            map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.72)]=1#rec_100  #0.30
            map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.72)]=1
        else:
            # default to 0.25
            map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.674)]=1#rec_90   #0.25
            map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.674)]=1            

        map_stop=-(map_ori-1)

        patch_stop=stop_ori*map_ori
        return stop_ori[0,:,:,:],map_ori[0,:,:,:],map_stop[0,:,:,:],patch_stop[0,:,:,:]   #output:original stop, map mask for patch, map mask for stop, four patch
        #original_stop,map_4_patches,map_4_stop,patch_four

    def get_stop_patch_from_mask(self, input_dim):
        stop_ori = cv2.imread(self.config.path_ori_stop)
        map4patch = cv2.imread(self.config.path_map4patch)
        map4stop = cv2.imread(self.config.path_map4stop)
        
        stop_ori = cv2.resize(stop_ori, (input_dim, input_dim))
        map4patch = cv2.resize(map4patch, (input_dim, input_dim))
        map4stop = cv2.resize(map4stop, (input_dim, input_dim))

        stop_ori = self.image_to_inp(stop_ori)
        map4patch = self.image_to_inp(map4patch)
        map4stop = self.image_to_inp(map4stop)
        
        patch_four=stop_ori*map4patch
        return stop_ori[0,:,:,:],map4patch[0,:,:,:],map4stop[0,:,:,:],patch_four[0,:,:,:] 
        
    def get_pole(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (201, 27))
        img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
        img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_out.to(self.config.device)

    def get_printability_array(self):
        printability_list = []
        side = 201
        # read in printability triplets and put them in a list
        with open(self.config.path + "helper/print.txt") as f:
            for line in f:
                printability_list.append(line.split(","))
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)
        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa

    def image_to_inp(self, img):
        img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
        img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        img_out=img_out.to(self.config.device)
        return img_out

    def write_archor(self, x, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results
        cls = int(x[7])
        label = "{0}".format(self.classes[cls])
        colors = pkl.load(open("helper/pallete", "rb"))
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img

    def det_and_save_img_i(self, img , path, i=0):
        prediction,feature_out = self.model(img , CUDA)
        output = write_results(self.config.device, prediction.data, self.config.confidence, self.num_classes, nms = True, nms_conf = nms_thesh)#final_output,[:,8] 8=input_img_num,xc,yc,width,height,confidence,confidence_class,class

        img_save=inp_to_image(img)

        cv2.imwrite(path+'/'+str(i)+'.png', img_save)
        print("Saved img: ", path+'/'+str(i)+'.png')

        img_draw= cv2.imread(path+'/'+str(i)+'.png')
        out_save=list(map(lambda x: self.write_archor(x,img_draw), output))
        cv2.imwrite(path+'_det/'+str(i)+'.png', img_draw)
        print("Saved img: ", path+'_det/'+str(i)+'.png')

    def save_img_i(self, img, path, i=0):
        img_save = inp_to_image(img)
        cv2.imwrite(path+str(i)+'.png', img_save)
        print("Saved img: ", path+str(i)+'.png')
            
    #get_loss_disappear
    def get_loss(self, device, prediction,adv_label,ori_index,img_adv,map_bounding):
        ind_nz = get_ind(prediction, ori_index)
        loss_disappear= get_loss_disappear(prediction,ind_nz,adv_label)
        #    print('loss_creation:',loss_creation)
        #   print('loss_index',loss_index)
        #loss_smoothness=get_loss_smoothness(img_adv,map_bounding)
        loss_smoothness=0#get_loss_median(img_adv,map_bounding)
        loss_saturation=get_loss_saturation(img_adv,map_bounding, device)
        loss=loss_disappear#+loss_saturation*(1/5000)#+loss_smoothness*(1/1000)#100*100..20000  loss_smoothness*(1/10000)+
        return  loss,loss_disappear,loss_saturation,ind_nz

    def get_feature_dist(self, start_x,end_x,start_y,end_y,feature_out,feature_target):
            dist=0
            for i in range(len(feature_out)):
                size = feature_out[i].shape[2]
                start_fx=math.floor(start_x*(size/416))
                end_fx=math.ceil(end_x*(size/416))
                start_fx=np.clip(start_fx,0,size-1)
                end_fx=np.clip(end_fx,0,size-1)
                start_fy=math.floor(start_y*(size/416))
                end_fy=math.ceil(end_y*(size/416))
                start_fy=np.clip(start_fy,0,size-1)
                end_fy=np.clip(end_fy,0,size-1)

                f_sa=feature_out[i][0,:,start_fx:end_fx,start_fy:end_fy]
                f_st=feature_target[i][0,:,start_fx:end_fx,start_fy:end_fy]
            
                #mean
                f_samean = torch.sum(torch.abs(f_sa),2)
                f_samean = torch.sum(f_samean,1)/(size*size)
                f_stmean = torch.sum(torch.abs(f_st),2)
                f_stmean = torch.sum(f_stmean,1)/(size*size)
                # generalization
                cores=len(f_stmean)
                # print(cores)
                # f_stmean=f_stmean/torch.max(f_stmean)
                # print(f_stmean.shape)
                f_stmean=torch.softmax(f_stmean,0)
                # f_samean=f_samean/torch.max(f_samean)
                f_samean=torch.softmax(f_samean,0)
                dist_one=torch.sum(torch.abs(f_stmean-f_samean)**2)#/cores
                dist=dist+dist_one
                
            return dist

    def yolo_test(self, num_test, patch_test):
        patch_test=Variable(patch_test,requires_grad=False)
        start_x = 200
        start_y = 100
        width = 201
        height = 201
        radius=(width-1)/2

        x_c=100
        y_c=150
        
        width_pole=201
        height_pole=27

        #img_ori
        img_ori= self.get_test_input(self.inp_dim, self.config.path_img_ori)

        #ori_stop
        if self.config.from_mask:
            original_stop,map_4_patches,map_4_stop,patch_four=self.get_stop_patch_from_mask(self.config.patch_dim)
        else:
            original_stop,map_4_patches,map_4_stop,patch_four=self.get_stop_patch(self.config.patch_dim)      
            
        #pole
        patch_pole=torch.zeros(3,27,201).to(self.config.device)
        ori_pole = self.get_pole(self.config.path_pole)

        #patch_fix
        patch_fix=torch.zeros(3,201,201).to(self.config.device)

        #map
        map_ori=torch.zeros(1,3,416,416).to(self.config.device)
        map_resize=torch.zeros(1,3,416,416).to(self.config.device)# the map of all three patches
        
        #img_input
        img_input=torch.zeros(1,3,416,416).to(self.config.device)
        img_input2=torch.zeros(1,3,416,416).to(self.config.device)

        #perspective
        ori_stop_perspective=torch.zeros(3,width,width).to(self.config.device)
        ori_stop2=torch.zeros(3,201,201).to(self.config.device)
        ori_stop2_pers=torch.zeros(3,201,201).to(self.config.device)

        #patch
        patch_transform=img_ori[ 0,:, start_x:start_x+width,start_y:start_y+height]
        patch_pole_transform=torch.zeros(3,27,201).to(self.config.device)

        suc_step = 0
        suc_rate = 0
        total_frames = 0
        try: 
            for i in range(num_test):

                # get_random_stop_ori # cut out the patch and paste it to different stop sign
                original_stop = get_random_stop_ori(self.imlist_stop).to(self.config.device)
                original_stop = original_stop[0, :, :, :]
                patch_f = patch_test * map_4_patches
                patch_fix = original_stop * map_4_stop + patch_f
                patch_fix = torch.clamp(patch_fix, 0, 1)

                if i%2==0:
                    patch_transform[:,:,:],ori_stop2[:,:,:] = addweight_transform_multiple(patch_fix[:,:,:],original_stop[:,:,:])
                    patch_pole_transform[:,:,:]=ori_pole[:,:,:]
                    patch_transform = torch.clamp(patch_transform, 0.000001, 0.999999)
                    ori_stop2 = torch.clamp(ori_stop2, 0.000001, 0.999999)
                else:
                    # gamma_transform_multiple(patch_fix)
                    patch_transform[:,:,:] = patch_fix[:,:,:]
                    patch_pole_transform[:,:,:]=ori_pole[:,:,:]
                    ori_stop2[:,:,:]=original_stop[:,:,:]

                angle = 1000 #random
                if i % 3 == 0:
                    #perspective transform   
                    patch_transform,org,dst,angle=perspective_transform_multiple(patch_transform, set_angle=angle)
                    ori_stop_perspective,org_abandon,dst_abandon,angle_abandon=perspective_transform_multiple(original_stop,True,angle)
                    ori_stop2_pers,org_abandon,dst_abandon,angle_abandon=perspective_transform_multiple(ori_stop2,True,angle)
                else:
                    ori_stop_perspective[:,:,:]=original_stop[:,:,:]
                    ori_stop2_pers[:,:,:]=ori_stop2[:,:,:]
                #random background
                img_ori = get_random_img_ori(self.imlist_back_test).to(self.config.device)
                
                if ((i+1)/2==0):
                        ratio = random.uniform(0.2, 1)
                else:
                        ratio=random.uniform(0.2,0.5)

                # x_c=random.randint(10+int(ratio*radius),400-int(ratio*radius))
                x_c=random.randint(99,400-int(ratio*(100+201)))
                #y_c=random.randint(10+int(ratio*radius),400-int(ratio*radius))
                y_c=random.randint(208-25,300)
                width_r=math.ceil(ratio*width)
                height_r=math.ceil(ratio*width)
                width_pole_r=math.ceil(ratio*width_pole)
                height_pole_r=math.ceil(ratio*height_pole)
                if(width_r%2==0):
                    width_r=width_r+1
                if(height_r%2==0):
                    height_r=height_r+1
                if(width_pole_r%2==0):
                    width_pole_r=width_pole_r+1
                if(height_r%2==0):
                    width_pole_r=width_pole_r+1

                #patch_resize=resize(stop+patch)
                patch_resize=cv2.resize(patch_transform.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)
                #patch_pole_resize=resize(patch_pole)
                patch_pole_resize=cv2.resize(patch_pole_transform.cpu().numpy().transpose(1,2,0) ,(height_pole_r,width_pole_r),cv2.INTER_CUBIC)
                #ori_stop_resize=resize(original stop) just for the stop_4 to get four corners
                ori_stop_resize=cv2.resize(ori_stop_perspective.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)
                ori_stop2_resize=cv2.resize(ori_stop2_pers.cpu().numpy().transpose(1,2,0) ,(width_r,height_r),cv2.INTER_CUBIC)

                patch_resize=torch.from_numpy(patch_resize.transpose(2,0,1)).to(self.config.device)
                patch_pole_resize=torch.from_numpy(patch_pole_resize.transpose(2,0,1)).to(self.config.device)
                ori_stop_resize=torch.from_numpy(ori_stop_resize.transpose(2,0,1)).to(self.config.device)

                ori_stop2_resize=torch.from_numpy(ori_stop2_resize.transpose(2,0,1)).to(self.config.device)

                map_resize[:,:,:,:]=map_ori[:,:,:,:]
                start_x=int(x_c-(width_r-1)/2)
                end_x=int(x_c+(width_r-1)/2+1)
                start_y=int(y_c-(height_r-1)/2)
                end_y=int(y_c+(height_r-1)/2+1)
                start_pole_y=int(y_c-(height_pole_r-1)/2)
                end_pole_y=int(y_c+(height_pole_r-1)/2+1)
                start_pole_x=int(x_c+(width_r-1)/2+1)
                end_pole_x=int(x_c+(width_r-1)/2+width_pole_r+1)

                img_input[:,:,:,:] = img_ori[:,:,:,:]
                img_input2[:,:,:,:] = img_ori[:,:,:,:]

                # get four corners of stop
                stop_4 = torch.sum(ori_stop_resize[:,:,:],0)
                stop_4 = (stop_4<0.1).float().unsqueeze(0)
                stop_4 = torch.cat((stop_4,stop_4,stop_4),0)
                
                # img_input[0,:,start_x:end_x,start_y:end_y]=torch.clamp((patch_resize+map_character_resize),0,1)+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
                # add adv_stop and pole to background img
                img_input[0,:,start_x:end_x,start_y:end_y]=patch_resize*(1-stop_4) + img_input[0,:,start_x:end_x,start_y:end_y] * stop_4
                img_input[0,:,start_pole_x:end_pole_x,start_pole_y:end_pole_y] = patch_pole_resize

                img_input2[0,:,start_x:end_x,start_y:end_y] = ori_stop2_resize + img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
                img_input2[0,:,start_pole_x:end_pole_x,start_pole_y:end_pole_y] = patch_pole_resize

                input1 = Variable(img_input, requires_grad=True)
                input2 = Variable(img_input2, requires_grad=True)
                
                #forward
                rn_noise = torch.from_numpy(np.random.uniform(-0.1,0.1,size=(1,3,416,416))).float().to(self.config.device)
                with torch.no_grad():
                    prediction, _ = self.model(torch.clamp(input1+rn_noise,0,1), self.config.CUDA)
                    prediction2, _ = self.model(torch.clamp(input2+rn_noise,0,1), self.config.CUDA)
                detect = get_ind2(prediction, self.config.ori_index)
                detect2 = get_ind2(prediction2, self.config.ori_index)
                
                
                is_success = 0
                if detect2 != 0:
                    total_frames += 1
                    if detect == 0:
                        # print("Success")
                        suc_step += 1
                        is_success = 1
                        # det_and_save_img_i(input1, i, output_dir + "adv_img", is_success) # Adversarial Example: woth background and adversarial stop sign
                        # det_and_save_img_i(input2, i, output_dir + "ori_img", is_success)
                    # else:
                        # print("Not success")
                # else:
                    # print("Not detected in original image.")
                if total_frames > 0: 
                    suc_rate = suc_step/total_frames
                    
                if (i + 1) % 100 == 0:  
                    print("Tests i:",i)
                    print('success_rate:', suc_rate)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            return suc_rate, suc_step, total_frames

        return suc_rate, suc_step, total_frames
