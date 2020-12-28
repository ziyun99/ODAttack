
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
import itertools
import numpy as np
import cv2
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data as Data
from torch import autograd
from torch import optim
import torch.nn.functional as F

from helper.util import *
from helper.util_creation import get_loss_disappear,get_loss_creation_select, get_ind,get_loss_creation,get_loss_smoothness,get_map_bounding,get_random_img_ori,get_loss_median,get_loss_saturation,get_random_stop_ori
from helper.patch import add_patch
from helper.patch2 import  addweight_transform_multiple, perspective_transform_multiple,inverse_perspective_transform, translation_center_multiple,gamma_transform_multiple, amplify_size,shear_transform,rotate_transform,translation_transform,gamma_transform,blur_transform,transform,transform_multiple,rescale_transform_multiple
from helper.preprocess import prep_image, inp_to_image
from helper.darknet4 import Darknet

from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from torchvision import transforms
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
        print(self.config.stop_dir, ": ", len(self.imlist_stop), "stop imgs")
        print(self.config.bg_dir, ": ", len(self.imlist_back), "background imgs")

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

        self.num_classes = 80
        self.classes = load_classes(self.config.classes)

    def attack(self):
        img_ori = self.get_test_input(self.inp_dim, self.config.CUDA, os.path.join(self.config.path, "imgs/det/stop_sign.jpg"))
        prediction_dog, feature_dog = self.model(img_ori, self.config.CUDA)  # feature_archor_output
        ori_index = 11
        ind_nz_out = get_ind(prediction_dog, ori_index)  # 1=bicycle,11=stop_sign,9=traffic_light
        first_index = ind_nz_out[0]

        adv_label = [1 if i == 11 else 0 for i in range(80)] #stop sign=11,traffic_light=9
        adv_label = np.array(adv_label)
        adv_label = np.reshape(adv_label,(1,80))
        adv_label = torch.from_numpy(adv_label).float()

        s = time.time()
        patch_adv, input1 = self.get_adv_episilon(img_ori, adv_label, first_index, ori_index)
        e = time.time()
        print("time_taken: ", str(e-s))
        
        save_img(patch_adv, os.path.join(self.config.out_path, 'final/adv_stop'))
        det_and_save_img(input1, os.path.join(self.config.out_path, 'final/adv_img'))

        print("Done and exit")
                          
    def get_adv_episilon(self, img_ori,adv_label,first_index,ori_index):
        self.writer = self.init_tensorboard()
        
        fir_p = 0.3
        dist_p = 1.0
        tv_p = 0.9
        nps_p = 0.5
        satur_p = 0.5
        fir_flag = False
        dist_flag = True
        tv_flag = True
        nps_flag = False
        satur_flag = True
        
        text = "Experiment: " + self.config.exp + " | fir: " + str(fir_flag) + " " + str(fir_p) + " | dist: " + str(dist_flag) + " " + str(dist_p) + " | tv: " + str(tv_flag) + " " + str(tv_p) + " | nps: " + str(nps_flag) + " " + str(nps_p) + " | satur: " + str(satur_flag) + " " + str(satur_p)
        self.writer.add_text('param', text, 0) 

        # x_c=100
        # y_c=150

        #img_ori
        img_ori= self.get_test_input(self.inp_dim, self.config.CUDA, os.path.join(self.config.path, "imgs/det/stop_sign.jpg"))

        #ori_stop
        original_stop,map_4_patches,map_4_stop,patch_four=self.get_stop_patch(input_dim=201)
        original_stop0 = original_stop
        
        #pole
        # patch_pole=torch.zeros(3,27,201).to(device)
        ori_pole = self.get_pole(os.path.join(self.config.path,'imgs/pole/pole.jpg'))

        #patch_fix
        # patch_fix=original_stop

        patch_four=Variable(patch_four, requires_grad=True)
        # patch_fix = original_stop*map_4_stop+patch_four
        # patch_fix=torch.clamp(patch_fix,0,1)

        optimizer = optim.Adam([patch_four], lr=0.03, amsgrad=True)
        scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        scheduler = scheduler_factory(optimizer)
        print("patch_four", patch_four.requires_grad)
        
        printability_array = self.get_printability_array().to(self.config.device)
        
        init_time = time.time()
        suc_rate = 0
        try: 
            for i in range(self.config.nsteps):
                start = time.time()
                jsteps=20
                original_stop = get_random_stop_ori(self.imlist_stop).to(self.config.device)
                original_stop = original_stop[0, :, :, :]
                for j in range(jsteps):
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
                    print("patch_transform", patch_transform.requires_grad)
                    
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
                    print("patch_pers_transform", patch_pers_transform.requires_grad)

                
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
                            
                    print(width_r,height_r)
                    patch_resize = F.interpolate(patch_pers_transform.unsqueeze(0), (width_r,height_r)).squeeze()
                    print("patch_resize", patch_resize.requires_grad)

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

                    #map_resize just for saturation calculation
                    map_resize[0,:,start_x:end_x,start_y:end_y]=map_4_patches_resize
                    img_input[:,:,:,:]=img_ori[:,:,:,:]
                    img_input2[:,:,:,:]=img_ori[:,:,:,:]

                    #     get four corners of stop
                    stop_4=torch.sum(ori_stop_resize[:,:,:],0)
                    stop_4=(stop_4<0.1).float().unsqueeze(0)
                    stop_4=torch.cat((stop_4,stop_4,stop_4),0)
                    #   img_input[0,:,start_x:end_x,start_y:end_y]=torch.clamp((patch_resize+map_character_resize),0,1)+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
    #                 adv_stop_img = patch_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
                    img_input[0,:,start_x:end_x,start_y:end_y] = patch_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
                    img_input[0,:,start_pole_x:end_pole_x,start_pole_y:end_pole_y]=patch_pole_resize

    #                 ori_stop_img = ori_stop2_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4                 
                    img_input2[0,:,start_x:end_x,start_y:end_y] = ori_stop2_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
                    img_input2[0,:,start_pole_x:end_pole_x,start_pole_y:end_pole_y]=patch_pole_resize
                    
                    input1=img_input
                    input2=Variable(img_input2, requires_grad=False)
                    print("input1", input1.requires_grad)

                    #forward
                    rn_noise=torch.from_numpy(np.random.uniform(-0.1,0.1,size=(1,3,416,416))).float().to(self.config.device)
                    prediction,feature_out = self.model(torch.clamp(input1+rn_noise,0,1), self.config.CUDA)
                    prediction_target,feature_target = self.model(torch.clamp(input2+rn_noise,0,1), self.config.CUDA)
                    det_loss, loss_dis, satur_loss, ind_nz= get_loss(self.config.device, prediction, adv_label,ori_index,input1,map_resize)
                    fir_loss = get_feature_dist(start_x,end_x,start_y,end_y,feature_out,feature_target) 
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
                    
                    loss = det_loss
                    
                    if fir_flag:
                        adap=fir_p*float(det_loss.data/(1/fir_loss.data))
                        loss=det_loss+adap*(1/fir_loss)
                        print('adap:', adap)
                        print('1/fir_loss:', 1/fir_loss)
                        print('loss2:', loss)
    
                    if dist_flag:
                        adap=dist_p*float(det_loss.data/dist_loss.data)
                        loss=loss+adap*(dist_loss)
                        print('adap:', adap)           
                        print('loss3:', loss)
                
                    if tv_flag:
                        adap=tv_p*float(det_loss.data/tv_loss.data)
                        loss=loss+adap*(tv_loss)
                        print('adap:', adap)           
                        print('loss4:', loss)
                    
                    if nps_flag:
                        adap=nps_p*float(det_loss.data/nps_loss.data)
                        loss=loss+adap*(nps_loss)
                        print('adap:', adap)           
                        print('loss5:', loss)
                                    
                    if satur_flag:
                        adap=satur_p*float(det_loss.data/satur_loss.data)
                        loss=loss+adap*(satur_loss)
                        print('adap:', adap)           
                        print('loss6:', loss)
                        
                    print("loss", loss.requires_grad)
                    print("patch_four", patch_four.requires_grad)
    #                 print("patch_four", patch_four.shape, patch_fix.shape, input1.shape)
                    print("backward")
                    loss.backward()
    #                 print("After loss.backward(): patch_four grad data: ", patch_four.grad.data)

                    # no batch variation
    #                 grad_sum = torch.sum(patch_four.grad.data)
    #                 if grad_sum == 0:
    #                     print("WARNING: ZERO GRADIENT")
    #                     return
    #                 else: 
    #                     print("ACCUMULATED GRADIENT", grad_sum)
                    
    #                 optimizer.step()
    # #                 print("After optimizer.step(): patch_four grad data: ", patch_four.grad.data)
    #                 optimizer.zero_grad()
    # #                 print("After zero_grad: patch_four grad data: ", patch_four.grad.data)
    # #                 patch_four=patch_four*map_4_patches
    #                 patch_four.data.clamp_(0,1)       #keep patch in image range
                                
                    if j==0:
                        epoch_total_loss = loss
                        epoch_det_loss = det_loss
                        epoch_fir_loss = fir_loss
                        epoch_dist_loss = dist_loss
                    
                    else:
                        epoch_total_loss += loss
                        epoch_det_loss += det_loss
                        epoch_fir_loss += fir_loss
                        epoch_dist_loss += dist_loss
                        

                    if j%5 == 0:
                        iteration = jsteps * i + j
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/fir_loss', fir_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/dist_loss', dist_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/satur_loss', satur_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', i, iteration)
    #                     self.writer.add_scalar('misc/accumulated_gradient', grad_sum, iteration)
                        
    #                     if(addweight_transform == "False"):
    #                         writer.add_image('misc/patch_fix', patch_fix.squeeze(), iteration) 
    #                         writer.add_image('misc/patch_transform', patch_transform.squeeze(), iteration) 
    #                         writer.add_image('misc/patch_pers_transform', patch_pers_transform.squeeze(), iteration) 
    #                         writer.add_image('misc/patch_resize', patch_resize.squeeze(), iteration) 
    #                         writer.add_image('misc/ori_stop2_resize', ori_stop2_resize.squeeze(), iteration) 
    #                         writer.add_image('misc/stop_4', stop_4.squeeze(), iteration) 
    # #                         writer.add_image('misc/adv_stop_img', adv_stop_img.squeeze(), iteration) 
    #                         writer.add_image('misc/input1', input1.squeeze(), iteration) 
    #                         writer.add_image('misc/input2', input2.squeeze(), iteration) 

    #                         writer.add_text('misc/addweight_transform', addweight_transform, iteration) 

    #                 del loss1, loss2, patch_transform

                avg_total_loss = epoch_total_loss/jsteps
                avg_det_loss = epoch_det_loss/jsteps
                avg_fir_loss = epoch_fir_loss/jsteps
                avg_dist_loss = epoch_dist_loss/jsteps
            
                print('\nEpoch, i:', i)
                print('avg_total_loss:',avg_total_loss)
                
                # batch variation update
    #             need to average the grad_data in patch_four?
                print("Before averaging the gradients: ")
                grad_sum = torch.sum(patch_four.grad.data)           
                if grad_sum == 0:
                    print("WARNING: ZERO GRADIENT")
                    return
                else: 
                    print("ACCUMULATED GRADIENT", grad_sum)
                        
                patch_four.grad.data /= jsteps
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
                    
    #             patch_four=patch_four*map_4_patches
                patch_four.data.clamp_(0,1)       #keep patch in image range
            
        
    #             epsilon = optimizer.param_groups[0]['lr']
                if i%20 == 0:
    #                 print("New learning rate: ")
                    scheduler.step(avg_total_loss)
    #                 print(optimizer.param_groups[0]['lr'])                
                

                end = time.time()
                t = end - start
                print("time taken: ", t)
                
                
                if i > 0 and i % 250 == 0:
                    suc_rate, suc_step, total_frames = yolo_test(1000, copy.deepcopy(patch_four))
                    print("suc_rate, suc_step, total_frames", suc_rate, suc_step, total_frames)
                    
                torch.cuda.empty_cache()
                
                if i % 5 == 0:
                    end_time = time.time()
                    t = (end_time - init_time)/60
                    iteration = jsteps * (i+1)
                    self.writer.add_scalar('avg/total_loss', avg_total_loss.detach().cpu().numpy(), i)
                    self.writer.add_scalar('avg/det_loss', avg_det_loss.detach().cpu().numpy(), i)
                    self.writer.add_scalar('avg/fir_loss', avg_fir_loss.detach().cpu().numpy(), i) 
                    self.writer.add_scalar('avg/dist_loss', avg_dist_loss.detach().cpu().numpy(), i)

    #                 self.writer.add_scalar('misc/learning_rate', epsilon, i)
                    self.writer.add_scalar('misc/duration', round(t, 3), i)
                    self.writer.add_scalar('misc/suc_rate', round(suc_rate, 3), i)
                    self.writer.add_scalar('misc/avg_grad_sum', avg_grad_sum, i)
                    self.writer.add_image('adv_stop', patch_fix.squeeze(), i)
                    self.writer.add_image('patch', patch_f.squeeze(), i)
                    self.writer.add_image('adv_img', input1.squeeze(), i)
                    

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            return patch_fix, input1

        return patch_fix, input1
        #return patch_fix,output_adv

    def get_test_input(self, input_dim, CUDA, img):
        img = cv2.imread(img)
        # img = cv2.imread("det/stop_sign.png")
        img = cv2.resize(img, (input_dim, input_dim))

        img_ =  img.transpose((2,0,1))
        img_ = img_[np.newaxis,:,:,:]/255.0
        img_ = torch.from_numpy(img_).float()
        img_ = Variable(img_)

    #     if CUDA:
        img_ = img_.to(self.config.device)
        return img_

    def init_tensorboard(self,name=None):
    #     subprocess.Popen(['tensorboard', '--host 0.0.0.0 --port 8080 --logdir ./runs'])
    #     attack_name = "adam" + "_batch" + "_dist" + "test"
        time_str = time.strftime("%Y%m%d-%H%M%S")
        return SummaryWriter(f'{self.config.logdir}/{time_str}_{self.config.exp}')
    #     return SummaryWriter(comment=attack_name)

    def get_stop_patch(self, input_dim=201):
        # images='stop/'
        img = cv2.imread(os.path.join(self.config.path,'imgs/stop/stop1.jpg'))
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
        #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.58)]=1#rec_70  #ratio: 0.20
        #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.58)]=1
        
        #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.72)]=1#rec_100  #0.29
        #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.72)]=1
        #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.444)]=1#rec_90  #0.11
        #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.444)]=1
        map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.674)]=1#rec_90   #0.26
        map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.674)]=1
        map_stop=-(map_ori-1)

        patch_stop=stop_ori*map_ori
        return stop_ori[0,:,:,:],map_ori[0,:,:,:],map_stop[0,:,:,:],patch_stop[0,:,:,:]   #output:original stop, map mask for patch, map mask for stop, four patch
        #original_stop,map_4_patches,map_4_stop,patch_four

    def get_pole(self, character_path):
        img = cv2.imread(character_path)
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

#get_loss_disappear
def get_loss(device, prediction,adv_label,ori_index,img_adv,map_bounding):
    ind_nz = get_ind(prediction, ori_index)
    loss_disappear= get_loss_disappear(prediction,ind_nz,adv_label)
    #    print('loss_creation:',loss_creation)
    #   print('loss_index',loss_index)
    #loss_smoothness=get_loss_smoothness(img_adv,map_bounding)
    loss_smoothness=0#get_loss_median(img_adv,map_bounding)
    loss_saturation=get_loss_saturation(img_adv,map_bounding, device)
    loss=loss_disappear#+loss_saturation*(1/5000)#+loss_smoothness*(1/1000)#100*100..20000  loss_smoothness*(1/10000)+
    return  loss,loss_disappear,loss_saturation,ind_nz

def get_feature_loss(start_x,end_x,start_y,end_y,feature_out,feature_target):
        feature0=feature_out[0]#208*208*64
        feature1=feature_out[1]#104*104*128
        feature2=feature_out[2]#52*52*256
        feature3=feature_out[3]#26*26*512
        feature4=feature_out[4]#13*13*1024

        featuret0=feature_target[0].data#208*208*64
        featuret1=feature_target[1].data#104*104*128
        featuret2=feature_target[2].data#52*52*256
        featuret3=feature_target[3].data#26*26*512
        featuret4=feature_target[4].data#13*13*1024

        start_fx=math.floor(start_x*(208/416))
        end_fx=math.ceil(end_x*(208/416))
        start_fx=np.clip(start_fx,0,207)
        end_fx=np.clip(end_fx,0,207)
        start_fy=math.floor(start_y*(208/416))
        end_fy=math.ceil(end_y*(208/416))
        start_fy=np.clip(start_fy,0,207)
        end_fy=np.clip(end_fy,0,207)
        # print(feature0.shape)
        # print('start_fx:',start_fx)
        # print('end_fx:',end_fx)
        # print('start_fy:',start_fy)
        # print('end_fy:',end_fy)
        f0=feature0[0,:,start_fx:end_fx,start_fy:end_fy]
        # print(f0.shape)
        f0=torch.reshape(f0, [f0.shape[0]*f0.shape[1]*f0.shape[2]])
        f0=torch.softmax(f0,0)
        ft0=featuret0[0,:,start_fx:end_fx,start_fy:end_fy]
        ft0=torch.reshape(ft0, [ft0.shape[0]*ft0.shape[1]*ft0.shape[2]])
        ft0=torch.softmax(ft0,0)
        loss0=torch.sum(torch.abs(f0-ft0))#/f0.shape[0]

        start_fx=math.floor(start_x*(104/416))
        end_fx=math.ceil(end_x*(104/416))
        start_fx=np.clip(start_fx,0,103)
        end_fx=np.clip(end_fx,0,103)
        start_fy=math.floor(start_y*(104/416))
        end_fy=math.ceil(end_y*(104/416))
        start_fy=np.clip(start_fy,0,103)
        end_fy=np.clip(end_fy,0,103)

        f1=feature1[0,:,start_fx:end_fx,start_fy:end_fy]
        f1=torch.reshape(f1, [f1.shape[0]*f1.shape[1]*f1.shape[2]])
        f1=torch.softmax(f1,0)
        ft1=featuret1[0,:,start_fx:end_fx,start_fy:end_fy]
        ft1=torch.reshape(ft1, [ft1.shape[0]*ft1.shape[1]*ft1.shape[2]])
        ft1=torch.softmax(ft1,0)
        loss1=torch.sum(torch.abs(f1-ft1))#/f1.shape[0]

        start_fx=math.floor(start_x*(52/416))
        end_fx=math.ceil(end_x*(52/416))
        start_fx=np.clip(start_fx,0,51)
        end_fx=np.clip(end_fx,0,51)
        start_fy=math.floor(start_y*(52/416))
        end_fy=math.ceil(end_y*(52/416))
        start_fy=np.clip(start_fy,0,51)
        end_fy=np.clip(end_fy,0,51)

        # print(feature2.shape)
        # print('start_fx:',start_fx)
        # print('end_fx:',end_fx)
        # print('start_fy:',start_fy)
        # print('end_fy:',end_fy)

        f2=feature2[0,:,start_fx:end_fx,start_fy:end_fy]
        # print(f2.shape)
        f2=torch.reshape(f2, [f2.shape[0]*f2.shape[1]*f2.shape[2]])
        f2=torch.softmax(f2,0)
        ft2=featuret2[0,:,start_fx:end_fx,start_fy:end_fy]
        ft2=torch.reshape(ft2, [ft2.shape[0]*ft2.shape[1]*ft2.shape[2]])
        ft2=torch.softmax(ft2,0)
        loss2=torch.sum(torch.abs(f2-ft2))#/f2.shape[0]

        start_fx=math.floor(start_x*(26/416))
        end_fx=math.ceil(end_x*(26/416))
        start_fx=np.clip(start_fx,0,25)
        end_fx=np.clip(end_fx,0,25)
        start_fy=math.floor(start_y*(26/416))
        end_fy=math.ceil(end_y*(26/416))
        start_fy=np.clip(start_fy,0,25)
        end_fy=np.clip(end_fy,0,25)

        f3=feature3[0,:,start_fx:end_fx,start_fy:end_fy]
        f3=torch.reshape(f3, [f3.shape[0]*f3.shape[1]*f3.shape[2]])
        f3=torch.softmax(f3,0)
        ft3=featuret3[0,:,start_fx:end_fx,start_fy:end_fy]
        ft3=torch.reshape(ft3, [ft3.shape[0]*ft3.shape[1]*ft3.shape[2]])
        ft3=torch.softmax(ft3,0)
        loss3=torch.sum(torch.abs(f3-ft3))#/f3.shape[0]

        start_fx=math.floor(start_x*(13/416))
        end_fx=math.ceil(end_x*(13/416))
        start_fx=np.clip(start_fx,0,12)
        end_fx=np.clip(end_fx,0,12)
        start_fy=math.floor(start_y*(13/416))
        end_fy=math.ceil(end_y*(13/416))
        start_fy=np.clip(start_fy,0,12)
        end_fy=np.clip(end_fy,0,12)

        f4=feature4[0,:,start_fx:end_fx,start_fy:end_fy]
        f4=torch.reshape(f4, [f4.shape[0]*f4.shape[1]*f4.shape[2]])
        #  f4=torch.softmax(f4,0)
        f4=f4/torch.max(torch.abs(f4))
        ft4=featuret4[0,:,start_fx:end_fx,start_fy:end_fy]
        ft4=torch.reshape(ft4, [ft4.shape[0]*ft4.shape[1]*ft4.shape[2]])
        #  ft4=torch.softmax(ft4,0)
        ft4=ft4/torch.max(torch.abs(ft4))
        loss4=torch.sum(torch.abs(f4-ft4))#/f4.shape[0]
        loss=50/(loss2+loss3+loss4+0.000001)#*(1/5)
        return loss

def get_feature_dist(start_x,end_x,start_y,end_y,feature_out,feature_target):
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

def det_and_save_img_i(img,i,path):
    prediction,feature_out = model(img , CUDA)
    output = write_results(device, prediction.data, confidence, num_classes, nms = True, nms_conf = nms_thesh)#final_output,[:,8] 8=input_img_num,xc,yc,width,height,confidence,confidence_class,class

    img_save=inp_to_image(img)

    cv2.imwrite(path+'/'+str(i)+'.png', img_save)
    print("Saved img: ", path+'/'+str(i)+'.png')

    img_draw= cv2.imread(path+'/'+str(i)+'.png')
    out_save=list(map(lambda x: write_archor(x,img_draw), output))
    cv2.imwrite(path+'_det/'+str(i)+'.png', img_draw)
    print("Saved img: ", path+'_det/'+str(i)+'.png')

def det_and_save_img(img, path):
    prediction,feature_out = model(img , CUDA)
    output = write_results(device, prediction.data, confidence, num_classes, nms = True, nms_conf = nms_thesh)#final_output,[:,8] 8=input_img_num,xc,yc,width,height,confidence,confidence_class,class

    img_save=inp_to_image(img)

    cv2.imwrite(path+'.png', img_save)
    print("Saved img: ", path+'.png')

    img_draw= cv2.imread(path+'.png')
    out_save=list(map(lambda x: write_archor(x,img_draw), output))
    cv2.imwrite(path+'_det'+'.png', img_draw)
    print("Saved img: ", path+'_det'+'.png')

def save_img_i(img,i,path):
    img_save = inp_to_image(img)
    cv2.imwrite(path+str(i)+'.png', img_save)
    print("Saved img: ", path+str(i)+'.png')
   
def save_img(img, path):
    img_save = inp_to_image(img)
    cv2.imwrite(path +'.png', img_save)
    print("Saved img: ", path +'.png')

def get_character_input(input_dim, character_path):
    img = cv2.imread(character_path)
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_out.to(device)

def get_ind2(prediction,ori_index):
    conf_mask = ((prediction[:,:,4] > confidence)).float().unsqueeze(2)#confidence>0.5
    max_a,max_b=torch.max(prediction[:,:,5:5+ num_classes],2)
    conf_mask2 = (max_b ==ori_index).float().unsqueeze(2)#1=bicycle,11=stop_sign,9=traffic_light
    prediction = prediction*conf_mask2
    prediction=prediction*conf_mask
    ind_nz = torch.nonzero(prediction[0,:,4])
    if ind_nz.shape[0]==0:
        return  0
    else:
        return 1

def yolo_test(num_test, patch_test):
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
    img_ori=get_test_input_ori(inp_dim, CUDA)

    #ori_stop
    original_stop,map_4_patches,map_4_stop,patch_four=get_stop_patch(input_dim=201)
    
    #pole
    patch_pole=torch.zeros(3,27,201).to(device)
    ori_pole=get_pole('imgs/pole/pole.jpg')

    #patch_fix
    patch_fix=torch.zeros(3,201,201).to(device)
#     patch_fix[:,:,:]=stop_adv[:,:,:]
    
    #map
    map_ori=torch.zeros(1,3,416,416).to(device)
    map_resize=torch.zeros(1,3,416,416).to(device)# the map of all three patches
    
    #img_input
    img_input=torch.zeros(1,3,416,416).to(device)
    img_input2=torch.zeros(1,3,416,416).to(device)

    #perspective
    ori_stop_perspective=torch.zeros(3,width,width).to(device)
    ori_stop2=torch.zeros(3,201,201).to(device)
    ori_stop2_pers=torch.zeros(3,201,201).to(device)

    #patch
    patch_transform=img_ori[ 0,:, start_x:start_x+width,start_y:start_y+height]
    patch_pole_transform=torch.zeros(3,27,201).to(device)

    suc_step=0
    total_frames = 0
    try: 
        for i in range(num_test):

            # get_random_stop_ori # cut out the patch and paste it to different stop sign
            original_stop = get_random_stop_ori(imlist_stop).to(device)
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
#             print("patch_transform", patch_transform.requires_grad)
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
            img_ori = get_random_img_ori(imlist_back).to(device)
            
#             print("patch_transform", patch_transform.requires_grad)
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

            patch_resize=torch.from_numpy(patch_resize.transpose(2,0,1)).to(device)
            patch_pole_resize=torch.from_numpy(patch_pole_resize.transpose(2,0,1)).to(device)
            ori_stop_resize=torch.from_numpy(ori_stop_resize.transpose(2,0,1)).to(device)

            ori_stop2_resize=torch.from_numpy(ori_stop2_resize.transpose(2,0,1)).to(device)

            map_resize[:,:,:,:]=map_ori[:,:,:,:]
            start_x=int(x_c-(width_r-1)/2)
            end_x=int(x_c+(width_r-1)/2+1)
            start_y=int(y_c-(height_r-1)/2)
            end_y=int(y_c+(height_r-1)/2+1)
            start_pole_y=int(y_c-(height_pole_r-1)/2)
            end_pole_y=int(y_c+(height_pole_r-1)/2+1)
            start_pole_x=int(x_c+(width_r-1)/2+1)
            end_pole_x=int(x_c+(width_r-1)/2+width_pole_r+1)


            img_input[:,:,:,:]=img_ori[:,:,:,:]
            img_input2[:,:,:,:]=img_ori[:,:,:,:]

            # get four corners of stop
            stop_4=torch.sum(ori_stop_resize[:,:,:],0)
            stop_4=(stop_4<0.1).float().unsqueeze(0)
            stop_4=torch.cat((stop_4,stop_4,stop_4),0)
            
            # img_input[0,:,start_x:end_x,start_y:end_y]=torch.clamp((patch_resize+map_character_resize),0,1)+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
            # add adv_stop and pole to background img
            img_input[0,:,start_x:end_x,start_y:end_y]=patch_resize*(1-stop_4)+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
            img_input[0,:,start_pole_x:end_pole_x,start_pole_y:end_pole_y]=patch_pole_resize

            img_input2[0,:,start_x:end_x,start_y:end_y]=ori_stop2_resize+img_input[0,:,start_x:end_x,start_y:end_y]*stop_4
            img_input2[0,:,start_pole_x:end_pole_x,start_pole_y:end_pole_y]=patch_pole_resize

            input1=Variable(img_input, requires_grad=True)
            input2=Variable(img_input2, requires_grad=True)
            
            #forward
            rn_noise=torch.from_numpy(np.random.uniform(-0.1,0.1,size=(1,3,416,416))).float().to(device)
            with torch.no_grad():
                prediction, _ = model(torch.clamp(input1+rn_noise,0,1), CUDA)
                prediction2, _ = model(torch.clamp(input2+rn_noise,0,1), CUDA)
            detect=get_ind2(prediction,ori_index)
            detect2=get_ind2(prediction2,ori_index)
            
            
            is_success = 0
            save_every = save_interval
            if detect2 != 0:
                total_frames += 1
                if detect==0:
#                     print("Success")
                    suc_step += 1
                    is_success = 1
                    # det_and_save_img_i(input1, i, output_dir + "adv_img", is_success) # Adversarial Example: woth background and adversarial stop sign
                    # det_and_save_img_i(input2, i, output_dir + "ori_img", is_success)
#                 else:
#                     print("Not success")
#             else:
#                 print("Not detected in original image.")
            if total_frames > 0: 
                suc_rate = suc_step/total_frames
                
            if i % 100 == 0:  
                print("i:",i)
                print('success_rate:', suc_rate)
#                 output_dir = "./output/yolo_test/"
#                 det_and_save_img_i(input1, i, output_dir + "adv_img") # Adversarial Example: woth background and adversarial stop sign
#                 det_and_save_img_i(input2, i, output_dir + "ori_img")
                # save_img_i(patch_resize, i, output_dir + "debug/patch_resize/")
                # save_img_i(patch_pole_resize, i, output_dir + "debug/patch_pole_resize/")
                # save_img_i(stop_4, i, output_dir + "debug/stop_4/")
                # save_img_i(img_ori, i, output_dir + "debug/img_ori/")
    
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        return suc_rate, suc_step, total_frames

    return suc_rate, suc_step, total_frames


