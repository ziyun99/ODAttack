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
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

from helper.util import *
from helper.util_creation import get_loss_disappear,get_loss_creation_select, get_ind,get_loss_creation,get_loss_smoothness,get_map_bounding,get_random_img_ori,get_loss_median,get_loss_saturation,get_random_stop_ori
from helper.patch import add_patch
from helper.patch2 import  addweight_transform_multiple, perspective_transform_multiple,inverse_perspective_transform, translation_center_multiple,gamma_transform_multiple, amplify_size,shear_transform,rotate_transform,translation_transform,gamma_transform,blur_transform,transform,transform_multiple,rescale_transform_multiple
from helper.preprocess import prep_image, inp_to_image
from helper.darknet4 import Darknet

from generate_video import generate_video_concat

def get_img_input(img, CUDA, input_dim=201):
    img = cv2.imread(img)
    # img = cv2.imread("det/stop_sign.png")
    img = cv2.resize(img, (input_dim, input_dim))

    img_ =  img.transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    
    return img_

    
def det_and_save_img_i(img,i,path):
    prediction,feature_out = model(img , CUDA)
    output = write_results(prediction.data, confidence, num_classes, nms = True, nms_conf = nms_thesh)#final_output,[:,8] 8=input_img_num,xc,yc,width,height,confidence,confidence_class,class

    img_save=inp_to_image(img)

    cv2.imwrite(path+'/'+str(i)+'.png', img_save)
    print("Saved img: ", path+'/'+str(i)+'.png')

    img_draw= cv2.imread(path+'/'+str(i)+'.png')
    out_save=list(map(lambda x: write_archor(x,img_draw), output))
    cv2.imwrite(path+'_det/'+str(i)+'.png', img_draw)
    print("Saved img: ", path+'_det/'+str(i)+'.png')

def det_and_save_img(img, path):
    prediction,feature_out = model(img , CUDA)
    output = write_results(prediction.data, confidence, num_classes, nms = True, nms_conf = nms_thesh)#final_output,[:,8] 8=input_img_num,xc,yc,width,height,confidence,confidence_class,class

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
    # img =  img.transpose((1,2,0)).copy()
    img_save = inp_to_image(img)
    cv2.imwrite(path +'.png', img_save)
    print("Saved img: ", path +'.png')

def get_pole(character_path):
    img = cv2.imread(character_path)
    img = cv2.resize(img, (201, 27))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_out.cuda()

def get_stop_patch(stop_img, input_dim=201):
    img = cv2.imread(stop_img)
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    stop_ori=img_out.cuda()

    width=input_dim
    height=input_dim
    map_ori=torch.zeros(1,3,width,height).cuda()
    #get character image
    #   map_character_ori=torch.zeros(1,3,width,height).cuda()
    map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.58)]=1#rec_70
    map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.58)]=1
    #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.72)]=1#rec_100
    #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.72)]=1
    #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.444)]=1#rec_90
    #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.444)]=1
    #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.674)]=1#rec_90
    #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.674)]=1
    map_stop=-(map_ori-1)

    patch_stop=stop_ori*map_ori
    return map_ori[0,:,:,:],map_stop[0,:,:,:],patch_stop[0,:,:,:]   #output:original stop, map mask for patch, map mask for stop, four patch
#original_stop,map_4_patches,map_4_stop,patch_four



def get_patch():
    stop_img = "output/batch/adv_stop/450.png"
    #ori_stop
    map_4_patches,map_4_stop,patch_four=get_stop_patch(stop_img, input_dim=201)
    
    stop = get_random_stop_ori(imlist_stop).cuda()
    # stop = "imgs/stop/stop1.jpg"
    # stop2 = get_img_input(img=stop, input_dim=201, CUDA = torch.cuda.is_available())
    # stop2=torch.from_numpy(stop2.transpose(2,0,1)).cuda()
    # stop2 = stop2[0, :, :, :]
    patch_fix = stop*map_4_stop+patch_four
    patch_fix=torch.clamp(patch_fix,0,1)
    save_img(patch_fix, output_dir + "patch_stop/450")
    

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='Batch variation AE generation for YOLOv3')

    # main parameters in AE generation
    parser.add_argument("--session_name", dest = "session_name", help = "session_name", default = time.time() , type = str)
    parser.add_argument("--nsteps", dest = "nsteps", help = "Number of iterations", default = 601)
    
    parser.add_argument("--save_interval", dest = "save_interval", help = "save_interval", default = 10)
    parser.add_argument("--video", dest = "video", help = "generate video after all iterations (True: 1/False: 0)", default = 1, type = int)
    parser.add_argument("--fps", dest = "fps", help = "fps for video generation", default = 1.2)
    
    # directory
    parser.add_argument("--stop_dir", dest = 'stop_dir', help =
                        "Image / Directory containing stop signs to generate AE",
                        default = "imgs/stop/", type = str)
    parser.add_argument("--bg_dir", dest = 'bg_dir', help =
                        "Image / Directory containing backgrounds to generate AE",
                        default = "imgs/bg/road/", type = str)
    parser.add_argument("--output_dir", dest = 'output_dir', help =
                        "Image / Directory to store AE generated",
                        default = "output/batch/", type = str)
    
    #detection parameters
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    #model parameters
    parser.add_argument("--cfg", dest = 'cfg', help =
                        "Config file",
                        default = "yolov3/cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weights', help =
                        "weightsfile",
                        default = "yolov3/weights/yolov3.weights", type = str)
    parser.add_argument("--classes", dest = 'classes', help =
                        "classes",
                        default = "yolov3/data/coco.names", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()


if __name__ ==  '__main__':

    print("Start program")

    #parse arguments
    args = arg_parse()
    print(args)

    session_name = str(args.session_name)
    nsteps = int(args.nsteps)
    save_interval = int(args.save_interval)
    
    img_stop = args.stop_dir
    img_bg = args.bg_dir
    output_dir = args.output_dir

    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)


    # set gpu device
    CUDA = torch.cuda.is_available()
    torch.cuda.set_device(0)



    # load images
    print("Load images for AE generation")
    images_stop = img_stop
    try:
       imlist_stop = [osp.join(osp.realpath('.'), images_stop, img) for img in os.listdir(images_stop) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] == '.jpg']
    except NotADirectoryError:
       imlist_stop = []
       imlist_stop.append(osp.join(osp.realpath('.'), images_stop))
    except FileNotFoundError:
       print ("No file or directory with the name {}".format(images_stop))
       exit()
    images_back = img_bg
    try:
       imlist_back = [osp.join(osp.realpath('.'), images_back, img) for img in os.listdir(images_back) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg' or os.path.splitext(img)[1] =='.BMP']
    except NotADirectoryError:
       imlist_back = []
       imlist_back.append(osp.join(osp.realpath('.'), imlist_back))
    except FileNotFoundError:
       print ("No file or directory with the name {}".format(images_back))
       exit()
    print(images_stop, ": ", len(imlist_stop), "stop imgs")
    print(images_back, ": ", len(imlist_back), "background imgs")



    get_patch()

    print("Done and exit")


