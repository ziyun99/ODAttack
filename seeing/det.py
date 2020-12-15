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
from helper.util_creation import get_loss_disappear,get_loss_creation_select, get_ind,get_loss_creation,get_loss_smoothness,get_map_bounding,get_random_img_ori,get_loss_median,get_loss_saturation,get_random_stop_ori
from helper.patch import add_patch
from helper.patch2 import  addweight_transform_multiple, perspective_transform_multiple,inverse_perspective_transform, translation_center_multiple,gamma_transform_multiple, amplify_size,shear_transform,rotate_transform,translation_transform,gamma_transform,blur_transform,transform,transform_multiple,rescale_transform_multiple
from helper.preprocess import prep_image, inp_to_image, letterbox_image
from helper.darknet4 import Darknet


def get_test_input(img, input_dim, CUDA):
    img = cv2.imread(img)
    # img = cv2.imread("det/adv_stop_sign.png")
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = cv2.resize(img, (input_dim, input_dim))

    img_ =  img.transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_
        

def det_and_save_img(img, path):
    prediction,feature_out = model(img , CUDA)
    output = write_results(prediction.data, confidence, num_classes, nms = True, nms_conf = nms_thesh)#final_output,[:,8] 8=input_img_num,xc,yc,width,height,confidence,confidence_class,class

    img_save=inp_to_image(img)
    img_save = img_save[:,:,::-1]
    # img_save =  img_save.transpose((1,2,0))
    # img_save = cv2.cvtColor(img_save,cv2.COLOR_BGR2RGB)
    
    cv2.imwrite(path+'.png', img_save)
    print("Saved img: ", path+'.png')

    img_draw= cv2.imread(path+'.png')
    out_save=list(map(lambda x: write_archor(x,img_draw), output))
    cv2.imwrite(path+'_det'+'.png', img_draw)
    print("Saved img: ", path+'_det'+'.png')

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

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest = 'images', help =
                        "Image / Directory containing images to perform detection upon",
                        default = "/content/test_imgs/", type = str)
    parser.add_argument("--det", dest = 'det_dir', help =
                        "Image / Directory to store detections to",
                        default = "output/det/", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "yolov3/cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3/weights/yolov3.weights", type = str)
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
    classes = load_classes('yolov3/data/coco.names')

    # load val_data list
    # images="./val2014_2"
    # images_stop= "./back_stop"
    # images_back="./back"
    try:
       imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
       imlist = []
       imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
       print ("No file or directory with the name {}".format(images))
       exit()
    print(imlist)

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

    for img_path in imlist:
      img_test =get_test_input(img_path, inp_dim, CUDA)
      det_and_save_img(img_test, args.det_dir + os.path.basename(img_path)[:-4])



