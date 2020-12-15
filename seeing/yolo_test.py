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
from helper.util_creation import get_loss_disappear,get_loss_creation_select, get_ind,get_loss_creation,get_loss_smoothness,get_map_bounding,get_random_img_ori,get_loss_median,get_random_stop_ori
from helper.patch import add_patch
from helper.patch import addweight_transform_multiple, perspective_transform_multiple,inverse_perspective_transform, translation_center_multiple,gamma_transform_multiple, amplify_size,shear_transform,rotate_transform,translation_transform,gamma_transform,blur_transform,transform,transform_multiple,rescale_transform_multiple
from helper.preprocess import prep_image, inp_to_image
from helper.darknet import Darknet

from generate_video import generate_video_concat


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/det/stop_sign.jpg")
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

def get_ind2(prediction,ori_index):
    prediction=prediction.cpu()
    conf_mask = ((prediction[:,:,4] > confidence)).float().unsqueeze(2)#confidence>0.5
   # print("conf_mask")
    max_a,max_b=torch.max(prediction[:,:,5:5+ num_classes],2)
#    print(max_b)
    conf_mask2 = (max_b ==ori_index).float().unsqueeze(2)#1=bicycle,11=stop_sign,9=traffic_light
    prediction = prediction*conf_mask2
    prediction=prediction*conf_mask
    ind_nz = torch.nonzero(prediction[0,:,4])
#    print('ind_nz.shape:',ind_nz.shape)
    if ind_nz.shape[0]==0:
        return  0
    else:
        ind_out=np.zeros(shape=[3, ind_nz.shape[0]])
        ind_out[0,:]=ind_nz[:,0] #index of (confidence>0.5&&label==target_label)
        ind_out[1,:]=max_b[:,ind_nz[:,0]] # label of index
        ind_out[2,:]=prediction.data[0,ind_nz[:,0],4] #confidence of index
        ind_out=ind_out[:,(-ind_out[2,:]).argsort()] # sort index based on confidence
        ind_nz=ind_out[0,:].astype(np.int32)
        return 1



def get_test_input_ori(input_dim, CUDA):
    img = cv2.imread("imgs/det/stop_sign.jpg")
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
    cls = min(cls, 79)  #TODO: DEBUG
    label = "{0}".format(classes[cls])
    print(str(cls), label)
    colors = pkl.load(open("helper/pallete", "rb"))
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img



def save_img(img,i):
    img_save = inp_to_image(img)
    cv2.imwrite('det_record3/yue_record_'+str(i)+'.png', img_save)


def save_img2(img,i):
    img_save = inp_to_image(img)
    cv2.imwrite('yolo_test/yue_record_'+str(i)+'.png', img_save)



def get_character_input(input_dim, character_path):
    img = cv2.imread(character_path)
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_out.cuda()

def get_pole(character_path):
    img = cv2.imread(character_path)
    img = cv2.resize(img, (201, 27))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_out.cuda()


def get_stop_adv(character_path):
    img = cv2.imread(character_path)
    img = cv2.resize(img, (201, 201))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_out.cuda()


def get_stop_patch(input_dim=201):
    img = cv2.imread("imgs/stop/stop4.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    stop_ori=img_out.cuda()
    return stop_ori[0,:,:,:]

def get_stop_patch_random(input_dim=201):

    stop0 = adv_stop_test  # cut out patch from this stop
    item=random.randint(0,len(imlist_stop)-1)
    stop1 = imlist_stop[item]  # paste the patch to this random stop

    img = cv2.imread(stop0)
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    stop_ori=img_out.cuda()

    width=input_dim
    height=input_dim
    map_ori=torch.zeros(1,3,width,height).cuda()
    #get character image
    #   map_character_ori=torch.zeros(1,3,width,height).cuda()
    
    #  control the ratio of the patch on stop sign
    #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.58)]=1#rec_70  #ratio: 0.20
    #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.58)]=1
    
    map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.72)]=1#rec_100  #0.29
    map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.72)]=1
    #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.444)]=1#rec_90  #0.11
    #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.444)]=1
    # map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.674)]=1#rec_90   #0.26
    # map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.674)]=1
    map_stop=-(map_ori-1)

    patch_stop=stop_ori*map_ori

    img = cv2.imread(stop1)
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    stop_ori=img_out.cuda()

    return stop_ori[0,:,:,:],map_ori[0,:,:,:],map_stop[0,:,:,:],patch_stop[0,:,:,:]   #output:original stop, map mask for patch, map mask for stop, four patch


def get_adv_episilon(img_ori,adv_label,first_index,ori_index):
    start_x = 200
    start_y = 100
    width = 201
    height = 201
    radius=(width-1)/2

    x_c=100
    y_c=150
    
    width_pole=201
    height_pole=27

    #adv
    stop_adv=get_stop_adv(adv_stop_test)
    
    #img_ori
    img_ori=get_test_input_ori(inp_dim, CUDA)

    #ori_stop
    original_stop=get_stop_patch(input_dim=201)
    
    #pole
    patch_pole=torch.zeros(3,27,201).cuda()
    ori_pole=get_pole('imgs/pole/pole.jpg')

    #patch_fix
    patch_fix=torch.zeros(3,201,201).cuda()
    patch_fix[:,:,:]=stop_adv[:,:,:]
    
    #map
    map_ori=torch.zeros(1,3,416,416).cuda()
    map_resize=torch.zeros(1,3,416,416).cuda()# the map of all three patches
    
    #img_input
    img_input=torch.zeros(1,3,416,416).cuda()
    img_input2=torch.zeros(1,3,416,416).cuda()

    #perspective
    ori_stop_perspective=torch.zeros(3,width,width).cuda()
    ori_stop2=torch.zeros(3,201,201).cuda()
    ori_stop2_pers=torch.zeros(3,201,201).cuda()

    #patch
    patch_transform=img_ori[ 0,:, start_x:start_x+width,start_y:start_y+height]
    patch_pole_transform=torch.zeros(3,27,201).cuda()

    num_steps=nsteps
    suc_step=0
    loss_dis=1
    ratio=1
    ratio_turn=1
    back_turn=True
    loss=1
    snsteps=0
    perspective=0

    #patch_fix -> patch_transform -> patch_resize
    #ori_pole -> patch_pole_transform -> patch_pole_resize
    #original_stop -> ori_stop_perspective -> ori_stop_resize -> stop_4

    #img_input -> input1

    #loss.backward -> input_grad -> grad_resize -> grad_4_patches -> epsilon_4_patches -> patch_four -> patch_fix

    try: 

        for i in range(num_steps):

            # get_random_stop_ori # cut out the patch and paste it to different stop sign
            original_stop,map_4_patches,map_4_stop,patch_four=get_stop_patch_random(input_dim=201)
            patch_fix = original_stop*map_4_stop+patch_four
            patch_fix=torch.clamp(patch_fix,0,1)

            if i%2==0:
                #patch_transform, map_bounding_transform
                #patch_fix can only be changed by translation and adv_episilon
                #first step's out=patch_transform
                patch_transform[:,:,:]=addweight_transform_multiple(patch_fix[:,:,:])#gamma_transform_multiple(patch_fix[:,:,:])
                patch_pole_transform[:,:,:]=ori_pole[:,:,:]

            else:
                # gamma_transform_multiple(patch_fix)
                patch_transform[:,:,:] = patch_fix[:,:,:]
                patch_pole_transform[:,:,:]=ori_pole[:,:,:]
                ori_stop2[:,:,:]=original_stop[:,:,:]
            
            angle = test_angle
            if is_given_angle:
                if angle == 0:
                    ori_stop_perspective[:,:,:]=original_stop[:,:,:]
                    ori_stop2_pers[:,:,:]=ori_stop2[:,:,:]
                else:
                    # if a test angle is given, randomly set the angle to either positive or negative of the given angle
                    k = random.randint(0,3)
                    if k % 2 == 0: 
                        angle = -test_angle
                    patch_transform,org,dst,angle=perspective_transform_multiple(patch_transform, set_angle=angle)
                    ori_stop_perspective,org_abandon,dst_abandon,angle_abandon=perspective_transform_multiple(original_stop,True,angle)
                    ori_stop2_pers,org_abandon,dst_abandon,angle_abandon=perspective_transform_multiple(ori_stop2,True,angle)
            elif i % 3 == 0:
                #perspective transform
                perspective=1      
                patch_transform,org,dst,angle=perspective_transform_multiple(patch_transform, set_angle=angle)
                ori_stop_perspective,org_abandon,dst_abandon,angle_abandon=perspective_transform_multiple(original_stop,True,angle)
                ori_stop2_pers,org_abandon,dst_abandon,angle_abandon=perspective_transform_multiple(ori_stop2,True,angle)
            else:
                ori_stop_perspective[:,:,:]=original_stop[:,:,:]
                ori_stop2_pers[:,:,:]=ori_stop2[:,:,:]
            #random background
            img_ori = get_random_img_ori(imlist_back).cuda()
            
            #third_step's output=img_input, input=patch_transform
            snsteps=snsteps+1
            if ((i+1)/2==0):
                    ratio = random.uniform(0.2, 1)
            else:
                    ratio=random.uniform(0.2,0.5)
            
            if distance == 1:
                    print("1")
                    ratio = random.uniform(0.8, 0.98)
                    print(ratio)
            elif distance == 2:
                    print("2")
                    ratio = random.uniform(0.6, 0.8)
                    print(ratio)
            elif distance == 3:
                    print("3")
                    ratio = random.uniform(0.4, 0.6)
                    print(ratio)
            elif distance == 4:
                    print("4")
                    ratio = random.uniform(0.18, 0.4)
                    print(ratio)

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

            patch_resize=torch.from_numpy(patch_resize.transpose(2,0,1)).cuda()
            patch_pole_resize=torch.from_numpy(patch_pole_resize.transpose(2,0,1)).cuda()
            ori_stop_resize=torch.from_numpy(ori_stop_resize.transpose(2,0,1)).cuda()

            ori_stop2_resize=torch.from_numpy(ori_stop2_resize.transpose(2,0,1)).cuda()

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
            rn_noise=torch.from_numpy(np.random.uniform(-0.1,0.1,size=(1,3,416,416))).float().cuda()
            prediction = model(torch.clamp(input1+rn_noise,0,1), CUDA)
            prediction2 = model(torch.clamp(input2+rn_noise,0,1), CUDA)
            detect=get_ind2(prediction,ori_index)
            detect2=get_ind2(prediction2,ori_index)

            print("i:",i)
            
            is_success = 0
            save_every = save_interval
            if(detect2 != 0 and detect==0):
                print("Success")
                suc_step=suc_step+1
                is_success = 1
                # det_and_save_img_i(input1, i, output_dir + "adv_img", is_success) # Adversarial Example: woth background and adversarial stop sign
                # det_and_save_img_i(input2, i, output_dir + "ori_img", is_success)
            else:
                print("Not success")
            print('success_rate:',(suc_step)/(i+1))
            if i % save_every == 0:                
                det_and_save_img_i(input1, i, output_dir + "adv_img", is_success) # Adversarial Example: woth background and adversarial stop sign
                det_and_save_img_i(input2, i, output_dir + "ori_img", is_success)
                # save_img_i(patch_resize, i, output_dir + "debug/patch_resize/")
                # save_img_i(patch_pole_resize, i, output_dir + "debug/patch_pole_resize/")
                # save_img_i(stop_4, i, output_dir + "debug/stop_4/")
                # save_img_i(img_ori, i, output_dir + "debug/img_ori/")
    
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        return patch_fix, input1

    return patch_fix, input1

def det_and_save_img_i(img,i,path, is_success):
    prediction = model(img , CUDA)
    output = write_results(prediction.data, confidence, num_classes, nms = True, nms_conf = nms_thesh)#final_output,[:,8] 8=input_img_num,xc,yc,width,height,confidence,confidence_class,class

    img_save=inp_to_image(img)

    cv2.imwrite(path+'/'+str(i)+'.png', img_save)
    print("Saved img: ", path+'/'+str(i)+'.png')
    img_draw= cv2.imread(path+'/'+str(i)+'.png')
    out_save=list(map(lambda x: write_archor(x,img_draw), output))

    if is_success: 
        cv2.imwrite(path+'_det/success/'+str(i)+'.png', img_draw)
        print("Saved img: ", path+'_det/success/'+str(i)+'.png')
    else:
        cv2.imwrite(path+'_det/not_success/'+str(i)+'.png', img_draw)
        print("Saved img: ", path+'_det/not_success/'+str(i)+'.png')


def save_img_i(img,i,path):
    img_save = inp_to_image(img)
    cv2.imwrite(path+str(i)+'.png', img_save)
    print("Saved img: ", path+str(i)+'.png')

def det_and_save_img(img, path):
    prediction = model(img , CUDA)
    output = write_results(prediction.data, confidence, num_classes, nms = True, nms_conf = nms_thesh)#final_output,[:,8] 8=input_img_num,xc,yc,width,height,confidence,confidence_class,class

    img_save=inp_to_image(img)

    cv2.imwrite(path+'.png', img_save)
    print("Saved img: ", path+'.png')

    img_draw= cv2.imread(path+'.png')
    out_save=list(map(lambda x: write_archor(x,img_draw), output))
    cv2.imwrite(path+'_det'+'.png', img_draw)
    print("Saved img: ", path+'_det'+'.png')
    
def save_img(img, path):
    img_save = inp_to_image(img)
    cv2.imwrite(path +'.png', img_save)
    print("Saved img: ", path +'.png')

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='Test success rate of patch adversarial attack on YOLOv3')

    # main parameters in testing 
    parser.add_argument("--session_name", dest = "session_name", help = "session_name", default = time.time() , type = str)
    parser.add_argument("--nsteps", dest = "nsteps", help = "number of tests", default = 10000)
    parser.add_argument("--save_interval", dest = "save_interval", help = "save_interval", default = 2)
    
    parser.add_argument("--adv_stop", dest = 'adv_stop', help =
                        "Image of adversarial stop sign to be tested",
                        default = "imgs/adv_stop/stop1.png", type = str)
    parser.add_argument("--angle", dest = "angle", help = "test angle of stop sign", default = 1000) # default value is 1000, angle is randomly chosen from (-60, 60)
    parser.add_argument("--distance", dest = "distance", help = "distance of stop sign (value from near(large) to far(small): 1,2,3,4), default to random (value: 0)", default = 0)
    parser.add_argument("--video", dest = "video", help = "generate video after all iterations (True/False)", default = True, type = bool)
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
                        default = "output/yolo_test/", type = str)

    #detection parameters
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)

    #model parameters
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "yolov3/cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
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
    
    adv_stop_test = args.adv_stop
    test_angle = int(args.angle)
    distance = int(args.distance)

    img_stop = args.stop_dir
    img_bg = args.bg_dir
    output_dir = args.output_dir
    
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    is_given_angle = True
    if test_angle == 1000:  #random angle
        is_given_angle = False

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
    model.eval() #Set the model in evaluation mode

    num_classes = 80
    classes = load_classes(args.classes)

    img_ori = get_test_input(inp_dim, CUDA)
    prediction_dog = model(img_ori, CUDA)  # feature_archor_output
    ori_index = 11
    ind_nz_out = get_ind(prediction_dog, ori_index)  # 1=bicycle,11=stop_sign,9=traffic_light
    first_index = ind_nz_out[0]

    adv_label = [1 if i == 11 else 0 for i in range(80)] #stop sign=11,traffic_light=9
    adv_label = np.array(adv_label)
    adv_label = np.reshape(adv_label,(1,80))
    adv_label = torch.from_numpy(adv_label).float()

    patch_adv, output_adv = get_adv_episilon(img_ori, adv_label, first_index, ori_index)

    save_img(patch_adv, output_dir + 'final/adv_stop')
    det_and_save_img(output_adv, output_dir + 'final/adv_img')

    if args.video:
        fps = float(args.fps)

        ori_img_det = output_dir + "ori_img_det/success/"
        adv_img_det = output_dir + "adv_img_det/success/"
        path_avi = output_dir + "avi/" + session_name + "_success.avi"
        generate_video_concat(ori_img_det, adv_img_det, path_avi, fps)

        ori_img_det = output_dir + "ori_img_det/not_success/"
        adv_img_det = output_dir + "adv_img_det/not_success/"
        path_avi = output_dir + "avi/" + session_name + "_notsuccess.avi"
        generate_video_concat(ori_img_det, adv_img_det, path_avi, fps)

    print("Done and exit")
