from __future__ import division

import os
import os.path as osp

import time
import argparse
import random
import pickle as pkl
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

from helper.util import *
from helper.preprocess import prep_image, inp_to_image, letterbox_image
from helper.darknet import Darknet


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/stop/stop1.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    cls = min(79, cls)
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [225,255,255], 1);
    return img

def write_archor(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[7])
    cls = min(79, cls)
    label = "{0}".format(classes[cls])
    colors = pkl.load(open("helper/pallete", "rb"))
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def save_img(img, path):
    img_save = inp_to_image(img)
    cv2.imwrite(path +'.png', img_save)
    print("Saved img: ", path +'.png')

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


def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "drop.avi", type = str)
    parser.add_argument("--output", dest = 'output_dir', help = 
                        "Directory to save output to",
                        default = "./output/video_det/", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
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
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    print(args)

    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    bbox_attrs = 5 + num_classes
    
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
        
    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()
    
    images = args.video
    try:
       imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.mov' or os.path.splitext(img)[1] =='.avi' or os.path.splitext(img)[1] =='.mp4' or os.path.splitext(img)[1] =='.MP4' or os.path.splitext(img)[1] =='.MOV' ]
    except NotADirectoryError:
       imlist = []
       imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
       print ("No file or directory with the name {}".format(images))
       exit()
    imlist.sort()
    print(imlist)
    print(len(imlist))
    

    for video_path in imlist:
      video_name = os.path.basename(video_path)[:-4]
      cap = cv2.VideoCapture(video_path)
      
      assert cap.isOpened(), 'Cannot capture source'
      print("Starting on video: ", video_path)
      frames = 0
      start = time.time()

      frame_array =[]
      suc_step = 0
      while cap.isOpened():
          
          ret, frame = cap.read()
          if ret:
              # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
              img, orig_im, dim = prep_image(frame, inp_dim)
              # print(img.shape)
              # print(orig_im.shape)
              height, width, layers = orig_im.shape
              size = (width,height)
              
              im_dim = torch.FloatTensor(dim).repeat(1,2)                        
              
              if CUDA:
                  im_dim = im_dim.cuda()
                  img = img.cuda()
              
              with torch.no_grad():   
                  prediction = model(Variable(img), CUDA)
              output = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)

              # if type(output) == int:
              #     frames += 1
              #     print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
              #     cv2.imshow("frame", orig_im)
              #     key = cv2.waitKey(1)
              #     if key & 0xFF == ord('q'):
              #         break
              #     continue
              
              classes = load_classes('yolov3/data/coco.names')
              colors = pkl.load(open("helper/pallete", "rb"))

              # list(map(lambda x: write_archor(x, img), output))

              
              im_dim = im_dim.repeat(output.size(0), 1)
              scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
              
              output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
              output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
              
              output[:,1:5] /= scaling_factor
      
              for i in range(output.shape[0]):
                  output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                  output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
              
              list(map(lambda x: write(x, orig_im), output))
              # print(frames)
              
              if frames %100 == 0:
                cv2.imwrite(args.output_dir + str(frames) +'.png', orig_im)
                print("Saved img: ", args.output_dir + str(frames) +'.png')
              #     save_img(img, args.output_dir + str(frames))
              
              detect=get_ind2(prediction,ori_index=11) #stop sign=11
              
              if(detect==0):
                  print("Success. Not detected")
                  suc_step+=1
              else: 
                print("Not Success. Stop Sign Detected!")
              success_rate = (suc_step)/(frames+1)
              print('HA/AA success_rate:', success_rate, 1-success_rate)

              text = "Detection rate: " + str(frames + 1 - suc_step) + "/" + str(frames + 1) + " = " + str("{:.4f}".format(1-success_rate))
              orig_im = cv2.putText(orig_im, text, (50,50), cv2.FONT_HERSHEY_PLAIN, 4, [0,0,255], 3)

              x = video_name.split('_')
              angle = x[1]
              dist = x[2]
              d = {
                          '0' : "0-5m",
              '1' : "5-10m",
              '2' : "10-15m",
              '3' : "15-20m",
              '4' : "20-25m",
              'full': "5-25m"
              }
              distance =  d.get(dist, 0)
              text = video_name + ", angle: " + str(angle) + ", distance: " + str(distance)
              orig_im = cv2.putText(orig_im, text, (50,120), cv2.FONT_HERSHEY_PLAIN, 4, [0,0,255], 3)
              

              frame_array.append(orig_im)
              # out.write(orig_im)
              # cv2.imshow("frame", orig_im)
              # key = cv2.waitKey(1)
              # if key & 0xFF == ord('q'):
              #     break
              frames += 1
              # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

              
          else:
              break
          print(frames)
          
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      path_avi = args.output_dir  + video_name + ".avi" 
      out = cv2.VideoWriter(path_avi,fourcc, 50.0, size)

      print("Saving to video...", path_avi)
      for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])

      print("video saved to ", path_avi)
      out.release()


      total_frame = frames
      detected_frame = suc_step
      HA_success_rate = success_rate
      AA_success_rate = 1 - success_rate

      list_txt = [video_name, str(total_frame), str(detected_frame), str(HA_success_rate), str(AA_success_rate)]
      text = " ".join(list_txt)
      print(text)
      text_path = args.output_dir  + "concat/success_rate.txt" 
      f = open(text_path, "a")
      f.write(text + "\n")
      f.close()
      print("Success rate written to: ", text_path)
      
    
    




