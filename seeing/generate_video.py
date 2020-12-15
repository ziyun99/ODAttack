import cv2
import numpy as np
import os
from os.path import isfile, join

# pathIn1= 'batch/input1_det/'
# pathIn2 = "batch/input2/"
# pathOut = 'batch/avi/concat.avi'

# pathIn1= 'yolo_test/input1_det/'
# pathIn2 = "yolo_test/input2_det/"
# pathOut = 'yolo_test/avi/concat.avi'

def generate_video_concat(path1, path2, path_avi, fps = 1.2):
    print("generating video from: ", path1, path2)

    files1 = [f for f in os.listdir(path1) if isfile(join(path1, f))]
    files2 = [f for f in os.listdir(path2) if isfile(join(path2, f))]

    files1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    files2.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print(files1)
    print(files2)
    print(len(files1))
    print(len(files2))

    frame_array = []
    num_imgs = min(len(files1), len(files2))

    if num_imgs == 0:
        return

    for i in range(num_imgs):
        filename1 = path1 + files1[i]
        img1 = cv2.imread(filename1)
        height1, width1, layers1 = img1.shape
        size1 = (width1,height1)

        filename2 = path2 + files2[i]
        img2 = cv2.imread(filename2)
        height2, width2, layers2 = img2.shape
        size2 = (width2,height2)
        
        img_concat = np.hstack((img1, img2))
        # img_concat = np.concatenate((img2, img1), axis=1)
        height, width, layers = img_concat.shape
        size = (width,height)
        
        # text = "i = " + str(i*10)
        text = filename2
        img_concat = cv2.putText(img_concat, text, (50,50), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        
        # cv2.imshow("frame", img_concat)
        # key = cv2.waitKey(200)
        #inserting the frames into an image array
        frame_array.append(img_concat)
        
    out = cv2.VideoWriter(path_avi,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
        
        # cv2.imshow("frame", frame_array[i])
        # key = cv2.waitKey(500)
    
    print("video saved to ", path_avi)
    out.release()

def generate_video(path1, path_avi, fps = 1.2):
    print("generating video from: ", path1)

    files1 = [f for f in os.listdir(path1) if isfile(join(path1, f))]

    files1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print(files1)
    print(len(files1))

    frame_array = []
    num_imgs = len(files1)

    if num_imgs == 0:
        return

    for i in range(num_imgs):
        filename1 = path1 + files1[i]
        img1 = cv2.imread(filename1)
        height1, width1, layers1 = img1.shape
        size1 = (width1,height1)
        
        # text = "i = " + str(i*10)
        text = filename1
        img1 = cv2.putText(img1, text, (50,50), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        
        # cv2.imshow("frame", img_concat)
        # key = cv2.waitKey(200)
        #inserting the frames into an image array
        frame_array.append(img1)
        
    out = cv2.VideoWriter(path_avi,cv2.VideoWriter_fourcc(*'DIVX'), fps, size1)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
        
        # cv2.imshow("frame", frame_array[i])
        # key = cv2.waitKey(500)
    
    print("video saved to ", path_avi)
    out.release()



def arg_parse():

    parser = argparse.ArgumentParser(description='Generate video from images')

    parser.add_argument("--path1", dest = "path1", help = "first image directory", default = "./imgs" , type = str)
    parser.add_argument("--path2", dest = "path2", help = "second image directory", default = "./imgs" , type = str)
    parser.add_argument("--path_avi", dest = "path_avi", help = "path to save video to", default = "./avi/test.avi" , type = str)
    parser.add_argument("--concat", dest = "concat", help = "concat images side by side, 0: no, 1: yes", default = 0 , type = int)
    parser.add_argument("--fps", dest = "fps", help = "fps", default = 1.2 , type = float)

if __name__ ==  '__main__':

    print("Start program")

    #parse arguments
    args = arg_parse()
    print(args)

    path1 = str(args.path1)
    path2 = str(args.path2)
    path_avi = str(args.path_avi)
    concat = int(args.concat)
    fps = float(args.fps)
    
    if concat:
        generate_video_concat(path1, path2, path_avi, fps)
    else: 
        generate_video(path1, path_avi, fps)

    print("generate_video done and exit")


