#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python adam_batch_nodist.py')


# In[ ]:


get_ipython().system('python adam_batch_dist.py')


# In[ ]:


get_ipython().system('python adam_nobatch_nodist.py')


# In[ ]:


get_ipython().system('python adam_nobatch_dist.py')


# In[ ]:


get_ipython().system('python fgsm_batch_nodist.py')


# In[ ]:


get_ipython().system('python fgsm_batch_dist.py')


# In[3]:


import PIL.Image
import numpy as np
from io import StringIO
from IPython.display import clear_output, Image, display, HTML

def read_image(path):
    img = PIL.Image.open(path)
    img = np.array(img, dtype=np.uint8)
    return img
from io import BytesIO

def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


img = read_image('imgs/stop/stop1.jpg')
showarray(img)
img = read_image('imgs/stop_sign_mask.png')
showarray(img)


# In[161]:


import cv2
import torch

def get_stop_patch(input_dim=201):
    images='stop/'
#     img = cv2.imread('imgs/stop_sign_mask_patch.png')
    img = cv2.imread('imgs/stop/stop1.jpg')
#     print(img.shape, img2.shape)
#     cv2.imshow("img",img)
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    stop_ori=img_out.to(device)

    width=input_dim
    height=input_dim
    map_ori=torch.zeros(1,3,width,height).to(device)
    #get character image
    #   map_character_ori=torch.zeros(1,3,width,height).to(device)
    
    #  control the ratio of the patch on stop sign
    #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.58)]=1#rec_70  #ratio: 0.20
    #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.58)]=1
    
    #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.72)]=1#rec_100  #0.29
    #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.72)]=1
    #  map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.444)]=1#rec_90  #0.11
    #  map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.444)]=1
    map_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.674)]=1#rec_90   #0.26
    map_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.674)]=1
#     stop_ori[0,:,int(height*0.05):int(height*0.316),int(width*0.26):int(width*0.674)]=1#rec_90   #0.26
#     stop_ori[0,:,int(height*0.673):int(height*0.94), int(width*0.26):int(width*0.674)]=1
    map_stop=-(map_ori-1)
    x = stop_ori*map_stop
    patch_stop=stop_ori*map_ori
    print(stop_ori.shape, patch_stop.shape)
    return stop_ori[0,:,:,:],map_ori[0,:,:,:],map_stop[0,:,:,:],patch_stop[0,:,:,:], x[0,:,:,:]   #output:original stop, map mask for patch, map mask for stop, four patch
    #original_stop,map_4_patches,map_4_stop,patch_four


# In[162]:


device = torch.device("cuda")
original_stop,map_4_patches,map_4_stop,patch_four, x = get_stop_patch()


# In[171]:


def get_stop_patch_from_img(input_dim=300):
    stop_ori = cv2.imread('imgs/stop_sign_mask.png')
    print(stop_ori.shape)
    map4patch = cv2.imread('imgs/x.png')
    map4stop = cv2.imread('imgs/y.png')
    
    stop_ori = cv2.resize(stop_ori, (input_dim, input_dim))
    map4patch = cv2.resize(map4patch, (input_dim, input_dim))
    map4stop = cv2.resize(map4stop, (input_dim, input_dim))

    stop_ori = image_to_inp(stop_ori, device)
    map4patch = image_to_inp(map4patch, device)
    map4stop = image_to_inp(map4stop, device)
    
    patch_four=stop_ori*map4patch
    return stop_ori[0,:,:,:],map4patch[0,:,:,:],map4stop[0,:,:,:],patch_four[0,:,:,:] 


# In[172]:


device = torch.device("cuda")
original_stop,map_4_patches,map_4_stop,patch_four = get_stop_patch_from_img()


# In[173]:


img = read_image('imgs/stop/stop1.jpg')
showarray(img)
img = read_image('imgs/stop_sign_mask.png')
showarray(img)

save_img(original_stop, "imgs/original_stop")
img = read_image("imgs/original_stop.png")
showarray(img)

save_img(map_4_patches, "imgs/map_4_patches")
img = read_image("imgs/map_4_patches.png")
showarray(img)

save_img(map_4_stop, "imgs/map_4_stop")
img = read_image("imgs/map_4_stop.png")
showarray(img)

save_img(patch_four, "imgs/patch_four")
img = read_image("imgs/patch_four.png")
showarray(img)


# In[136]:


def save_img(img,path):
    img_save = inp_to_image(img)
    cv2.imwrite(path+'.png', img_save)
    print("Saved img: ", path+'.png')
    
def inp_to_image(inp):
    inp = inp.cpu().squeeze()
    inp = inp*255
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1,2,0)
    inp = inp[:,:,::-1]
    return inp

def image_to_inp(img, device):
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_out = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    img_out=img_out.to(device)
    return img_out

img = cv2.imread('imgs/x.png')
inp = image_to_inp(img, device)
save_img(inp, "imgs/x")
img = read_image("imgs/x.png")
showarray(img)


# Triangles

# In[175]:


import cv2 as cv
import numpy as np

WHITE = (255, 255, 255)
w = 201
h = 201

img = np.zeros((w, h, 3), np.uint8)

pts = [(15, 65), (65, 15), (65, 65)]
cv.fillPoly(img, np.array([pts]), WHITE)

pts = [(w-15, 65), (w-65, 15), (w-65, 65)]
cv.fillPoly(img, np.array([pts]), WHITE)

pts = [(15, h-65), (65, h-15), (65, h-65)]
cv.fillPoly(img, np.array([pts]), WHITE)

pts = [(w-15, h-65), (w-65, h-15), (w-65, h-65)]
cv.fillPoly(img, np.array([pts]), WHITE)

cv2.imwrite("imgs/x.png", img)
y = 255-img

cv2.imwrite("imgs/y.png", y)
img = read_image("imgs/x.png")
showarray(img)
img = read_image("imgs/y.png")
showarray(img)


# In[158]:


import cv2 as cv
import numpy as np

WHITE = (255, 255, 255)
w = 201
h = 201
img = np.zeros((w, h, 3), np.uint8)

x_c = w/2
y_c = h/2

x_r = 0.33
y_r = 0.4

x1 = x_c - w*(x_r/2)
x2 = x_c + w*(x_r/2)

y1 = h*0.05
y4 = h*0.94
y2 = y1 + h*(y_r/2)
y3 = y4 - h*(y_r/2)


p0 = int(x1), int(y1)
p1 = int(x2),int(y2)
p2 = int(x1), int(y3)
p3 = int(x2),int(y4)

cv.rectangle(img, p0, p1, WHITE, cv.FILLED)
cv.rectangle(img, p2, p3, WHITE, cv.FILLED)

cv2.imwrite("imgs/x.png", img)
y = 255-img
cv2.imwrite("imgs/y.png", y)
img = read_image("imgs/x.png")
showarray(img)
img = read_image("imgs/y.png")
showarray(img)


# In[ ]:


stop_4=torch.sum(ori_stop_resize[:,:,:],0)
stop_4=(stop_4<0.1).float().unsqueeze(0)
stop_4=torch.cat((stop_4,stop_4,stop_4),0)

