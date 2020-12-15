back:实际场地户外背景图片
back：实际场地室内背景图片
back_stop：不同的stop图片
stop：pole，以及不同shape的mask
val2014_2：随机背景图片

new_test_batch_variation2.py 用于测试哪一隐藏层的interference会起到正向作用（根据不完全测试记录，100,88,80,70这几层可以）
                                             层数很深，随机测得，没有完全覆盖每一层，但据经验，越靠近预测层的效果会好，较浅的layer可能会起反向作用

batch_variation_multilayer_disappear_test4.py 基于batch_variation的AE生成，如已确定interference hidden layer的层数，可以取消line406的注释

forward_disappear_rectangle.py  单次迭代的AE生成

disappear_0.1_0.3_0.1_0.5_love_satu_0.8.py 加入伪造图案的AE生成

yolo_test.py  模拟物理环境，用于初步测试AE性能

video_demo.py 对视频进行目标检测处理


---------------------------------------------------------------------
FILES 分类

#images folder:
back
back_inside
back_stop
batch
det
det_big
forward
imgs
imgs2
stop
val2014_2

#model-relates files:
cfg
data
darknet3.py
darknet4.py
darknet.py
yolov3.weights

#helper functions:
preprocess.py
util_creation.py
util.py
bbox.py
detect2.py
patch2.py
patch.py

#generating AE files:
(error) disappear_0.1_0.3_0.1_0.5_love_satu_0.8.py  #generate diverse styled-AEs with artifical shapes
(ok) batch_vriation_multilayer_loss_test4.py     #AE generation with batch variation method 
(ok) forward_disappear_rectangle.py              #single forward of AE generation
(ok) new_test_batch_variation2.py                #to test which hidden layers in inference will produce positive effect

#testing files:
(ok) video_demo.py   #object detection on video
(ok) yolo_test.py    #simulate pyshical environment, used for primary testing of AE robustness 

---------------------------------------------------------------------

目前遇到的errors

#ERRORS

(error) disappear_0.1_0.3_0.1_0.5_love_satu_0.8.py
 #RuntimeError: _th_clamp_out not supported on CUDAType for Bool
 #line 100, in get_loss_color
    grey_bool=torch.clamp((ra1<0.1)+(ra2<0.1),0,1)
 -> Solution: ???


(ok) yolo_test.py
 #TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
 #line 56, in get_ind2
    ind_out[0,:]=ind_nz[:,0] #index of (confidence>0.5&&label==target_label)
 ->Solution: change line 56-58 to
        ind_out[0,:]=ind_nz[:,0].cpu() #index of (confidence>0.5&&label==target_label)
        ind_out[1,:]=max_b[:,ind_nz[:,0]].cpu() # label of index
        ind_out[2,:]=prediction.data[0,ind_nz[:,0],4].cpu() #confidence of index


(ok) video_demo.py 
 #Error: missing video(.avi) file
 ->Solution: download a sample .avi file from internet
 #FileNotFoundError: pkl.load(open("pallete", "rb"))
 ->Solution: download "pallete" from here: https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-5/
 
 