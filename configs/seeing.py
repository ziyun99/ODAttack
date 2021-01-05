# from torch import optim
import importlib
import os
import time
import torch

class BaseConfig(object):
    """
    Default parameters for all config files.
    """
    def __init__(self):
        """
        Set the defaults.
        """
        self.attack_method = "seeing"
        self.attack_class = "Seeing"
        self.path = os.path.join(os.getcwd(),'seeing/')
        self.out_path = os.path.join(os.getcwd(),'seeing/output/')
        self.logdir = os.path.join(self.path,"logdir/")
        self.name = None
        
        # main parameters in AE generation
        self.session_name = time.strftime("%Y%m%d-%H%M%S")
        self.nepochs = 2001  #Number of iterations
        self.batch_size = 20
        self.save_interval = 10

        # directory
        self.stop_dir = os.path.join(self.path,"imgs/stop/")    #Image / Directory containing stop signs to generate AE
        self.bg_dir = os.path.join(self.path,"imgs/bg/road/")   #Image / Directory containing backgrounds to generate AE
        self.output_dir = os.path.join(self.path,"output/batch/")   #Image / Directory to store AE generated
        
        #detection parameters
        
        self.confidence = 0.5
        self.nms_thresh = 0.4
        self.scales = "1,2,3"
    
        #model parameters
        self.cfg = os.path.join(self.path,"yolov3/cfg/yolov3.cfg") 
        self.weights = os.path.join(self.path,"yolov3/weights/yolov3.weights")
        self.classes = os.path.join(self.path,"yolov3/data/coco.names")
        self.reso = "416"  #Input resolution of the network. Increase to increase accuracy. Decrease to increase speed
    
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.CUDA else "cpu")

        #params
        self.fir_p = 0.3
        self.dist_p = 1.0
        self.tv_p = 0.1
        self.nps_p = 0.5
        self.satur_p = 0.5
        self.fir_flag = True
        self.dist_flag = False
        self.tv_flag = False
        self.nps_flag = False
        self.satur_flag = False
        
        # attack approach
        self.batch_variation = True
        self.optimizers = ["fgsm", "adam"]
        self.optimizer = self.optimizers[0]
        
        self.run_test = True
        self.ntests = 1000
        self.test_interval = 250
        self.bg_test_dir = os.path.join(self.path,"imgs/bg/road_test/") 
        
        self.ori_index = 11
        
        self.path_img_ori = os.path.join(self.path,'imgs/det/stop_sign.jpg')
        self.path_pole = os.path.join(self.path,'imgs/pole/pole.jpg')
        self.path_ori_stop = os.path.join(self.path,'imgs/stop/stop1.jpg')
        self.path_map4patch = os.path.join(self.path,'imgs/mask/map4patch.png')
        self.path_map4stop = os.path.join(self.path,'imgs/mask/map4stop.png')
        self.from_mask = False
        self.patch_ratio = 0.25
        self.patch_dim = 201
        
        self.debug = False
        
class Fgsm_NoBatch(BaseConfig):
    def __init__(self):
        super().__init__()
        self.optimizer = self.optimizers[0]
        self.batch_variation = False
        self.name = "_".join([self.optimizer, "nobatch"])

class Fgsm_Batch(BaseConfig):
    def __init__(self):
        super().__init__()
        self.optimizer = self.optimizers[0]
        self.batch_variation = True
        self.name = "_".join([self.optimizer, "batch"])


class Adam_NoBatch(BaseConfig):
    def __init__(self):
        super().__init__()
        self.optimizer = self.optimizers[1]  
        self.batch_variation = False       
        self.name = "_".join([self.optimizer, "nobatch"])


class Adam_Batch(BaseConfig):
    def __init__(self):
        super().__init__()
        self.optimizer = self.optimizers[1]
        self.batch_variation = True
        self.name = "_".join([self.optimizer, "batch"])        
        
class Test_bg(Adam_Batch):
    def __init__(self):
        super().__init__()
        self.optimizer = self.optimizers[1]
        self.batch_variation = True
        self.bg_dir = os.path.join(self.path,"imgs/bg/road/img131.jpg") 
        self.name = "_".join([self.optimizer, "batch", "test_bg"])      

class Ratio30(Adam_Batch):
    def __init__(self):
        super().__init__()
        self.optimizer = self.optimizers[1]
        self.batch_variation = True
        self.patch_ratio = 0.30
        self.name = "_".join([self.optimizer, "batch", "ratio30"])        
                
class Triangle(Adam_Batch):
    def __init__(self):
        super().__init__()
        self.optimizer = self.optimizers[1]
        self.batch_variation = True
        self.from_mask = True
        self.name = "_".join([self.optimizer, "batch", "triangle"])        
        
class Triangle2(Adam_Batch):
    def __init__(self):
        super().__init__()
        self.optimizer = self.optimizers[1]
        self.batch_variation = True
        self.from_mask = True
        self.path_map4patch = os.path.join(self.path,'imgs/mask/c.png')
        self.path_map4stop = os.path.join(self.path,'imgs/mask/d.png')
        self.name = "_".join([self.optimizer, "batch", "triangle"])


class Test(BaseConfig):
    def __init__(self):
        super().__init__()
        self.optimizer = self.optimizers[1]
        self.batch_variation = True
        self.name = "_".join([self.optimizer, "batch", "test"])        
        self.nepochs = 6  #Number of iterations
        self.save_interval = 2
        self.ntests = 3
        self.test_interval = 2
        self.from_mask = False
        self.patch_ratio = 0.30
        self.from_mask = True
        self.bg_dir = os.path.join(self.path,"imgs/bg/road/img131.jpg") 
        
custom_configs = {
    "base": BaseConfig,
    "fgsm_nobatch": Fgsm_NoBatch,
    "fgsm_batch": Fgsm_Batch,
    "adam_nobatch": Adam_NoBatch,
    "adam_batch": Adam_Batch,    
    "test_bg": Test_bg,
    "ratio30": Ratio30,
    "triangle": Triangle, 
    "triangle2": Triangle2, 
    "test": Test,
}
