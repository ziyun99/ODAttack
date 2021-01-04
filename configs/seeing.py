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
        self.save_interval = 20

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
        
        self.ntests = 1000
        self.test_interval = 250
        
        self.ori_index = 11
        
class Fgsm_NoBatch(BaseConfig):
    def __init__(self):
        super().__init__()
        self.name = "fgsm" + "_nobatch"
        self.batch_variation = False
        self.optimizer = self.optimizers[0]

class Fgsm_Batch(BaseConfig):
    def __init__(self):
        super().__init__()
        self.name = "fgsm" + "_batch" 
        self.batch_variation = True
        self.optimizer = self.optimizers[0]

class Adam_NoBatch(BaseConfig):
    def __init__(self):
        super().__init__()
        self.name = "adam" + "_nobatch"
        self.batch_variation = False
        self.optimizer = self.optimizers[1]

class Adam_Batch(BaseConfig):
    def __init__(self):
        super().__init__()
        self.name = "adam" + "_batch"
        self.batch_variation = True
        self.optimizer = self.optimizers[1]

        
      
custom_configs = {
    "base": BaseConfig,
    "fgsm_nobatch": Fgsm_NoBatch,
    "fgsm_batch": Fgsm_Batch,
    "adam_nobatch": Adam_NoBatch,
    "adam_batch": Adam_Batch,
}
