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
        
        # main parameters in AE generation
        self.session_name = time.strftime("%Y%m%d-%H%M%S")
        self.nepochs = 1001  #Number of iterations
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
        self.tv_p = 0.9
        self.nps_p = 0.5
        self.satur_p = 0.5
        self.fir_flag = False
        self.dist_flag = True
        self.tv_flag = True
        self.nps_flag = False
        self.satur_flag = True
        
        # attack approach
        self.batch_variation = True
        self.optimizers = ["fgsm, adam"]
        self.optimizer = self.optimizers[0]
        
class Seeing(BaseConfig):
    def __init__(self):
        super().__init__()
        self.exp = "adam" + "_batch" + "_dist"


custom_configs = {
    "base": BaseConfig,
    "seeing": Seeing
}
