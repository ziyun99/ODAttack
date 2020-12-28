# from torch import optim
import importlib
import os

class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.attack_method = "invisible"
        self.attack_class = "Invisible"
        self.path = os.path.join(os.getcwd(),'invisible/')
        self.out_path = os.path.join(os.getcwd(),'invisible/result/')

class ODD(BaseConfig):
    def __init__(self):
        # default value
        super().__init__()
        self.disp_console = True
        model_file = importlib.import_module("invisible.object_detectors.yolo_tiny_model_updated")
        self.model = getattr(model_file, "YOLO_tiny_model_updated")
        self.success = 0
        self.overall_pics = 0
        
        self.very_small = 0.000001
        self.mask_list = None
        
        # global variable to be parsed in argv_parser()
        self.h_img = None
        self.w_img = None
        self.d_img = None
        self.fromfile = os.path.join(self.path, "data_sampling") 
        self.frommaskfile = os.path.join(self.path, "test/EOTB.xml")
        self.fromlogofile = None
        self.fromfolder = None


        # init global variable
        self.filewrite_img = False
        self.filewrite_txt = False
        self.tofile_img = os.path.join(self.out_path,'output.jpg')
        self.tofile_txt = os.path.join(self.out_path,'output.txt')
        
        self.imshow = False
        self.useEOT = True
        self.Do_you_want_ad_sticker = True
        self.weights_file = os.path.join(self.path,'weights/YOLO_tiny.ckpt')
        
        # optimization settings
        self.learning_rate = 1e-2
        self.steps = 4
        self.alpha = 0.1
        self.threshold = 0.2
        self.iou_threshold = 0.5
        self.num_class = 20
        self.num_box = 2
        self.grid_size = 7
        self.classes =  ["aeroplane",
                         "bicycle", 
                         "bird", 
                         "boat", 
                         "bottle", 
                         "bus", 
                         "car", 
                         "cat", 
                         "chair", 
                         "cow", 
                         "diningtable", 
                         "dog", 
                         "horse", 
                         "motorbike", 
                         "person", 
                         "pottedplant", 
                         "sheep", 
                         "sofa", 
                         "train",
                         "tvmonitor"]


custom_configs = {
    "base": BaseConfig,
    "odd": ODD
}
