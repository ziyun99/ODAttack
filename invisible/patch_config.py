from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "inria/Train/pos"
        self.lab_dir = "inria/Train/pos/yolo-labels"
        self.cfgfile = "cfg/yolo.cfg"
        self.weightfile = "weights/yolo.weights"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 300

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 20

        self.loss_target = lambda obj, cls: obj * cls

from object_detectors.yolo_tiny_model_updated import YOLO_tiny_model_updated

class ODD_Base(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        # default value
        self.disp_console = True
        self.model = YOLO_tiny_model_updated
        self.success = 0
        self.overall_pics = 0
        self.path = './result/'
        self.very_small = 0.000001
        self.mask_list = None
        
        # global variable to be parsed in argv_parser()
        self.h_img = None
        self.w_img = None
        self.d_img = None
        self.fromfile = None 
        self.frommaskfile = None
        self.fromlogofile = None
        self.fromfolder = None

class ODD(ODD_Base):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        # default value
        self.disp_console = True
        self.model = YOLO_tiny_model_updated
        self.success = 0
        self.overall_pics = 0
        self.path = './result/'
        self.very_small = 0.000001
        self.mask_list = None
        
        # global variable to be parsed in argv_parser()
        self.h_img = None
        self.w_img = None
        self.d_img = None
        self.fromfile = None 
        self.frommaskfile = None
        self.fromlogofile = None
        self.fromfolder = None


        # init global variable
        self.filewrite_img = False
        self.filewrite_txt = False
        self.tofile_img = os.path.join(self.path,'output.jpg')
        self.tofile_txt = os.path.join(self.path,'output.txt')
        
        self.imshow = False
        self.useEOT = True
        self.Do_you_want_ad_sticker = True
        self.weights_file = 'weights/YOLO_tiny.ckpt'
        
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


patch_configs = {
    "base": BaseConfig,
    "odd": ODD
}
