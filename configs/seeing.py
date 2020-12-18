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
        self.attack_method = "seeing"
        self.attack_class = "Seeing"
        self.path = os.path.join(os.getcwd(),'seeing/')
        self.out_path = os.path.join(os.getcwd(),'seeing/output/')
        
        # main parameters in AE generation
        self.session_name = time.strftime("%Y%m%d-%H%M%S")
        self.nsteps = 1001  #Number of iterations
        self.save_interval = 20
  
        self.video = 1
        self.fps = 1.2

        # directory
        self.stop_dir = "imgs/stop/"    #Image / Directory containing stop signs to generate AE
        self.bg_dir = "imgs/bg/road/"   #Image / Directory containing backgrounds to generate AE
        self.output_dir = "output/batch/"   #Image / Directory to store AE generated
    
        #detection parameters
        self.bs = 1     # Batch size
        self.confidence = 0.5
        self.nms_thresh = 0.4
        self.scales = "1,2,3"
    
        #model parameters
        self.cfg = "yolov3/cfg/yolov3.cfg" 
        self.weights = "yolov3/weights/yolov3.weights"
        self.classes = "yolov3/data/coco.names"
        self.reso = "416"  #Input resolution of the network. Increase to increase accuracy. Decrease to increase speed
    
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if CUDA else "cpu")
        
class Seeing(BaseConfig):
    def __init__(self):
        super().__init__()
        self.exp = "adam" + "_batch" + "_dist"


custom_configs = {
    "base": BaseConfig,
    "seeing": Seeing
}