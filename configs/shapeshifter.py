# from torch import optim
import importlib
import os
import numpy as np

MODEL_NAME="faster_rcnn_inception_v2_coco_2017_11_08"
MODELS_DIR="models/"
DATA_DIR="data/"
LOG_DIR="logdir/"

class BaseConfig(object):
    """
    Default parameters for all config files.
    """
    def __init__(self):
        self.attack_method = "shapeshifter"
        self.attack_class = "Shapeshifter"
        self.path = os.path.join(os.getcwd(),'shapeshifter/')
        self.out_path = os.path.join(os.getcwd(),'shapeshifter/result/')
        self.logdir = os.path.join(self.path, LOG_DIR)
        self.name = None
        self.new_run = False
        
        self.train_steps = 8000
        self.run_test = True
        
        self.verbose = True
        self.seed = None

        #Object Detection Model
        self.model = os.path.join(self.path, MODELS_DIR + MODEL_NAME)
        self.batch_size = 1
        self.train_batch_size = 1000
        self.test_batch_size = 1000

        #Inputs
        self.backgrounds = ""

        self.textures = ""
        self.textures_masks = ""

        self.texture_yaw_min = 0.
        self.texture_yaw_max = 0.
        self.texture_yaw_bins = 100
        self.texture_yaw_fn = np.linspace #np.logspace #texture_yaw_logspace
        self.texture_pitch_min = 0.
        self.texture_pitch_max = 0.
        self.texture_pitch_bins = 100
        self.texture_pitch_fn = np.linspace #np.logspace #texture_pitch_logspace
        self.texture_roll_min = -10.0
        self.texture_roll_max = 10.0
        self.texture_roll_bins = 100
        self.texture_roll_fn = np.linspace #np.logspace texture_roll_logspace

        self.texture_x_min = -200
        self.texture_x_max = 50
        self.texture_x_bins = 100
        self.texture_x_fn  = np.linspace #np.logspace texture_x_logspace
        self.texture_y_min = -100
        self.texture_y_max = 150
        self.texture_y_bins = 100
        self.texture_y_fn  = np.linspace #np.logspace texture_y_logspace
        self.texture_z_min = 0.3
        self.texture_z_max = 0.35
        self.texture_z_bins = 100
        self.texture_z_fn = np.linspace #np.logspace texture_z_logspace

        self.texture_multiplicative_channel_noise_min = 1.
        self.texture_multiplicative_channel_noise_max = 1.
        self.texture_additive_channel_noise_min = 0.
        self.texture_additive_channel_noise_max = 0.
        self.texture_multiplicative_pixel_noise_min = 1.
        self.texture_multiplicative_pixel_noise_max = 2.
        self.texture_additive_pixel_noise_min = 0.
        self.texture_additive_pixel_noise_max = 0.
        self.texture_gaussian_noise_stddev_min = 0.
        self.texture_gaussian_noise_stddev_max = 0.

        self.objects = []

        self.object_yaw_min = 0.
        self.object_yaw_max = 0.
        self.object_yaw_bins = 100
        self.object_yaw_fn = np.linspace #np.logspace object_yaw_logspace
        self.object_pitch_min = 0.
        self.object_pitch_max = 0.
        self.object_pitch_bins = 100
        self.object_pitch_fn = np.linspace #np.logspace object_pitch_logspace
        self.object_roll_min = -5.0
        self.object_roll_max = 5.0
        self.object_roll_bins = 100
        self.object_roll_fn = np.linspace #np.logspace object_roll_logspace

        self.object_x_min = 0.
        self.object_x_max = 0.
        self.object_x_bins = 100
        self.object_x_fn = np.linspace #np.logspace object_x_logspace
        self.object_y_min = -500
        self.object_y_max = 500
        self.object_y_bins = 100
        self.object_y_fn = np.linspace #np.logspace object_y_logspace
        self.object_z_min = 0.5
        self.object_z_max = 1.1
        self.object_z_bins = 100
        self.object_z_fn = np.linspace #np.logspace object_z_logspace

        self.object_multiplicative_channel_noise_min = 1.
        self.object_multiplicative_channel_noise_max = 1.
        self.object_additive_channel_noise_min = 0.
        self.object_additive_channel_noise_max = 0.
        self.object_multiplicative_pixel_noise_min = 1.
        self.object_multiplicative_pixel_noise_max = 1.
        self.object_additive_pixel_noise_min = 0.
        self.object_additive_pixel_noise_max = 0.
        self.object_gaussian_noise_stddev_min = 0.
        self.object_gaussian_noise_stddev_max = 0.

        self.image_multiplicative_channel_noise_min = 1.
        self.image_multiplicative_channel_noise_max = 1.
        self.image_additive_channel_noise_min = 0.
        self.image_additive_channel_noise_max = 0.
        self.image_multiplicative_pixel_noise_min = 1.
        self.image_multiplicative_pixel_noise_max = 1.
        self.image_additive_pixel_noise_min = 0.
        self.image_additive_pixel_noise_max = 0.
        self.image_gaussian_noise_stddev_min = 0.
        self.image_gaussian_noise_stddev_max = 0.

        #Attack
        self.optimizer = 'gd'  #['gd', 'momentum', 'rmsprop', 'adam']
        self.learning_rate = 1.
        self.momentum = 0.
        self.decay = 0.
        self.sign_gradients = False

        self.gray_start = False
        self.random_start = 0

        self.spectral = True 
        self.soft_clipping = False

        self.target = 'bird'
        self.victim = 'person'

        self.rpn_iou_threshold = 0.7
        self.rpn_cls_weight = 1.
        self.rpn_loc_weight = 2.
        self.rpn_foreground_weight = 0.
        self.rpn_background_weight = 0.
        self.rpn_cw_weight = 0.
        self.rpn_cw_conf = 0.

        self.box_iou_threshold = 0.5
        self.box_cls_weight = 1.
        self.box_loc_weight = 2.
        self.box_target_weight = 0.
        self.box_victim_weight = 0.
        self.box_target_cw_weight = 0.
        self.box_target_cw_conf = 0.
        self.box_victim_cw_weight = 0.
        self.box_victim_cw_conf = 0.

        self.sim_weight = 0.

        #Metrics
        self.logdir = ""
        self.save_graph = False
        self.save_train_every = 1
        self.save_texture_every = 10
        self.save_checkpoint_every = 10
        self.save_test_every = 10

class ShapeshifterBase(BaseConfig):      
    def __init__(self):
        super().__init__()
        self.rpn_loc_weight = 0 
        self.rpn_cls_weight = 0 
        self.rpn_cw_weight = 0 
        self.rpn_cw_conf = 0 
        self.box_cls_weight = 1
        self.box_loc_weight = 0
        self.box_victim_cw_weight = 0
        self.box_victim_cw_conf = 0
        self.box_target_cw_weight = 0
        self.box_target_cw_conf = 0
        self.box_victim_weight = 0
        self.box_target_weight = 0
        self.sim_weight = 0.0001

        self.image_multiplicative_channel_noise_min = 0.7     
        self.image_multiplicative_channel_noise_max = 1.3                   
        self.image_additive_channel_noise_min = -0.15 
        self.image_additive_channel_noise_max = 0.15 
        self.image_multiplicative_pixel_noise_min = 0.5 
        self.image_multiplicative_pixel_noise_max = 2.0 
        self.image_additive_pixel_noise_min = -0.15 
        self.image_additive_pixel_noise_max = 0.15 
        self.image_gaussian_noise_stddev_min = 0.0 
        self.image_gaussian_noise_stddev_max = 0.1 

        self.batch_size = 1 
        self.train_batch_size = 10 
        self.test_batch_size = 1000 
        self.save_train_every = 10 
        self.save_texture_every = 100 
        self.save_checkpoint_every = 100 
        self.save_test_every = 100

class StopSignBase(ShapeshifterBase):
    def __init__(self):
        super().__init__()
        self.verbose = True ###
        self.model = os.path.join(self.path, MODELS_DIR + MODEL_NAME)
        self.backgrounds = [os.path.join(self.path, DATA_DIR + "background.png")]
        self.objects = [os.path.join(self.path, DATA_DIR + "stop_sign_object.png")]
        self.textures = [os.path.join(self.path, DATA_DIR + "stop_sign_mask.png")]
        self.textures_masks = [os.path.join(self.path, DATA_DIR + "stop_sign_mask.png")]

        self.victim = "stop sign"
        self.target = "person"
        self.logdir = os.path.join(self.path, LOG_DIR)

        self.object_roll_min = -15
        self.object_roll_max = 15
        self.object_x_min = -500
        self.object_x_max = 500
        self.object_y_min = -200
        self.object_y_max = 200
        self.object_z_min = 0.1
        self.object_z_max = 1.0

        self.texture_roll_min = 0
        self.texture_roll_max = 0
        self.texture_x_min = 0
        self.texture_x_max = 0
        self.texture_y_min = 0
        self.texture_y_max = 0
        self.texture_z_min = 1.0
        self.texture_z_max = 1.0

        self.optimizer = "gd"
        self.learning_rate = 0.001
        self.spectral = False
        self.sign_gradients = True
        self.gray_start = True   ###


# 2d_stopsign_targeted_attack  ## Create 2d stop sign that is detected as a person.
class StopSignTargeted(StopSignBase):
    def __init__(self):
        super().__init__()
        self.name = "stop_sign_targeted"
        self.new_run = True
        self.victim = "stop sign"
        self.target = "person"
        self.rpn_cls_weight = 0 
        self.box_cls_weight = 1 
  
# 2d_stopsign_untargeted_attack ## Create 2d stop sign that is not detected as a stop sign.
class StopSignUntargeted(StopSignBase):
    def __init__(self):
        super().__init__()
        self.victim = "stop sign"
        self.target = "stop sign"
        self.rpn_cls_weight = 0 
        self.box_cls_weight = -1 

# 2d_stopsign_proposal_attack ## Create 2d stop sign that is not detected.
class StopSignProposal(StopSignBase):
    def __init__(self):
        super().__init__()
        self.victim = "stop sign"
        self.target = "person"
        self.rpn_cls_weight = -1 
        self.box_cls_weight = 0
        self.sim_weight = 0.00005

# 2d_stopsign_hybrid_targeted_attack ## Create 2d stop sign that is either not detected at all or detected as a person.
class StopSignHybridTargeted(StopSignBase):
    def __init__(self):
        super().__init__()
        self.victim = "stop sign"
        self.target = "person"
        self.rpn_cls_weight = -1 
        self.box_cls_weight = 4

# 2d_stopsign_hybrid_untargeted_attack ## Create 2d stop sign that is either not detected at all or not detected as a stop sign.
class StopSignHybridUntargeted(StopSignBase):
    def __init__(self):
        super().__init__()
        self.victim = "stop sign"
        self.target = "stop sign"
        self.rpn_cls_weight = -4 
        self.box_cls_weight = 1

class PersonBase(ShapeshifterBase):
    def __init__(self):
        super().__init__()

        self.verbose = True ###
        self.model = os.path.join(self.path, MODELS_DIR + MODEL_NAME)
        self.backgrounds = [os.path.join(self.path, DATA_DIR + "background2.png")]
        self.objects = [os.path.join(self.path, DATA_DIR + "person.png")]
        self.textures = [os.path.join(self.path, DATA_DIR + "quarter_sheet.png")]
        self.textures_masks = [os.path.join(self.path, DATA_DIR + "quarter_sheet.png")]

        self.victim = "person"
        self.target = "bird"
        self.logdir = os.path.join(self.path, LOG_DIR)

        self.object_roll_min = -5
        self.object_roll_max = 5
        self.object_roll_bins = 10
        self.object_x_min = -1000
        self.object_x_max = 1000
        self.object_x_bins = 100
        self.object_y_min = -500
        self.object_y_max = 500
        self.object_y_bins = 100
        self.object_z_min = 0.5
        self.object_z_max = 1.0
        self.object_z_bins = 5

        self.texture_roll_min = -5
        self.texture_roll_max = 5
        self.texture_roll_bins = 10
        self.texture_x_min = -30
        self.texture_x_max = 30
        self.texture_x_bins = 60
        self.texture_y_min = -100
        self.texture_y_max = 20
        self.texture_y_bins = 120
        self.texture_z_min = 0.35
        self.texture_z_max = 0.4
        self.texture_z_bins = 10

        self.optimizer = "gd"
        self.learning_rate = 0.00392156862
        self.spectral = False
        self.sign_gradients = True
        self.gray_start = True   ###
        self.random_start = 1

# 2d_person_targeted_attack ## Create 2d tshirt that is detected as a bird.
class PersonTargeted(PersonBase):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn_cls_weight = 0 
        self.box_cls_weight = 1

# 2d_person_untargeted ## Create 2d tshirt that is not detected as a person.
class PersonUntargeted(PersonBase):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "person"
        self.rpn_cls_weight = 0 
        self.box_cls_weight = -1

# 2d_person_proposal_attack ## Create 2d tshirt that is not detected.
class PersonProposal(PersonBase):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn_cls_weight = -1
        self.box_cls_weight = 0

# 2d_person_hybrid_targeted ## Create 2d tshirt that is either not detected or is detected as a bird.
class PersonHybridTargeted(PersonBase):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn_cls_weight = -1
        self.box_cls_weight = 5

# 2d_person_hybrid_untargeted ## Create 2d tshirt that is either not detected at all or not detected as a person.
class PersonHybridUntargeted(PersonBase):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "person"
        self.rpn_cls_weight = -1
        self.box_cls_weight = -0.1

class PersonBase3D(ShapeshifterBase):
    def __init__(self):
        super().__init__()

        self.verbose = True ###
        self.model = os.path.join(self.path, MODELS_DIR + MODEL_NAME)
        self.backgrounds = [os.path.join(self.path, DATA_DIR + "background2.png")]
        self.objects = [os.path.join(self.path, DATA_DIR + "man_*.obj")]
        self.textures = [os.path.join(self.path, DATA_DIR + "man_outfit_small.png"), os.path.join(self.path, DATA_DIR + "man_skin_small.png")]
        self.textures_masks = [os.path.join(self.path, DATA_DIR + "man_outfit_small_mask.png"), os.path.join(self.path, DATA_DIR + "man_skin_small_mask.png")]

        self.victim = "person"
        self.target = "bird"
        self.logdir = os.path.join(self.path, LOG_DIR)

        self.object_yaw_min = 85 
        self.object_yaw_max = 130 
        self.object_yaw_bins = 45 
        self.object_pitch_min = -5 
        self.object_pitch_max = 5 
        self.object_pitch_bins = 10

        self.object_roll_min = -5
        self.object_roll_max = 5
        self.object_roll_bins = 10
        self.object_x_min = -10
        self.object_x_max = 6.5
        self.object_x_bins = 33
        self.object_y_min = 0.5
        self.object_y_max = 1.0
        self.object_y_bins = 5
        self.object_z_min = 32
        self.object_z_max = 70
        self.object_z_bins = 38

        self.optimizer = "gd"
        self.learning_rate = 0.00392156862
        self.spectral = False
        self.sign_gradients = True
        self.gray_start = True   ###
        self.random_start = 1

# 3d_person_targeted_attack ## Create 3d outfit that is detected as a bird.
class PersonTargeted3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn_cls_weight = 1
        self.box_cls_weight = 0

# 3d_person_untargeted_attack ## Create 3d outfit that is not detected as a person.
class PersonUntargeted3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "person"
        self.rpn_cls_weight = 0
        self.box_cls_weight = -1

# 3d_person_proposal_attack ## Create 3d outfit that is not detected.
class PersonProposal3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn_cls_weight = -1
        self.box_cls_weight = 0

# 3d_person_hybrid_targeted_attack ## Create 3d outfit that is either not detected at all or detected as a bird.
class PersonHybridTargeted3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn_cls_weight = -1
        self.box_cls_weight = 5

# 3d_person_hybrid_untargeted_attack ## Create 3d outfit that is either not detected at all or not detected as a person.
class PersonHybridUntargeted3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "person"
        self.rpn_cls_weight = -1
        self.box_cls_weight = -5

        
class Patch(StopSignTargeted):
    def __init__(self):
        super().__init__()
        self.name = "patch"
        self.new_run = False
        self.textures_masks = [os.path.join(self.path, DATA_DIR + "patch_mask.png")]
        
class Triangle(StopSignTargeted):
    def __init__(self):
        super().__init__()
        self.name = "triangle"
        self.new_run = False
        self.textures_masks = [os.path.join(self.path, DATA_DIR + "triangle_mask.png")]          
#         self.run_test = False
#         self.save_texture_every = 2
        
custom_configs = {
    "base": BaseConfig,
    "stop_t": StopSignTargeted,
    "stop_ut": StopSignUntargeted,
    "stop_p": StopSignProposal,
    "stop_hybrid_t": StopSignHybridTargeted,
    "stop_hybrid_ut": StopSignHybridUntargeted,
    
    "person_t": PersonTargeted,
    "person_ut": PersonUntargeted,
    "person_p": PersonProposal,
    "person_hybrid_t": PersonHybridTargeted,
    "person_hybrid_ut": PersonHybridUntargeted,

    "patch": Patch,
    "triangle": Triangle,
    # "person_t_3d": PersonTargeted3D,
    # "person_ut_3d": PersonUntargeted3D,
    # "person_p_3d": PersonProposal3D,
    # "person_hybrid_t_3d": PersonHybridTargeted3D,
    # "person_hybrid_ut_3d": PersonHybridUntargeted3D,
}
