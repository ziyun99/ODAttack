# from torch import optim
import importlib
import os

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

        self.verbose = True
        self.seed = None

        #Object Detection Model
        self.model = ""
        self.batch-size = 1
        self.train-batch-size = 1000
        self.test-batch-size = 1000

        #Inputs
        self.backgrounds = ""

        self.textures = ""
        self.textures-masks = ""

        self.texture-yaw-min = 0.
        self.texture-yaw-max = 0.
        self.texture-yaw-bins = 100
        self.texture_yaw_fn = np.linspace #np.logspace #texture-yaw-logspace
        self.texture-pitch-min = 0.
        self.texture-pitch-max = 0.
        self.texture-pitch-bins = 100
        self.texture_pitch_fn = np.linspace #np.logspace #texture-pitch-logspace
        self.texture-roll-min = 0.
        self.texture-roll-max = 0.
        self.texture-roll-bins = 100
        self.texture_roll_fn = np.linspace #np.logspace texture-roll-logspace

        self.texture-x-min = 0.
        self.texture-x-max = 0.
        self.texture-x-bins = 100
        self.texture_x_fn  = np.linspace #np.logspace texture-x-logspace
        self.texture-y-min = 0.
        self.texture-y-max = 0.
        self.texture-y-bins = 100
        self.texture_y_fn  = np.linspace #np.logspace texture-y-logspace
        self.texture-z-min = 0.
        self.texture-z-max = 0.
        self.texture-z-bins = 100
        self.texture_z_fn = np.linspace #np.logspace texture-z-logspace

        self.texture-multiplicative-channel-noise-min = 1.
        self.texture-multiplicative-channel-noise-max = 1.
        self.texture-additive-channel-noise-min = 0.
        self.texture-additive-channel-noise-max = 0.
        self.texture-multiplicative-pixel-noise-min = 1.
        self.texture-multiplicative-pixel-noise-max = 2.
        self.texture-additive-pixel-noise-min = 0.
        self.texture-additive-pixel-noise-max = 0.
        self.texture-gaussian-noise-stddev-min = 0.
        self.texture-gaussian-noise-stddev-max = 0.

        self.objects = []

        self.object-yaw-min = 0.
        self.object-yaw-max = 0.
        self.object-yaw-bins = 100
        self.object_yaw_fn = np.linspace #np.logspace object-yaw-logspace
        self.object-pitch-min = 0.
        self.object-pitch-max = 0.
        self.object-pitch-bins = 100
        self.object_pitch_fn = np.linspace #np.logspace object-pitch-logspace
        self.object-roll-min = 0.
        self.object-roll-max = 0.
        self.object-roll-bins = 100
        self.object_roll_fn = np.linspace #np.logspace object-roll-logspace

        self.object-x-min = 0.
        self.object-x-max = 0.
        self.object-x-bins = 100
        self.object_x_fn = np.linspace #np.logspace object-x-logspace
        self.object-y-min = 0.
        self.object-y-max = 0.
        self.object-y-bins = 100
        self.object_y_fn = np.linspace #np.logspace object-y-logspace
        self.object-z-min = 0.
        self.object-z-max = 0.
        self.object-z-bins = 100
        self.object_z_fn = np.linspace #np.logspace object-z-logspace

        self.object-multiplicative-channel-noise-min = 1.
        self.object-multiplicative-channel-noise-max = 1.
        self.object-additive-channel-noise-min = 0.
        self.object-additive-channel-noise-max = 0.
        self.object-multiplicative-pixel-noise-min = 1.
        self.object-multiplicative-pixel-noise-max = 1.
        self.object-additive-pixel-noise-min = 0.
        self.object-additive-pixel-noise-max = 0.
        self.object-gaussian-noise-stddev-min = 0.
        self.object-gaussian-noise-stddev-max = 0.

        self.image-multiplicative-channel-noise-min = 1.
        self.image-multiplicative-channel-noise-max = 1.
        self.image-additive-channel-noise-min = 0.
        self.image-additive-channel-noise-max = 0.
        self.image-multiplicative-pixel-noise-min = 1.
        self.image-multiplicative-pixel-noise-max = 1.
        self.image-additive-pixel-noise-min = 0.
        self.image-additive-pixel-noise-max = 0.
        self.image-gaussian-noise-stddev-min = 0.
        self.image-gaussian-noise-stddev-max = 0.

        #Attack
        self.optimizer = 'gd'  #['gd', 'momentum', 'rmsprop', 'adam']
        self.learning-rate = 1.
        self.momentum = 0.
        self.decay = 0.
        self.sign-gradients = False

        self.gray-start = False
        self.random-start = 0

        self.spectral = True 
        self.soft-clipping = False

        self.target = 'bird'
        self.victim = 'person'

        self.rpn-iou-threshold = 0.7
        self.rpn-cls-weight = 1.
        self.rpn-loc-weight = 2.
        self.rpn-foreground-weight = 0.
        self.rpn-background-weight = 0.
        self.rpn-cw-weight = 0.
        self.rpn-cw-conf = 0.

        self.box-iou-threshold = 0.5
        self.box-cls-weight = 1.
        self.box-loc-weight = 2.
        self.box-target-weight = 0.
        self.box-victim-weight = 0.
        self.box-target-cw-weight = 0.
        self.box-target-cw-conf = 0.
        self.box-victim-cw-weight = 0.
        self.box-victim-cw-conf = 0.

        self.sim-weight = 0.

        #Metrics
        self.logdir = ""
        self.save-graph = False
        self.save-train-every = 1
        self.save-texture-every = 10
        self.save-checkpoint-every = 10
        self.save-test-every = 10

class ShapeshifterBase(BaseConfig):      
    def __init__(self):
        super().__init__()
        self.rpn-loc-weight = 0 
        self.rpn-cls-weight = 0 
        self.rpn-cw-weight = 0 
        self.rpn-cw-conf = 0 
        self.box-cls-weight = 1
        self.box-loc-weight = 0
        self.box-victim-cw-weight = 0
        self.box-victim-cw-conf = 0
        self.box-target-cw-weight = 0
        self.box-target-cw-conf = 0
        self.box-victim-weight = 0
        self.box-target-weight = 0
        self.sim-weight = 0.0001

        self.image-multiplicative-channel-noise-min = 0.7     
        self.image-multiplicative-channel-noise-max = 1.3                   
        self.image-additive-channel-noise-min = -0.15 
        self.image-additive-channel-noise-max = 0.15 
        self.image-multiplicative-pixel-noise-min = 0.5 
        self.image-multiplicative-pixel-noise-max = 2.0 
        self.image-additive-pixel-noise-min = -0.15 
        self.image-additive-pixel-noise-max = 0.15 
        self.image-gaussian-noise-stddev-min = 0.0 
        self.image-gaussian-noise-stddev-max = 0.1 

        self.batch-size = 1 
        self.train-batch-size = 10 
        self.test-batch-size = 1000 
        self.save-train-every = 10 
        self.save-texture-every = 100 
        self.save-checkpoint-every = 100 
        self.save-test-every = 100

class StopSignBase(ShapeshifterBase):
    def __init__(self):
        super().__init__()
        self.verbose = True ###
        self.model = os.path.join(self.path, MODELS_DIR + MODEL_NAME)
        self.backgrounds = [os.path.join(self.path, DATA_DIR + "background.png"))]
        self.objects = [os.path.join(self.path, DATA_DIR + "stop_sign_object.png")]
        self.textures = [os.path.join(self.path, DATA_DIR + "stop_sign_mask.png")]
        self.textures-masks = [os.path.join(self.path, DATA_DIR + "stop_sign_mask.png")]

        self.victim = "stop sign"
        self.target = "person"
        self.logdir = os.path.join(self.path, LOG_DIR)

        self.object-roll-min = -15
        self.object-roll-max = 15
        self.object-x-min = -500
        self.object-x-max = 500
        self.object-y-min = -200
        self.object-y-max = 200
        self.object-z-min = 0.1
        self.object-z-max = 1.0

        self.texture-roll-min = 0
        self.texture-roll-max = 0
        self.texture-x-min = 0
        self.texture-x-max = 0
        self.texture-y-min = 0
        self.texture-y-max = 0
        self.texture-z-min = 1.0
        self.texture-z-max = 1.0

        self.optimizer = "gd"
        self.learning-rate = 0.001
        self.spectral = False
        self.sign-gradients = True
        self.gray-start = True   ###


# 2d_stopsign_targeted_attack  ## Create 2d stop sign that is detected as a person.
class StopSignTargeted(StopSignBase):
    def __init__(self):
        super().__init__()
        self.victim = "stop sign"
        self.target = "person"
        self.rpn-cls-weight = 0 
        self.box-cls-weight = 1 

# 2d_stopsign_untargeted_attack ## Create 2d stop sign that is not detected as a stop sign.
class StopSignUntargeted(StopSignBase):
    def __init__(self):
        super().__init__()
        self.victim = "stop sign"
        self.target = "stop sign"
        self.rpn-cls-weight = 0 
        self.box-cls-weight = -1 

# 2d_stopsign_proposal_attack ## Create 2d stop sign that is not detected.
class StopSignProposal(StopSignBase):
    def __init__(self):
        super().__init__()
        self.victim = "stop sign"
        self.target = "person"
        self.rpn-cls-weight = -1 
        self.box-cls-weight = 0
        self.sim-weight = 0.00005

# 2d_stopsign_hybrid_targeted_attack ## Create 2d stop sign that is either not detected at all or detected as a person.
class StopSignHybridTargeted(StopSignBase):
    def __init__(self):
        super().__init__()
        self.victim = "stop sign"
        self.target = "person"
        self.rpn-cls-weight = -1 
        self.box-cls-weight = 4

# 2d_stopsign_hybrid_untargeted_attack ## Create 2d stop sign that is either not detected at all or not detected as a stop sign.
class StopSignHybridUntargeted(StopSignBase):
    def __init__(self):
        super().__init__()
        self.victim = "stop sign"
        self.target = "stop sign"
        self.rpn-cls-weight = -4 
        self.box-cls-weight = 1

class PersonBase(ShapeshifterBase):
    def __init__(self):
        super().__init__()

        self.verbose = True ###
        self.model = os.path.join(self.path, MODELS_DIR + MODEL_NAME)
        self.backgrounds = [os.path.join(self.path, DATA_DIR + "background2.png"))]
        self.objects = [os.path.join(self.path, DATA_DIR + "person.png")]
        self.textures = [os.path.join(self.path, DATA_DIR + "quarter_sheet.png")]
        self.textures-masks = [os.path.join(self.path, DATA_DIR + "quarter_sheet.png")]

        self.victim = "person"
        self.target = "bird"
        self.logdir = os.path.join(self.path, LOG_DIR)

        self.object-roll-min = -5
        self.object-roll-max = 5
        self.object-roll-bins = 10
        self.object-x-min = -1000
        self.object-x-max = 1000
        self.object-x-bins = 100
        self.object-y-min = -500
        self.object-y-max = 500
        self.object-y-bins = 100
        self.object-z-min = 0.5
        self.object-z-max = 1.0
        self.object-z-bins = 5

        self.texture-roll-min = -5
        self.texture-roll-max = 5
        self.texture-roll-bins = 10
        self.texture-x-min = -30
        self.texture-x-max = 30
        self.texture-x-bins = 60
        self.texture-y-min = -100
        self.texture-y-max = 20
        self.texture-y-bins = 120
        self.texture-z-min = 0.35
        self.texture-z-max = 0.4
        self.texture-z-bins = 10

        self.optimizer = "gd"
        self.learning-rate = 0.00392156862
        self.spectral = False
        self.sign-gradients = True
        self.gray-start = True   ###
        self.random-start = 1

# 2d_person_targeted_attack ## Create 2d tshirt that is detected as a bird.
class PersonTargeted(PersonBase):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn-cls-weight = 0 
        self.box-cls-weight = 1

# 2d_person_untargeted ## Create 2d tshirt that is not detected as a person.
class PersonUntargeted(PersonBase):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "person"
        self.rpn-cls-weight = 0 
        self.box-cls-weight = -1

# 2d_person_proposal_attack ## Create 2d tshirt that is not detected.
class PersonProposal(PersonBase):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn-cls-weight = -1
        self.box-cls-weight = 0

# 2d_person_hybrid_targeted ## Create 2d tshirt that is either not detected or is detected as a bird.
class PersonHybridTargeted(PersonBase):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn-cls-weight = -1
        self.box-cls-weight = 5

# 2d_person_hybrid_untargeted ## Create 2d tshirt that is either not detected at all or not detected as a person.
class PersonHybridUntargeted(PersonBase):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "person"
        self.rpn-cls-weight = -1
        self.box-cls-weight = -0.1

class PersonBase3D(ShapeshifterBase):
    def __init__(self):
        super().__init__()

        self.verbose = True ###
        self.model = os.path.join(self.path, MODELS_DIR + MODEL_NAME)
        self.backgrounds = [os.path.join(self.path, DATA_DIR + "background2.png")]
        self.objects = [os.path.join(self.path, DATA_DIR + "man_*.obj")]
        self.textures = [os.path.join(self.path, DATA_DIR + "man_outfit_small.png"), os.path.join(self.path, DATA_DIR + "man_skin_small.png")]
        self.textures-masks = [os.path.join(self.path, DATA_DIR + "man_outfit_small_mask.png"), os.path.join(self.path, DATA_DIR + "man_skin_small_mask.png")]

        self.victim = "person"
        self.target = "bird"
        self.logdir = os.path.join(self.path, LOG_DIR)

        self.object-yaw-min = 85 
        self.object-yaw-max = 130 
        self.object-yaw-bins = 45 
        self.object-pitch-min = -5 
        self.object-pitch-max = 5 
        self.object-pitch-bins = 10

        self.object-roll-min = -5
        self.object-roll-max = 5
        self.object-roll-bins = 10
        self.object-x-min = -10
        self.object-x-max = 6.5
        self.object-x-bins = 33
        self.object-y-min = 0.5
        self.object-y-max = 1.0
        self.object-y-bins = 5
        self.object-z-min = 32
        self.object-z-max = 70
        self.object-z-bins = 38

        self.optimizer = "gd"
        self.learning-rate = 0.00392156862
        self.spectral = False
        self.sign-gradients = True
        self.gray-start = True   ###
        self.random-start = 1

# 3d_person_targeted_attack ## Create 3d outfit that is detected as a bird.
class PersonTargeted3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn-cls-weight = 1
        self.box-cls-weight = 0

# 3d_person_untargeted_attack ## Create 3d outfit that is not detected as a person.
class PersonUntargeted3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "person"
        self.rpn-cls-weight = 0
        self.box-cls-weight = -1

# 3d_person_proposal_attack ## Create 3d outfit that is not detected.
class PersonProposal3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn-cls-weight = -1
        self.box-cls-weight = 0

# 3d_person_hybrid_targeted_attack ## Create 3d outfit that is either not detected at all or detected as a bird.
class PersonHybridTargeted3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "bird"
        self.rpn-cls-weight = -1
        self.box-cls-weight = 5

# 3d_person_hybrid_untargeted_attack ## Create 3d outfit that is either not detected at all or not detected as a person.
class PersonHybridUntargeted3D(PersonBase3D):
    def __init__(self):
        super().__init__()
        self.victim = "person"
        self.target = "person"
        self.rpn-cls-weight = -1
        self.box-cls-weight = -5

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

    # "person_t_3d": PersonTargeted3D,
    # "person_ut_3d": PersonUntargeted3D,
    # "person_p_3d": PersonProposal3D,
    # "person_hybrid_t_3d": PersonHybridTargeted3D,
    # "person_hybrid_ut_3d": PersonHybridUntargeted3D,
}
