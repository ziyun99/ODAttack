import sys
import argparse

# choose attack method
from attack_methods.eotb_attack import EOTB_attack
from attack_config import attack_config
# choose white-box models
# from object_detectors.yolo_tiny_model_updated import YOLO_tiny_model_updated

def load_config(argvs):
    '''
    Load config object.
    input: '-cfg configName'
    return: config, type: Object
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Required attack configuration, pls refer to attack_config/attack_config.py')
    args = parser.parse_args(argvs)
    print("Loading input configuration: ", args.cfg)
    try:
        config = attack_config.patch_configs[args.cfg]()
        return config
    except KeyError:
        print("Configuration {} not found, pls refer to attack_config/attack_config.py".format(args.cfg))
        print("Possible configurations are: ", list(attack_config.patch_configs.keys()))
        exit()

def create_attacker(argvs):
    config = load_config(argvs[1:3])
    args = argvs[3:]
    attack_class = globals()[config.attack_method]
    attacker = attack_class(config, args)
    return attacker

def main(argvs):
    attacker = create_attacker(argvs)
    print(attacker.__dict__)
    print(attacker.config.__dict__)
    # attacker.attack()
    
    
if __name__=='__main__':    
    main(sys.argv)
