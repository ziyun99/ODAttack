import sys
import os
import importlib

def create_attacker(argvs):
    # get params
    if len(argvs) < 3:
        print("Usage: python3 main.py [attack_method] [attack_config]")
        print("eg. python3 main.py fooling paper_obj")
        exit()
    input_method = argvs[1]
    input_config = argvs[2]
    input_args = argvs[3:]

    # add to PYTHONPATH
    attackpath = os.path.join(os.getcwd(),input_method) 
    sys.path.insert(0, attackpath)

    # get config
    try:
        cfgfile = importlib.import_module("configs." + input_method)
        config = cfgfile.custom_configs[input_config]()
    # except ModuleNotFoundError as e:
    #     print(e)
    #     exit()
    except KeyError:
        print("Configuration '{}' not found, pls refer to configs/{}".format(input_config, input_method))
        print("Possible configurations for '{}' are: {}".format(input_method, list(cfgfile.custom_configs.keys())))
        exit()
        
    # get attacker class
    attackfile = importlib.import_module(input_method + ".attack")
    attack_class = getattr(attackfile, config.attack_class)
    attacker = attack_class(config, input_args)
    
    return attacker

def main():
    attacker = create_attacker(sys.argv)
    print(attacker.__dict__)
    print(attacker.config.__dict__)
    # attacker.attack()
    
    
if __name__=='__main__':
    main()
