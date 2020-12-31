import sys
import os
import importlib

attack_methods = ["fooling", "invisible", "seeing", "shapeshifter"]

def create_attacker(argvs):
    # get params

    if len(argvs) == 2:
        input_method = argvs[1]
        if input_method not in attack_methods:
            print("Usage: python3 main.py [attack_method] [attack_config]")
            print("Attack method '{}' not found, possible attack methods are {}".format(input_method, str(attack_methods)))
            exit()
        cfgfile = importlib.import_module("configs." + input_method)
        print("Usage: python3 main.py [attack_method] [attack_config]") 
        print("Possible configurations for '{}' are: {}, pls refer to 'configs/{}.py' for more details".format(input_method, list(cfgfile.custom_configs.keys()), input_method))
        exit()
        
    if len(argvs) < 3:
        print("Usage: python3 main.py [attack_method] [attack_config]")
        print("Possible attack methods are {}".format(str(attack_methods)))
        print("eg. python3 main.py fooling paper_obj")
        exit()
        
        
    input_method = argvs[1]
    input_config = argvs[2]
    input_args = argvs[3:]
    
    if input_method not in attack_methods:
        print("Attack method '{}' not found, possible attack methods are {}".format(input_method, str(attack_methods)))    

    # add to PYTHONPATH
    attackpath = os.path.join(os.getcwd(),input_method) 
    sys.path.insert(0, attackpath)
    
    # get config
    try:
        cfgfile = importlib.import_module("configs." + input_method)
        config = cfgfile.custom_configs[input_config]()
    except KeyError:
        print("Configuration '{}' not found, pls refer to configs/{}.py".format(input_config, input_method))
        print("Possible configurations for '{}' are: {}".format(input_method, list(cfgfile.custom_configs.keys())))
        exit()
        
    # get attacker class
    attackfile = importlib.import_module("attack")   #import [input_method].attack
    attack_class = getattr(attackfile, config.attack_class)
    attacker = attack_class(config, input_args)
    
    return attacker

def main():
    attacker = create_attacker(sys.argv)
    print(attacker.__dict__)
    print(attacker.config.__dict__)
    attacker.attack()
    
    
if __name__=='__main__':
    main()
