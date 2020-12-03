from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import range
from builtins import object
from past.utils import old_div
from abc import ABCMeta
from abc import abstractmethod

import numpy as np
import cv2
import os
import xmltodict
import pdb
from future.utils import with_metaclass

import attack_config.attack_config
from attack_config import attack_config

import sys
import argparse

class Attack_Base(with_metaclass(ABCMeta, object)):

    def __init__(self, config, args):
        """
        Abstract base class for ODD. With some visualization tools inside.
        """
        # default 
        print("ODD Logic")
        self.load_config(config)
        self.argv_parser(args)
        self.load_attack()

        return
        self.build_attack()

    def load_config(self, config):
        # self.__dict__ = config.__dict__.copy()  
        self.config = config

    def argv_parser(self, args):
        """
        Parse given command information as attack setting. Overwrite the configuration.
        """
        if len(args) < 2:
            return
        print("Parsing additional input arguments: ")
        for i in range(0, len(args),2):
            setattr(self.config, args[i][2:], args[i+1]) 
            print(args[i][2:], ":", args[i+1])

    @abstractmethod
    def load_attack(self):
        raise NotImplementedError    

    @abstractmethod
    def build_attack(self):
        raise NotImplementedError    

    @abstractmethod
    def attack(self):
        raise NotImplementedError    

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()
