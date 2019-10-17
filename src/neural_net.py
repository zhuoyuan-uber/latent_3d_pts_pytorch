'''
Created on August 28, 2017

@author: optas
'''

import os.path as osp
import torch
from torch import nn


MODEL_SAVER_ID = 'models.ckpt'


class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def is_training(self):
        return self.training

    def restore_model(self, PATH):
        self.load_state_dict(torch.load(PATH))

