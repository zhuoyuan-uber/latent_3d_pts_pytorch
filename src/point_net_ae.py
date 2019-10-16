'''
Created on January 26, 2017

@author: optas
'''

import time
import os.path as osp
import torch
from torch import nn

from . in_out import create_dir
from . autoencoder import AutoEncoder
from . general_utils import apply_augmentations

import sys
sys.path.append('..')

from chamfer_distance import ChamferDistance
"""
from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost
"""

class PointNetAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''
    def __init__(self, name, configuration, graph=None):
        AutoEncoder.__init__(self, configuration)
        c = configuration
        self.configuration = c

        self.enc = c.encoder(**c.encoder_args)
        self.bottleneck_size = 128
        self.dec = c.decoder(**c.decoder_args)

        self.cd_loss = ChamferDistance()

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)

        if self.configuration.exists_and_is_not_none('close_with_tanh'):
            out = nn.Tanh(out)

        x_reconstr = out.view(-1, self.n_output[0], self.n_output[1])
        return x_reconstr

    def loss(self, x_reconstr, gt):
        # p1: x_
        c = self.configuration

        if c.loss == 'chamfer':
            cost_p1_p2, cost_p2_p1 = self.cd_loss(x_reconstr, gt)
            loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            raise NotImplementedError("emd is not implemented in pytorch")

        #TODO: add regularization loss?
        return loss

    def _setup_optimizer(self):
        # in pytorch, optimizer is supposed to be defined outside
        # set it in main loop
        pass

    def _single_epoch_train(self):
        # supposed to be in main loop
        pass
        