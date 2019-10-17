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


class PointNetAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''
    def __init__(self, args):
        AutoEncoder.__init__(self, args)
        self.args = args

        self.enc = args.encoder(**args.encoder_args)
        self.bottleneck_size = args.encoder_args['n_filters'][-1]
        self.dec = args.decoder(**args.decoder_args)

        self.cd_loss = ChamferDistance()

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)

        if self.args.exists_and_is_not_none('close_with_tanh'):
            out = nn.Tanh(out)

        x_reconstr = out.view(-1, self.n_output[0], self.n_output[1])
        return x_reconstr

    def loss(self, x_reconstr, gt):
        # x_reconstr and gt are of shape (?, 3, 2048), should transpose first
        x_reconstr = x_reconstr.permute(0, 2, 1).contiguous()
        gt = gt.permute(0, 2, 1).contiguous()
        if self.args.loss == 'chamfer':
            cost_p1_p2, cost_p2_p1 = self.cd_loss(x_reconstr, gt)
            loss = torch.mean(cost_p1_p2) + torch.mean(cost_p2_p1)
        elif self.args.loss == 'emd':
            raise NotImplementedError("emd is not implemented in pytorch")

        #TODO: add regularization loss?
        return loss

    def _setup_optimizer(self):
        # in pytorch, optimizer is supposed to be defined outside
        # set it in main loop
        pass
        