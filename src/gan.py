'''
Created on May 3, 2017

@author: optas
'''

import os.path as osp
import torch
from torch import nn
# import warnings

# from . neural_net import Neural_Net
# from . tf_utils import safe_log

class GAN(nn.Module):
    def __init__(self):
        super().__init__()

    def save_model(self):
        pass

    def restore_model(self, model_path, epoch, verbose=False):
        pass

    def optimizer(self):
        pass

    def generate(self, n_samples, noise_params):
        noise = self.generator_noise_distribution(n_samples, self.noise_dim, **noise_params)
        # forward
        # return generate_out

    def vanilla_gan_objective(self, real_prob, synthetic_prob, use_safe_log=True):
        if use_safe_log:
            log = safe_log # log(max(x, eps)) with eps=1e-12
        else:
            log = torch.log

        loss_d = torch.mean(-log(real_prob) - log(1 - synthetic_prob))
        loss_g = torch.mean(-log(synthetic_prob))
        return loss_d, loss_g

    def w_gan_objective(self, real_logit, synthetic_logit):
        loss_d = torch.mean(synthetic_logit) - torch.mean(real_logit)
        loss_g = -torch.mean(synthetic_logit)
        return loss_d, loss_g
