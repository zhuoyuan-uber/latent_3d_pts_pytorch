'''
Created on May 22, 2018

Author: Achlioptas Panos (Github ID: optas)
'''

import numpy as np
import time
# import tensorflow as tf

# from tflearn import is_training
from . gan import GAN

class W_GAN_GP(GAN):
    '''Gradient Penalty.
    https://arxiv.org/abs/1704.00028
    '''
    def __init__(self, lam, n_output, noise_dim, discriminator, generator, gen_kwargs={}, disc_kwargs={}):
        # lam: lambda for gradient-penalty regularizer
        # n_output: [3, 2048]
        # noise_dim: latent code size (default: 128)
        super().__init__()

        self.noise_dim = noise_dim # generator input size
        self.n_output = n_output

        self.discriminator = discriminator(**disc_kwargs)
        self.generator = generator(noise_dim, self.n_output, **gen_kwargs) # self.noise,

        # Compute WGAN losses
        # self.loss_d = tf.reduce_mean(self.synthetic_logit) - tf.reduce_mean(self.real_logit)
        # self.loss_g = -tf.reduce_mean(self.synthetic_logit) 

        # randomly combine real-pc and generated out -> interpolates
        # gradients loss on interpolates

    def single_epoch_train(self):
        pass
