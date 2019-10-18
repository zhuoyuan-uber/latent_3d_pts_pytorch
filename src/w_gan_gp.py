'''
Created on May 22, 2018

Author: Achlioptas Panos (Github ID: optas)
'''

import numpy as np
import time
import torch
import torch.autograd as autograd
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

        self.lam = lam

        self.noise_dim = noise_dim # generator input size
        self.n_output = n_output

        self.netD = discriminator(**disc_kwargs)
        self.netG = generator(noise_dim, self.n_output, **gen_kwargs) # self.noise,

        # Compute WGAN losses
        # self.loss_d = tf.reduce_mean(self.synthetic_logit) - tf.reduce_mean(self.real_logit)
        # self.loss_g = -tf.reduce_mean(self.synthetic_logit) 

        # randomly combine real-pc and generated out -> interpolates
        # gradients loss on interpolates

    def single_epoch_train(self):
        pass

    def sample_noise(self, batch_size, noise_params):
        z = torch.randn(batch_size, noise_params['noise_dim'])
        z = z * noise_params['sigma'] + noise_params['mu']
        return z

    def loss_D(self, data, z):
        # data is real data

        batch_size = data.shape[0]

        #TODO: check netD can have gradient
        fake_data = self.netG(z).detach()

        _, real_logit = self.netD(data)
        _, synth_logit = self.netD(fake_data)

        loss_D = torch.mean(synth_logit) - torch.mean(real_logit)

        # gradient penalty
        alpha = torch.rand(batch_size, 1, 1) # random uniform
        alpha = alpha.expand(data.size())
        # if torch.cuda.is_available():
        #     alpha = alpha.cuda()
        interpolates = (1.-alpha) * data + alpha * fake_data
        interpolates.requires_grad = True
        _, d_int_logit = self.netD(interpolates)

        # get gradient of netD only
        gradients = autograd.grad(outputs=d_int_logit, inputs=interpolates,
                              grad_outputs=torch.ones(d_int_logit.size()).cuda() if False else torch.ones(
                                  d_int_logit.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return loss_D + gradient_penalty * self.lam


    def loss_G(self, z):
        fake_data = model.netG(z)
        _, synth_logit = model.netD(fake_data)
        loss_G = -torch.mean(synth_logit)
        return loss_G
