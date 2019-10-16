'''
Created on February 2, 2017

@author: optas
'''

import warnings
import os.path as osp
import numpy as np
import torch
from torch import nn
# from tflearn import is_training

from . in_out import create_dir, pickle_data, unpickle_data
from . general_utils import apply_augmentations, iterate_in_chunks
from . neural_net import Neural_Net, MODEL_SAVER_ID


class Configuration():
    def __init__(self, n_input, encoder, decoder, encoder_args={}, decoder_args={},
                 training_epochs=200, batch_size=10, learning_rate=0.001, denoising=False,
                 saver_step=None, train_dir=None, z_rotate=False, loss='chamfer', gauss_augment=None,
                 saver_max_to_keep=None, loss_display_step=1, debug=False,
                 n_z=None, n_output=None, latent_vs_recon=1.0, consistent_io=None):

        # Parameters for any AE
        self.n_input = n_input
        self.is_denoising = denoising
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

        # Training related parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.gauss_augment = gauss_augment
        self.z_rotate = z_rotate
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs
        self.debug = debug

        # Used in VAE
        self.latent_vs_recon = np.array([latent_vs_recon], dtype=np.float32)[0]
        self.n_z = n_z

        # Used in AP
        if n_output is None:
            self.n_output = n_input
        else:
            self.n_output = n_output

        self.consistent_io = consistent_io

    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    def __str__(self):
        keys = list(self.__dict__.keys())
        vals = list(self.__dict__.values())
        index = np.argsort(keys)
        res = ''
        for i in index:
            if callable(vals[i]):
                v = vals[i].__name__
            else:
                v = str(vals[i])
            res += '%30s: %s\n' % (str(keys[i]), v)
        return res

    def save(self, file_name):
        pickle_data(file_name + '.pickle', self)
        with open(file_name + '.txt', 'w') as fout:
            fout.write(self.__str__())

    @staticmethod
    def load(file_name):
        return unpickle_data(file_name + '.pickle').__next__()


class AutoEncoder(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.is_denoising = configuration.is_denoising
        self.n_input = configuration.n_input
        self.n_output = configuration.n_output

        in_shape = [None] + self.n_input
        out_shape = [None] + self.n_output

        # (self.x, self.gt) for data, label

    def partial_fit(self, X, GT=None):
        '''Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a denoising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        self.train()
        pass

    def reconstruct(self, X, GT=None):
        '''Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)'''
        pass

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        pass

    def interpolate(self, x, y, steps):
        ''' Interpolate between and x and y input vectors in latent space.
        x, y np.arrays of size (n_points, dim_embedding).
        '''

    def decode(self, z):
        if np.ndim(z) == 1:  # single example
            z = np.expand_dims(z, 0)
        
    def train(self, train_data, configuration):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in range(c.training_epochs):
            # loss, duration = self._single_epoch_train(train_data, c)
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
        return stats
    
    def evaluate(self, in_data, configuration, ret_pre_augmentation=False):
        pass
    
    def embedding_at_tensor(self, dataset, conf, feed_original=True, apply_augmentation=False, tensor_name='bottleneck'):
        '''
        Observation: the NN-neighborhoods seem more reasonable when we do not apply the augmentation.
        Observation: the next layer after latent (z) might be something interesting.
        tensor_name: e.g. model.name + '_1/decoder_fc_0/BiasAdd:0'
        '''
        pass

    def get_latent_codes(self, pclouds, batch_size=100):
        ''' Convenience wrapper of self.transform to get the latent (bottle-neck) codes for a set of input point 
        clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
        '''
        pass
