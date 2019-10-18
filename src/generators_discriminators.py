'''
Created on May 11, 2017

@author: optas
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# import tensorflow as tf
# from tflearn.layers.normalization import batch_normalization
# from tflearn.layers.core import fully_connected, dropout

from .encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only
# from . tf_utils import leaky_relu
# from . tf_utils import expand_scope_by_name


class mlp_discriminator(nn.Module):
    def __init__(self, non_linearity=F.relu, b_norm=True, dropout_prob=None):
        super().__init__()
        encoder_args = {'n_filters': [64, 128, 256, 256, 512], 'filter_sizes': [1, 1, 1, 1, 1], 'strides': [1, 1, 1, 1, 1]}
        encoder_args['non_linearity'] = non_linearity
        encoder_args['dropout_prob'] = dropout_prob
        encoder_args['b_norm'] = b_norm
        self.conv_base = encoder_with_convs_and_symmetry(**encoder_args)
        self.dis = decoder_with_fc_only(encoder_args['n_filters'][-1], layer_sizes=[128, 64, 1], b_norm=b_norm)

    def forward(self, x):
        x = self.conv_base(x)
        d_logit = self.dis(x) # (?, 1)
        # d_prob = torch.sigmoid(d_logit) # (?, 1)
        # return d_prob, d_logit
        return d_logit


class point_cloud_generator(nn.Module):
    def __init__(self, in_size, pc_dims, layer_sizes=[64, 128, 512, 1024],
                 non_linearity=F.relu, b_norm=False, b_norm_last=False, dropout_prob=None):
        super().__init__()
        dummy, n_points = pc_dims
        if dummy != 3:
            raise ValueError("point clouds should have 3 coordinates.")

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob else None
        self.non_linearity = non_linearity
        self.batchnorm = nn.BatchNorm1d(layer_sizes[-1]) if b_norm_last else None

        self.dec = decoder_with_fc_only(in_size=in_size, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm)
        self.last_layer = nn.Linear(layer_sizes[-1], n_points * 3)
        nn.init.xavier_uniform_(self.last_layer.weight)
    
    def forward(self, x):
        x = self.non_linearity(self.dec(x)) # (?, 1024)
        if self.dropout:
            x = self.dropout(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        x = self.last_layer(x)
        x = x.view(x.shape[0], 3, -1)
        return x


if __name__ == "__main__":
    noise = torch.randn(20, 128)
    generator = point_cloud_generator(128, [3, 2048], [64, 128, 512, 1024])
    out = generator(noise)
    print(out.shape)

    pts = torch.randn(20, 3, 2048)
    discriminator = mlp_discriminator(b_norm=False)
    out1, out2 = discriminator(pts)
    print(out1.shape, out2.shape)


"""
def convolutional_discriminator(in_signal, non_linearity=tf.nn.relu,
                                encoder_args={'n_filters': [128, 128, 256, 512], 'filter_sizes': [40, 20, 10, 10], 'strides': [1, 2, 2, 1]},
                                decoder_layer_sizes=[128, 64, 1],
                                reuse=False, scope=None):

    encoder_args['reuse'] = reuse
    encoder_args['scope'] = scope
    encoder_args['non_linearity'] = non_linearity
    layer = encoder_with_convs_and_symmetry(in_signal, **encoder_args)

    name = 'decoding_logits'
    scope_e = expand_scope_by_name(scope, name)
    d_logit = decoder_with_fc_only(layer, layer_sizes=decoder_layer_sizes, non_linearity=non_linearity, reuse=reuse, scope=scope_e)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def latent_code_generator(z, out_dim, layer_sizes=[64, 128], b_norm=False):
    layer_sizes = layer_sizes + out_dim
    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, b_norm=b_norm)
    out_signal = tf.nn.relu(out_signal)
    return out_signal


def latent_code_discriminator(in_singnal, layer_sizes=[64, 128, 256, 256, 512], b_norm=False, non_linearity=tf.nn.relu, reuse=False, scope=None):
    layer_sizes = layer_sizes + [1]
    d_logit = decoder_with_fc_only(in_singnal, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm, reuse=reuse, scope=scope)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def latent_code_discriminator_two_layers(in_signal, layer_sizes=[256, 512], b_norm=False, non_linearity=tf.nn.relu, reuse=False, scope=None):
    ''' Used in ICML submission.
    '''
    layer_sizes = layer_sizes + [1]
    d_logit = decoder_with_fc_only(in_signal, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm, reuse=reuse, scope=scope)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def latent_code_generator_two_layers(z, out_dim, layer_sizes=[128], b_norm=False):
    ''' Used in ICML submission.
    '''
    layer_sizes = layer_sizes + out_dim
    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, b_norm=b_norm)
    out_signal = tf.nn.relu(out_signal)
    return out_signal
"""
