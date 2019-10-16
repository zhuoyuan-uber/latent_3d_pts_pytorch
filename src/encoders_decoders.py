'''
Created on February 4, 2017

@author: optas

'''

# import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
# import numpy as np
import warnings

from .torch_utils import replicate_parameter_for_all_layers

"""
from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout

from . tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers
"""

class conv1d_op(nn.Module):
    def __init__(self, nb_in, nb_out, n_filter, stride=1, b_norm=True, non_linear=F.relu):
        super().__init__()
        self.conv = torch.nn.Conv1d(nb_in, nb_out, n_filter, stride=stride)
        self.bn = nn.BatchNorm1d(nb_out) if b_norm else None
        self.act = non_linear

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)


class linear_op(nn.Module):
    def __init__(self, nb_in, nb_out, init=None, b_norm=False, non_linear=None):
        super().__init__()
        self.linear = nn.Linear(nb_in, nb_out)
        if init == 'xavier':
            nn.init.xavier_uniform_(self.linear.weight)
        self.bn = nn.BatchNorm1d(nb_out) if b_norm else None
        self.act = non_linear

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            return self.act(x)
        else:
            return x


class encoder_with_convs_and_symmetry(nn.Module):
    def __init__(self, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                       b_norm=True, non_linearity=F.relu, regularizer=None, weight_decay=0.001,
                       symmetry='max', dropout_prob=None, padding='same', verbose=False, closing=None):
        super().__init__()

        if verbose:
            print('Building Encoder')

        n_layers = len(n_filters)
        filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
        strides = replicate_parameter_for_all_layers(strides, n_layers)
        dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

        if n_layers < 2:
            raise ValueError('More than 1 layers are expected.')

        layers = []
        for i in range(n_layers):
            nb_in = 3 if i == 0 else n_filters[i-1]
            nb_out = n_filters[i]
            #TODO: how to set weight decay? regularizer?
            layers.append(conv1d_op(nb_in, nb_out, filter_sizes[i], strides[i], b_norm, non_linearity))

        self.encoder = nn.Sequential(*layers)
        self.symmetry = symmetry
        self.closing = closing

    def forward(self, x):
        x = self.encoder(x)
        if self.symmetry == 'max':
            x, _ = torch.max(x, dim=2)
        if self.closing:
            x = self.closing(x)
        return x


class decoder_with_fc_only(nn.Module):
    def __init__(self, in_size, layer_sizes=[], b_norm=True, non_linearity=F.relu,
                       regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                       b_norm_finish=False, verbose=False):
        super().__init__()

        if verbose:
            print('Building Decoder')

        n_layers = len(layer_sizes)
        dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

        if n_layers < 2:
            raise ValueError('For an FC decoder with single a layer use simpler code.')

        layers = []
        for i in range(0, n_layers - 1):
            nb_in = in_size if i == 0 else layer_sizes[i-1]
            nb_out = layer_sizes[i]
            #TODO: how to set weight decay? regularizer?
            layers.append(linear_op(nb_in, nb_out, 'xavier', b_norm, non_linearity))

            if dropout_prob is not None and dropout_prob[i] > 0:
                layer = dropout(layer, 1.0 - dropout_prob[i])
        
        # Last decoding layer never has a non-linearity.
        #TODO: how to set weight init as Xavier? weight decay? regularizer?
        self.last_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        nn.init.xavier_uniform_(self.last_layer.weight)

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder(x)
        return self.last_layer(x)            


if __name__ == "__main__":
    model_conv = conv1d_op(3, 64, 1)
    tmp_input = torch.randn(4, 3, 2048)
    out = model_conv(tmp_input)
    print(out.shape)

    model_enc = encoder_with_convs_and_symmetry([64,128,128,256,128])
    out2 = model_enc(tmp_input)
    print(out2.shape)

    model_linear = linear_op(128, 64)
    tmp_input = torch.randn(32, 128)
    out = model_linear(tmp_input)
    print(out.shape)

    model_dec = decoder_with_fc_only(128, [256, 256, 6144], b_norm=False)
    out = model_dec(tmp_input)
    print(out.shape)

    print(model_dec.training)

"""
def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    

    

def decoder_with_convs_only(in_signal, n_filters, filter_sizes, strides, padding='same', b_norm=True, non_linearity=tf.nn.relu,
                            conv_op=conv_1d, regularizer=None, weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False):

    if verbose:
        print('Building Decoder')

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    for i in range(n_layers):
        if i == 0:
            layer = in_signal

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], padding=padding, regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i)

        if verbose:
            print(name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None and i < n_layers - 1:  # Last layer doesn't have a non-linearity.
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])

        if verbose:
            print(layer)
            print('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    return layer
"""
