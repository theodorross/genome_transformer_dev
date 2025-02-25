import tensorflow as tf
import numpy as np
from keras.layers import Layer, EinsumDense
import pandas as pd




class ContigPA_LinearEncoding(Layer):
    '''
    Linear model for computing embeddings of contigs from gene pressence/absence vectors
    '''
    def __init__(self, input_dim, embedding_dim, activation=None, use_bias=False,
                 kernel_initializer="glorot_uniform", bias_initializer="zeros", 
                 **kwargs):
        super().__init__(**kwargs)
        
        self.xdim = input_dim
        self.zdim = embedding_dim

        # self.input = Input(shape=[input_dim])
        # self.dense = Dense(embedding_dim, input_dim=input_dim, activation=activation, use_bias=use_bias,
        #                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        if use_bias: 
            bias_axes="z"
        else: 
            bias_axes=None
        self.dense = EinsumDense("bsx,xz->bsz", output_shape=(None,embedding_dim), activation=activation,
                                 bias_axes=bias_axes, kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer)

    def call(self, x):
        return(self.dense(x))





