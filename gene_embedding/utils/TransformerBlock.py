import tensorflow as tf
import numpy as np
from keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization, Dropout
import pandas as pd


class TransformerBlock(Layer):
    '''
    Transformer model for whole genome assemblies.
    '''
    def __init__(self, output_dim:int, ff_dim:int, num_heads:int, key_dim:int, dropout_rate:float=0.1, **kwargs):
        super().__init__(**kwargs)
        
        ## Define needed paramters
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim

        ## Attention layer
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        ## Normalization layers
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

        ## Feed forward layers
        self.ff1 = Dense(ff_dim, activation="leaky_relu")
        self.ff2 = Dense(output_dim)

        ## Dropout layers
        if dropout_rate != 0:
            self.dropout1 = Dropout(rate=dropout_rate)
            self.dropout2 = Dropout(rate=dropout_rate)


    def call(self, inputs):
        ## Multi-head attention
        attn_output = self.attn(inputs, inputs)
        if self.dropout_rate != 0:
            attn_output = self.dropout1(attn_output)

        ## Add and norm
        # print(inputs.shape, attn_output.shape)
        out1 = self.norm1(inputs + attn_output)
        # print(out1.shape)

        ## Feed-forward
        ffn_output = self.ff1(out1)
        if self.dropout_rate != 0:
            ffn_output = self.dropout2(ffn_output)
        ffn_output = self.ff2(ffn_output)
        # print(out1.shape, ffn_output.shape)
        out2 = self.norm2(out1 + ffn_output)
        # print(out2.shape)
        
        return out2

    def build(self, input_shape):
        return
    
    def compute_output_shape(self, input_shape):
        b,s,_ = input_shape
        return (b, s, self.ff_dim)
    


