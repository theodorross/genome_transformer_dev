import tensorflow as tf
import numpy as np
# from keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add
from keras import layers
import pandas as pd


@tf.keras.saving.register_keras_serializable()
class TransformerEncoderBlock(layers.Layer):
    '''
    Transformer model for whole genome assemblies.
    '''
    def __init__(self, output_dim:int, 
                       ff_dim:int, 
                       num_heads:int, 
                       key_dim:int, 
                       dropout_rate:float=0.1, 
                       activation="relu",
                       **kwargs):
        super().__init__(**kwargs)
        
        ## Define needed paramters
        self.output_dim = output_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.supports_masking = True

        ## Attention layer
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)

        ## Normalization layers
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        ## Addition layer
        self.add = layers.Add()

        ## Feed forward layers
        self.ff1 = layers.Dense(ff_dim, activation=activation)
        self.ff2 = layers.Dense(output_dim)
        if dropout_rate != 0:
            self.dropout = layers.Dropout(rate=dropout_rate)


    def call(self, query, value, key=None, attention_mask=None, use_causal_mask=False, **kwargs):
        ## Multi-head attention
        attn_output,attn_score = self.attn(query, value, key, 
                                           attention_mask=attention_mask, 
                                           use_causal_mask=use_causal_mask, 
                                           return_attention_scores=True,
                                           training=kwargs["training"])
        self.last_attention = attn_score

        ## Add and norm
        out1 = self.norm1( self.add([query, attn_output]) )

        ## Feed-forward
        ffn_output = self.ff1(out1)
        if self.dropout_rate != 0:
            ffn_output = self.dropout(ffn_output)
        ffn_output = self.ff2(ffn_output)

        ## Add and norm
        out2 = self.norm2( self.add([out1, ffn_output]) )
        
        return out2
    
    def compute_mask(self, inputs, mask=None):
        return super().compute_mask(inputs, mask)
    
    def compute_output_shape(self, input_shape):
        # b,s,_ = input_shape
        # return (b, s, self.output_dim)
        return input_shape
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "output_dim":self.output_dim,
            "ff_dim":self.ff_dim,
            "num_heads":self.num_heads,
            "key_dim":self.key_dim,
            "dropout_rate":self.dropout_rate,
            "activation":self.activation
        }
        return {**base_config, **config}





@tf.keras.saving.register_keras_serializable()
class TransformerDecoderBlock(layers.Layer):
    '''
    Transformer model for whole genome assemblies.
    '''
    def __init__(self, output_dim:int, 
                       ff_dim:int, 
                       num_heads:int, 
                       key_dim:int, 
                       dropout_rate:float=0.1, 
                       activation="relu",
                       **kwargs):
        super().__init__(**kwargs)
        
        ## Define needed paramters
        self.output_dim = output_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.supports_masking = True

        ## Attention layer
        self.self_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)
        self.cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)

        ## Normalization layers
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

        ## Addition layer
        self.add = layers.Add()

        ## Feed forward layers
        self.ff1 = layers.Dense(ff_dim, activation=activation)
        self.ff2 = layers.Dense(output_dim)
        if dropout_rate != 0:
            self.dropout = layers.Dropout(rate=dropout_rate)


    def compute_mask(self, inputs, mask=None):
        return super().compute_mask(inputs, mask)

    def call(self, query, value, key=None, attention_mask=None, use_causal_mask=False, **kwargs):
        ## Multi-head self-attention
        self_attn_output,self_attn_score = self.self_attn(query=query, value=query, key=query, 
                                                          attention_mask=attention_mask, 
                                                          use_causal_mask=use_causal_mask, 
                                                          return_attention_scores=True,
                                                          **kwargs)
        self.last_self_attention = self_attn_score

        ## Add and norm
        out1 = self.norm1( self.add([query, self_attn_output]) )

        ## Multi-head cross-attention
        cross_attn_output,cross_attn_score = self.cross_attn(query=out1, value=value, key=key,
                                                             return_attention_scores=True,
                                                             **kwargs)
        self.last_cross_attention = cross_attn_score

        ## Add and norm
        out2 = self.norm2( self.add([out1, cross_attn_output]) )

        ## Feed-forward
        ffn_output = self.ff1(out2)
        if self.dropout_rate != 0:
            ffn_output = self.dropout(ffn_output)
        ffn_output = self.ff2(ffn_output)

        ## Add and norm
        out3 = self.norm3( self.add([out2, ffn_output]) )
        return out3
    
    def compute_output_shape(self, input_shape):
        # b,s,_ = input_shape
        # return (b, s, self.output_dim)
        return input_shape
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "output_dim":self.output_dim,
            "ff_dim":self.ff_dim,
            "num_heads":self.num_heads,
            "key_dim":self.key_dim,
            "dropout_rate":self.dropout_rate,
            "activation":self.activation
        }
        return {**base_config, **config}
