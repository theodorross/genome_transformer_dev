import tensorflow as tf
import numpy as np
# from keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add
from keras import layers
import pandas as pd
# from matplotlib import pyplot as plt


@tf.keras.utils.register_keras_serializable()
class TransformerDecoderBlock(layers.Layer):
    '''
    Transformer model for whole genome assemblies.
    '''
    def __init__(self, output_dim:int, 
                       ff_dim:int, 
                       num_heads:int, 
                       key_dim:int, 
                       dropout_rate:float=0.1, 
                       activation="gelu",
                       **kwargs):
        super().__init__(**kwargs)
        
        ## Define needed paramters
        self.output_dim = output_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        # self.supports_masking = True

        ## Attention layer
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)

        ## Normalization layers
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        ## Addition layer
        self.add1 = layers.Add()
        self.add2 = layers.Add()

        ## Feed forward layers
        self.ff1 = layers.Dense(ff_dim, activation=activation)
        self.ff2 = layers.Dense(output_dim)
        if dropout_rate != 0:
            self.dropout = layers.Dropout(rate=dropout_rate)


    def call(self, query, value, key=None, attention_mask=None, use_causal_mask=False, use_residuals=True, **kwargs):
        ## Multi-head attention
        attn_output,attn_score = self.attn(query=query, value=value, key=key, 
                                           attention_mask=attention_mask, 
                                           use_causal_mask=use_causal_mask, 
                                           return_attention_scores=True,
                                           training=kwargs["training"])
        self.last_attention = attn_score

        ## Add and norm
        if use_residuals:
            out1 = self.norm1( self.add1([query, attn_output]) )
            # out1 = self.add([query, attn_output])
        else:
            out1 = self.norm1( attn_output )
            # out1 = attn_output

        ## Feed-forward
        ffn_output = self.ff1(out1)
        if self.dropout_rate != 0:
            ffn_output = self.dropout(ffn_output)
        ffn_output = self.ff2(ffn_output)

        ## Add and norm
        if use_residuals:
            out2 = self.norm2( self.add2([out1, ffn_output]) )
            # out2 = self.add([out1, ffn_output])
        else:
            out2 = self.norm2( ffn_output )
            # out2 = ffn_output
        
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





@tf.keras.utils.register_keras_serializable()
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
                    #    value_dim=None,
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
        # self.value_dim = value_dim

        ## Attention layer
        self.cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)
        self.self_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)

        ## Normalization layers
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

        ## Addition layers
        self.add1 = layers.Add()
        self.add2 = layers.Add()
        self.add3 = layers.Add()

        ## Feed forward layers
        self.ff1 = layers.Dense(ff_dim, activation=activation)
        self.ff2 = layers.Dense(output_dim)
        if dropout_rate != 0:
            self.dropout = layers.Dropout(rate=dropout_rate)


    def compute_mask(self, inputs, mask=None):
        return super().compute_mask(inputs, mask)

    def call(self, query, value, key=None, attention_mask=None, use_causal_mask=False, use_residuals=True, **kwargs):

        ## Multi-head cross-attention
        cross_attn_output,cross_attn_score = self.cross_attn(query=query, value=value, key=key,
                                                             return_attention_scores=True,
                                                             **kwargs)
        self.last_cross_attention = cross_attn_score

        ## Add and norm
        if use_residuals:
            out1 = self.norm1( self.add1([query, cross_attn_output]) )
        else:
            out1 = self.norm1( cross_attn_output )

        ## Multi-head self-attention
        self_attn_output,self_attn_score = self.self_attn(query=out1, value=out1, key=out1, 
                                                          attention_mask=attention_mask, 
                                                          use_causal_mask=use_causal_mask, 
                                                          return_attention_scores=True,
                                                          **kwargs)
        self.last_self_attention = self_attn_score

        ## Add and norm
        if use_residuals:
            out2 = self.norm2( self.add2([out1, self_attn_output]) )
        else:
            out2 = self.norm2( self_attn_output )

        ## Feed-forward
        ffn_output = self.ff1(out2)
        if self.dropout_rate != 0:
            ffn_output = self.dropout(ffn_output)
        ffn_output = self.ff2(ffn_output)

        ## Add and norm
        if use_residuals:
            out3 = self.norm3( self.add3([out2, ffn_output]) )
        else:
            out3 = self.norm3( ffn_output )

        # try:
        #     f,a = plt.subplots(2,1)
        #     ca = cross_attn_score[0,0].numpy()
        #     ca[ca==0] = np.nan
        #     sa = self_attn_score[0,0].numpy()
        #     sa[sa==0] = np.nan
        #     a[0].imshow(ca)
        #     a[1].imshow(sa)
        #     a[0].set_title("cross_attn_score")
        #     a[1].set_title("self_attn_score")
        #     # f,a = plt.subplots(4,1)
        #     # a[0].imshow(query[0].numpy())
        #     # a[1].imshow(out1[0].numpy())
        #     # a[2].imshow(out2[0].numpy())
        #     # a[3].imshow(out3[0].numpy())
        #     # a[0].set_title("query")
        #     # a[1].set_title("out1")
        #     # a[2].set_title("out2")
        #     # a[3].set_title("out3")
        #     # plt.show()
        # except:
        #     plt.close(f)
        #     pass

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
            # "value_dim":self.value_dim
        }
        return {**base_config, **config}
