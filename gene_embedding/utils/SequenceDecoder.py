import tensorflow as tf
from keras.layers import Dense, Identity
from keras import models

from utils.PositionalEmbedding import PositionalEmbedding
from utils.TransformerBlock import TransformerBlock
from utils.DNATokenizer import DNATokenizer



class SequenceDecoder(models.Model):

    def __init__(self, decoder_layers:int, 
                       embedding_dim:int,
                       key_dim:int,
                       num_heads:int,
                       dropout_rate:float,
                       ff_dim:int,
                       vocab_size:int,
                       **kwargs):
        super().__init__(**kwargs)

        self.decoder_layers = decoder_layers
        self.embedding_dim = embedding_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size

        ## Define the transformer block layers
        self.transformer_layers = [
            TransformerBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads, 
                             key_dim=key_dim, dropout_rate=dropout_rate) 
            for _ in range(decoder_layers)
        ]

        self.final_layer = Dense(vocab_size, activation="softmax")


    def call(self, x, attention_mask=None):
        ## First layer with potential attention mask
        if attention_mask is not None:
            causal = True
        else:
            causal = False
        z = self.transformer_layers[0](x, attention_mask=attention_mask, use_causal_mask=False)
        ## Subsequent layers with no attention mask
        if self.decoder_layers > 1:
            for dec_layer in self.transformer_layers[1:]:
                z = dec_layer(z)
        return self.final_layer(z)
    

    def get_config(self):
        base_config = super().get_config()
        config = {
            'decoder_layers' : self.decoder_layers,
            'embedding_dim' : self.embedding_dim,
            'key_dim' : self.key_dim,
            'num_heads' : self.num_heads,
            'dropout_rate' : self.dropout_rate,
            'ff_dim' : self.ff_dim,
            'vocab_size' : self.vocab_size
        }
        return {**base_config, **config}