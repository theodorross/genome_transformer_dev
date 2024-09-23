import tensorflow as tf
from keras import layers
from keras import models

from utils.PositionalEmbedding import PositionalEmbedding
from utils.TransformerBlock import TransformerBlock
from utils.DNATokenizer import DNATokenizer



@tf.keras.saving.register_keras_serializable()
class SequenceEncoder(models.Model):

    def __init__(self, tokenization_method:str,
                       encoder_layers:int, 
                       embedding_dim:int,
                       key_dim:int,
                       num_heads:int,
                       dropout_rate:float,
                       ff_dim:int,
                       masking_rate:float,
                       sequence_token=None,
                       max_length:int=5000,
                       **kwargs):
        super().__init__(**kwargs)

        self.tokenization_method = tokenization_method
        self.encoder_layers = encoder_layers
        self.embedding_dim = embedding_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.masking_rate = masking_rate
        self.sequence_token = sequence_token
        self.max_length = max_length

        ## Define the tokenizing and positional embedding
        self.tokenizing_layer = DNATokenizer(tokenization_method=tokenization_method,
                                             sequence_token=sequence_token)
        self.vocabulary = self.tokenizing_layer.tokenizing_layer.get_vocabulary()
        self.vocab_size = len(self.vocabulary)

        self.token_masker = layers.Dropout(rate=masking_rate)
        self.embedding_layer = PositionalEmbedding(self.vocab_size, embedding_dim, max_length=max_length)

        ## Define the transformer block layers
        self.transformer_layers = [
            TransformerBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads, 
                             key_dim=key_dim, dropout_rate=dropout_rate) 
            for _ in range(encoder_layers)
        ]

        # ## Define the optional pooling layer
        # self.pooling_layer = layers.GlobalAveragePooling1D()


    def call(self, x):
        ## Convert the input strings to tokens
        tokens = self.tokenizing_layer(x)
        float_tokens = tf.cast(tokens, "float32")
        ## Randomly mask the tokens for training
        masked_tokens = self.token_masker(float_tokens)
        ## Compute positional embeddings
        enc_z = self.embedding_layer(masked_tokens)
        ## Pass through transformer layers
        for enc_layer in self.transformer_layers:
            enc_z = enc_layer(enc_z)
        return enc_z
    

    def get_config(self):
        base_config = super().get_config()
        config = {
            'tokenization_method' : self.tokenization_method,
            'encoder_layers' : self.encoder_layers,
            'embedding_dim' : self.embedding_dim,
            'key_dim' : self.key_dim,
            'num_heads' : self.num_heads,
            'dropout_rate' : self.dropout_rate,
            'ff_dim' : self.ff_dim,
            'masking_rate' : self. masking_rate,
            'sequence_token' : self.sequence_token,
            'max_length' : self.max_length
        }
        return {**base_config, **config}
        

