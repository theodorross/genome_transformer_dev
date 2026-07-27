import tensorflow as tf
from keras import layers
from keras import models

# from utils.PositionalEmbedding import PositionalEmbedding
from utils.RotaryPositionEmbedding import RotaryPositionEncoding
from utils.TransformerBlock import TransformerEncoderBlock, TransformerDecoderBlock
from utils.DNATokenizer import DNATokenizer

# from matplotlib import pyplot as plt



@tf.keras.utils.register_keras_serializable()
class SequenceEncoder(models.Model):


    def __init__(self, tokenization_method:str,
                       encoder_layers:int, 
                       embedding_dim:int,
                       latent_dim:int,
                       key_dim:int,
                       num_heads:int,
                       dropout_rate:float,
                       ff_dim:int,
                       masking_rate:float,
                       n_sequence_tokens:int=1,
                       max_length:int=5000,
                       decode_length:int=5000,
                       hypersphere_norm:bool=False,
                       **kwargs):
        super(SequenceEncoder, self).__init__(**kwargs)

        self.tokenization_method = tokenization_method
        self.encoder_layers = encoder_layers
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.masking_rate = masking_rate
        self.n_sequence_tokens = n_sequence_tokens
        self.max_length = max_length
        self.decode_length = decode_length
        self.hypersphere_norm = hypersphere_norm


        ## Define the tokenizing and positional embedding
        self.tokenizing_layer = DNATokenizer(tokenization_method=self.tokenization_method,
                                             max_length=self.decode_length)
        self.vocabulary = self.tokenizing_layer.tokenizing_layer.get_vocabulary()
        self.vocab_size = len(self.vocabulary)

        self.token_masker = layers.Dropout(rate=self.masking_rate)
        self.embedding_layer = layers.Embedding(input_dim=self.vocab_size, 
                                                output_dim=self.embedding_dim, 
                                                mask_zero=True)
        self.position_encoder = RotaryPositionEncoding(self.max_length, self.embedding_dim)
        self.latent_position_encoder = RotaryPositionEncoding(self.n_sequence_tokens, self.latent_dim)

        ## Define the transformer block layers
        self.transformer_layers = [
            TransformerEncoderBlock(output_dim=self.latent_dim, ff_dim=self.ff_dim, num_heads=self.num_heads, 
                                    key_dim=self.key_dim, dropout_rate=self.dropout_rate)
        ] + [
            TransformerDecoderBlock(output_dim=self.latent_dim, ff_dim=self.ff_dim, num_heads=self.num_heads, 
                                    key_dim=self.key_dim, dropout_rate=self.dropout_rate)
            for _ in range(self.encoder_layers-1)
        ]
        # self.transformer_layers = [
        #     TransformerEncoderBlock(output_dim=self.latent_dim, ff_dim=self.ff_dim, num_heads=self.num_heads, 
        #                             key_dim=self.key_dim, dropout_rate=self.dropout_rate)
        #     for _ in range(self.encoder_layers)
        # ]

        ## Define the flattening layer
        self.flatten = layers.Flatten()

        ## Define linear projectino layer
        self.projector = layers.Dense(latent_dim, activation=None, use_bias=True)

        ## Define the querry token sequence
        self.latent_tokens = self.add_weight(
            name="latent_tokens",
            shape=(1,1,self.latent_dim),
            initializer="uniform",
            trainable=True
        )

        ## Run a dummy input through the model
        dummy_in = tf.convert_to_tensor([["atgatgatg"]])
        self(dummy_in)


    def call(self, x, **kwargs):
        x = tf.cast(x, tf.string)
        ## Convert the input strings to tokens
        tokens = self.tokenizing_layer(x, **kwargs)
        float_tokens = tf.cast(tokens, "float32")
        ## Randomly mask the tokens for training
        masked_tokens = self.token_masker(float_tokens, **kwargs)
        # Undo the Dropout layer's normalization if needed
        if kwargs.get('training', False): 
            masked_tokens *= (1-self.masking_rate)
        ## Compute positional embeddings
        # enc_z0 = self.embedding_layer(float_tokens)
        enc_z0 = self.embedding_layer(masked_tokens)
        enc_z = self.position_encoder(enc_z0)

        ## Repeat the latent sequence along the batch dimension
        latent_seq = tf.tile(self.latent_tokens, (tf.shape(x)[0], self.n_sequence_tokens,1))
        latent_seq = self.latent_position_encoder(latent_seq)

        ## Pass through the transformer blocks
        for ix,enc_layer in enumerate(self.transformer_layers):
            if ix == 0:
                latent_seq = enc_layer(query=latent_seq, value=enc_z, use_residuals=True, **kwargs)
            else:
                latent_seq = enc_layer(query=latent_seq, value=latent_seq, use_residuals=True, **kwargs)

        ## Flatten the embedding into a single vector and project it to final dimensions
        latent_vec = self.flatten(latent_seq)
        latent_vec = self.projector(latent_vec)

        ## Optional hypersphere normalization
        if self.hypersphere_norm:
            latent_vec = tf.math.l2_normalize(latent_vec, axis=-1)
        return latent_vec
    
    
    def _update_decode_length(self, new_length):
        self.decode_length = new_length
        self.tokenizing_layer.max_length.assign(new_length)


    def build(self, input_shape):
        super().build(input_shape)


    def get_config(self):
        base_config = super().get_config()
        config = {
            'tokenization_method' : self.tokenization_method,
            'encoder_layers' : self.encoder_layers,
            'embedding_dim' : self.embedding_dim,
            'latent_dim' : self.latent_dim,
            'key_dim' : self.key_dim,
            'num_heads' : self.num_heads,
            'dropout_rate' : self.dropout_rate,
            'ff_dim' : self.ff_dim,
            'masking_rate' : self. masking_rate,
            'n_sequence_tokens' : self.n_sequence_tokens,
            'max_length' : self.max_length,
            'decode_length' : self.decode_length,
            'hypersphere_norm' : self.hypersphere_norm
        }
        return {**base_config, **config}
        

