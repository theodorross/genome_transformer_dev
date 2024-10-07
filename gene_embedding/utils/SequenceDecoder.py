import tensorflow as tf
from keras.layers import Dense, Identity
from keras import models

# from utils.PositionalEmbedding import PositionalEmbedding
from utils.TransformerBlock import TransformerDecoderBlock, TransformerEncoderBlock
from utils.RotationalPositionEmbedding import RotaryPositionEncoding
# from utils.DNATokenizer import DNATokenizer


@tf.function
def positional_encoding(length, depth):
    depth = depth/2

    positions = tf.range(length, dtype=tf.float32)[:, None]     # (seq, 1)
    depths = tf.range(depth, dtype=tf.float32)[None, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = tf.concat(
        [tf.sin(angle_rads), tf.cos(angle_rads)],
        axis=-1
    ) 

    return tf.cast(pos_encoding, dtype=tf.float32)



class SequenceDecoder(models.Model):

    def __init__(self, decoder_layers:int, 
                       embedding_dim:int,
                       key_dim:int,
                       num_heads:int,
                       dropout_rate:float,
                       ff_dim:int,
                       vocab_size:int,
                       n_sequence_tokens:int=2,
                       max_length:int=5000,
                       **kwargs):
        super().__init__(**kwargs)

        self.decoder_layers = decoder_layers
        self.embedding_dim = embedding_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.n_sequence_tokens = n_sequence_tokens
        self.max_length = max_length

        ## Define the transformer block layers
        # Initial layer without self-attention for the mask sequence
        self.transformer_layers = [
            TransformerEncoderBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads,
                                    key_dim=key_dim, dropout_rate=dropout_rate)
        ]
        # Subsequent layers with self-attention and cross-attention layers
        if decoder_layers > 1:
            for _ in range(decoder_layers-1):
                self.transformer_layers.append(
                    TransformerEncoderBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads, 
                                            key_dim=key_dim, dropout_rate=dropout_rate) 
                )
        # self.transformer_layers = [
        #     TransformerDecoderBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads, 
        #                             key_dim=key_dim, dropout_rate=dropout_rate) 
        #     for _ in range(decoder_layers-1)
        # ]

        ## Define the positional encoding
        # self.pos_encoding = positional_encoding(self.max_length, self.embedding_dim)
        self.position_encoder = RotaryPositionEncoding(max_length, embedding_dim)

        ## Define the final output layer
        self.final_layer = Dense(vocab_size, activation="softmax")

        ## Define the mask token variable
        #  -> defined to allow broadcasting across the batch and sequence dimensions
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(1,1,embedding_dim),
            initializer="glorot_uniform"
        )

    def get_decoder_mask(self, seq_len):
        ## Define a mask to apply to the encoded sequences
        mask = tf.expand_dims(tf.eye(seq_len, self.n_sequence_tokens, dtype=tf.float32), axis=0)
        mask = tf.reduce_sum(mask, axis=2, keepdims=True)
        return mask


    def call(self, x, **kwargs):
        ## Create the query sequence and add positional encoding
        query_seq = tf.tile(self.mask_token, [tf.shape(x)[0], self.max_length,1])
        # query_seq = query_seq + self.pos_encoding[tf.newaxis,:,:]
        query_seq = self.position_encoder(query_seq)

        ## Pass through the first transformer layer
        z = self.transformer_layers[0](query=query_seq, value=x, key=x, **kwargs)
        ## Subsequent layers with no attention mask
        if self.decoder_layers > 1:
            for dec_layer in self.transformer_layers[1:]:
                # z = dec_layer(z, context,context, **kwargs)
                z = dec_layer(query=z, value=x, key=x, **kwargs)
        return self.final_layer(z, **kwargs)



        ## Apply the mask
        # mask = self.get_decoder_mask(tf.shape(x)[1])
        # x = mask*x + (1-mask)*self.mask_token

        ## Separate the context and sequence tokens
        # context_v = tf.slice(x, [0,0,0], [-1,self.n_sequence_tokens//2,-1])
        # context_k = tf.slice(x, [0,self.n_sequence_tokens//2,0], [-1,self.n_sequence_tokens//2,-1])
        # context = tf.slice(x, [0,0,0], [-1,self.n_sequence_tokens,-1])
        # x = tf.slice(x, [0,self.n_sequence_tokens,0], [-1,-1,-1])

        ## Add the positional encoding
        # # z = x + self.pos_encoding[tf.newaxis, :tf.shape(x)[1], :]
        # ## First layer with potential attention mask
        # # z = self.transformer_layers[0](z,context,context, **kwargs)
        # z = self.transformer_layers[0](z,z, **kwargs)
        # ## Subsequent layers with no attention mask
        # if self.decoder_layers > 1:
        #     for dec_layer in self.transformer_layers[1:]:
        #         # z = dec_layer(z, context,context, **kwargs)
        #         z = dec_layer(z, z, **kwargs)
        # z = tf.slice(z, [0,self.n_sequence_tokens,0], [-1,-1,-1])
        # return self.final_layer(z, **kwargs)
    

    def get_config(self):
        base_config = super().get_config()
        config = {
            'decoder_layers' : self.decoder_layers,
            'embedding_dim' : self.embedding_dim,
            'key_dim' : self.key_dim,
            'num_heads' : self.num_heads,
            'dropout_rate' : self.dropout_rate,
            'ff_dim' : self.ff_dim,
            'vocab_size' : self.vocab_size,
            'n_sequence_tokens' : self.n_sequence_tokens
        }
        return {**base_config, **config}