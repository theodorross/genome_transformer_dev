import tensorflow as tf
from keras import models, layers

# from utils.PositionalEmbedding import PositionalEmbedding
from utils.TransformerBlock import TransformerDecoderBlock, TransformerEncoderBlock
from utils.RotaryPositionEmbedding import RotaryPositionEncoding
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
                       latent_dim:int,
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
        self.latent_dim = latent_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.n_sequence_tokens = n_sequence_tokens
        self.max_length = max_length

        ## Define the transformer block layers
        self.transformer_layer = TransformerEncoderBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads,
                                                         key_dim=key_dim, dropout_rate=dropout_rate)
        # self.transformer_layers = [
        #     layers.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim, dropout=dropout_rate)
        #     for _ in range(decoder_layers)
        # ]

        ## Define the positional encoding
        # self.pos_encoding = positional_encoding(self.max_length, self.embedding_dim)
        self.query_position_encoder = RotaryPositionEncoding(max_length, embedding_dim)
        self.latent_position_encoder = RotaryPositionEncoding(n_sequence_tokens, latent_dim)

        ## Define the final output layer
        self.final_layer = layers.Dense(vocab_size, activation="softmax")

        ## Define the mask token variable
        #  -> defined to allow broadcasting across the batch and sequence dimensions
        self.query_tokens = self.add_weight(
            name="query_tokens",
            shape=(1,max_length,embedding_dim),
            initializer="uniform",
            trainable=True
        )

        self.build(input_shape=(None, n_sequence_tokens, latent_dim))


    def call(self, x, **kwargs):
        ## Repeat the query sequence along the batch and sequence dimensions
        # query_seq = tf.tile(self.query_tokens, [tf.shape(x)[0], self.max_length,1])
        query_seq = tf.tile(self.query_tokens, [tf.shape(x)[0], 1,1])
        ## Apply positional encoding to the query and key sequence
        query_seq = self.query_position_encoder(query_seq)
        x = self.latent_position_encoder(x)
        ## Pass through the decoding layers
        query_seq = self.transformer_layer(query=query_seq, value=x, use_residuals=True, verbose=False, **kwargs)
        # return layers.Softmax()(query_seq)
        out = self.final_layer(query_seq, training=kwargs.get("training", False))
        return out
    

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