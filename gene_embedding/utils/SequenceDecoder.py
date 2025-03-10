import tensorflow as tf
from keras import models, layers

# from utils.PositionalEmbedding import PositionalEmbedding
from utils.TransformerBlock import TransformerDecoderBlock
from utils.RotaryPositionEmbedding import RotaryPositionEncoding
from utils.DNATokenizer import DNATokenizer


# @tf.function
# def positional_encoding(length, depth):
#     depth = depth/2

#     positions = tf.range(length, dtype=tf.float32)[:, None]     # (seq, 1)
#     depths = tf.range(depth, dtype=tf.float32)[None, :]/depth   # (1, depth)

#     angle_rates = 1 / (10000**depths)         # (1, depth)
#     angle_rads = positions * angle_rates      # (pos, depth)

#     pos_encoding = tf.concat(
#         [tf.sin(angle_rads), tf.cos(angle_rads)],
#         axis=-1
#     ) 

#     return tf.cast(pos_encoding, dtype=tf.float32)



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
                       decode_length:int=5000,
                       **kwargs):
        super(SequenceDecoder, self).__init__(**kwargs)

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
        self.decode_length = tf.Variable(decode_length, trainable=False, dtype=tf.int32, name="decoder_decode_length")

        

        # self.test_token =  DNATokenizer(tokenization_method="nucleotide",
        #                                 max_length=max_length)
        # self.test_embed = layers.Embedding(input_dim=self.vocab_size, 
        #                               output_dim=embedding_dim, 
        #                               mask_zero=True)

        ## Define the transformer block layers
        # self.transformer_layer = TransformerEncoderBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads,
        #                                                  key_dim=key_dim, dropout_rate=dropout_rate)
        # self.transformer_layer = TransformerDecoderBlock(output_dim=self.embedding_dim, ff_dim=self.ff_dim, num_heads=self.num_heads,
        #                                                  key_dim=self.key_dim, dropout_rate=self.dropout_rate)

        self.transformer_layers = [
            TransformerDecoderBlock(output_dim=self.embedding_dim, ff_dim=self.ff_dim, num_heads=self.num_heads,
                                    key_dim=self.key_dim, dropout_rate=self.dropout_rate)
            for _ in range(self.decoder_layers)
        ]

        ## Define the positional encoding
        # self.pos_encoding = positional_encoding(self.max_length, self.embedding_dim)
        self.query_position_encoder = RotaryPositionEncoding(self.max_length, self.embedding_dim)
        self.latent_position_encoder = RotaryPositionEncoding(self.n_sequence_tokens, self.latent_dim)

        ## Define the final output layer
        self.final_layer = layers.Dense(self.vocab_size, activation="softmax")
        # self.final_layer = layers.Softmax()

        ## Define the mask token variable
        #  -> defined to allow broadcasting across the batch and sequence dimensions
        self.query_tokens = self.add_weight(
            name="query_tokens",
            shape=(1,1,self.embedding_dim),
            initializer="uniform",
            trainable=True
        )



    def call(self, x, **kwargs):
        ## Repeat the query sequence along the batch and sequence dimensions and add position encoding
        query_seq = tf.tile(self.query_tokens, [tf.shape(x)[0], self.decode_length, 1])
        query_seq = self.query_position_encoder(query_seq)
        ## Apply positional encoding to the key sequence
        # x = self.test_token(x)
        # x = tf.cast(x, "float32")
        # x = self.test_embed(x)
        x = self.latent_position_encoder(x)
        ## Pass through the decoding layers
        for dec_layer in self.transformer_layers: 
            query_seq = dec_layer(query=query_seq, value=x, use_residuals=True, **kwargs)
        # return layers.Softmax()(query_seq)
        out = self.final_layer(query_seq)
        # out.set_shape(self.compute_output_shape(x.shape))
        # print("decode length debug:", self.decode_length.numpy())
        # out = tf.ensure_shape(out, [x.shape[0], self.decode_length, self.vocab_size])
        # print("decoder call debug:", out.shape)
        return out
    

    def _update_decode_length(self, new_length):
        # self.decode_length = new_length
        self.decode_length.assign(new_length)


    # def compute_output_shape(self, input_shape):
    #     return [input_shape[0], self.decode_length.numpy(), self.vocab_size]


    def build(self, input_shape):
        super().build(input_shape)
    

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
            'n_sequence_tokens' : self.n_sequence_tokens,
            'max_length' : self.max_length,
            'decode_length' : self.decode_length.numpy()
        }
        return {**base_config, **config}