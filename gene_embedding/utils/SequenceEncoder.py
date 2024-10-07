import tensorflow as tf
from keras import layers
from keras import models

# from utils.PositionalEmbedding import PositionalEmbedding
from utils.RotationalPositionEmbedding import RotaryPositionEncoding
from utils.TransformerBlock import TransformerEncoderBlock
from utils.DNATokenizer import DNATokenizer



@tf.keras.saving.register_keras_serializable()
class SequenceEncoder(models.Model):

    '''
    @TODO
    '''

    def __init__(self, tokenization_method:str,
                       encoder_layers:int, 
                       embedding_dim:int,
                       key_dim:int,
                       num_heads:int,
                       dropout_rate:float,
                       ff_dim:int,
                       masking_rate:float,
                       sequence_token=None,
                       n_sequence_tokens:int=1,
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
        self.n_sequence_tokens = n_sequence_tokens
        self.max_length = max_length

        ## Define the tokenizing and positional embedding
        self.tokenizing_layer = DNATokenizer(tokenization_method=tokenization_method,
                                             sequence_token=sequence_token,
                                             max_length=max_length)
        self.vocabulary = self.tokenizing_layer.tokenizing_layer.get_vocabulary()
        self.vocab_size = len(self.vocabulary)

        self.token_masker = layers.Dropout(rate=masking_rate)
        self.embedding_layer = layers.Embedding(input_dim=self.vocab_size, 
                                                output_dim=embedding_dim, 
                                                mask_zero=True)
        self.position_encoder = RotaryPositionEncoding(max_length, embedding_dim)

        ## Define the transformer block layers
        self.transformer_layers = [
            TransformerEncoderBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads, 
                                    key_dim=key_dim, dropout_rate=dropout_rate) 
            for _ in range(encoder_layers)
        ]

        ## Define the querry token sequence
        self.query_tokens = self.add_weight(
            name="query_tokens",
            shape=(1,n_sequence_tokens,embedding_dim),
            initializer="glorot_uniform"
        )

        ## Define some arrithemtic layers
        # self.mult = layers.Multiply()
        # self.subtract = layers.Subtract()


    def call(self, x, **kwargs):
        ## Convert the input strings to tokens
        tokens = self.tokenizing_layer(x, **kwargs)
        float_tokens = tf.cast(tokens, "float32")
        ## Randomly mask the tokens for training
        masked_tokens = self.token_masker(float_tokens, **kwargs)
        # Undo the Dropout layer's normalization if needed
        if kwargs['training']: 
            masked_tokens *= (1-self.masking_rate)
        ## Compute positional embeddings
        enc_z = self.embedding_layer(masked_tokens)
        enc_z = self.position_encoder(enc_z)

        ## Pass through the early transformer layers
        if self.encoder_layers > 1:
            for enc_layer in self.transformer_layers[:-1]:
                enc_z = enc_layer(query=enc_z, key=enc_z, value=enc_z, **kwargs)
        ## Pass through the final transformer layer
        query_seq = tf.repeat(self.query_tokens, repeats=tf.shape(x)[0], axis=0)
        enc_z = self.transformer_layers[-1](query=query_seq, key=enc_z, value=enc_z, **kwargs)
        return enc_z

    def prepend_sequence_tokens(self, batch):
        prepend_seq = self.n_sequence_tokens*self.sequence_token
        chars = tf.constant([prepend_seq], dtype=tf.string)
        chars = tf.repeat(chars, tf.shape(batch)[0], axis=0)
        newbatch = tf.strings.join([chars, batch])
        return newbatch

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
        

