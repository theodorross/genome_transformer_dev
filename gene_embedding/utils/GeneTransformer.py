import tensorflow as tf
from keras import layers
from keras import models

# from utils import PositionalEmbedding, TransformerBlock, DNATokenizer, LevenshteinDistance
# from utils.PositionalEmbedding import PositionalEmbedding
# from utils.TransformerBlock import TransformerBlock
# from utils.DNATokenizer import DNATokenizer
from utils.LevenshteinDistance import LevenshteinDistance
from utils.SequenceEncoder import SequenceEncoder
from utils.SequenceDecoder import SequenceDecoder


@tf.keras.saving.register_keras_serializable()
class GeneTransformer(models.Model):
    '''
    Transformer model for embedding gene sequences
    - default values taken from "Attention is All You Need": https://arxiv.org/pdf/1706.03762
    '''

    def __init__(self, tokenization_method:str, 
                       embedding_dim:int=512, 
                       encoder_layers:int=6, 
                       decoder_layers:int=6, 
                       key_dim:int=64, 
                       num_heads:int=8, 
                       dropout_rate:float=0.1, 
                       ff_dim:int=2048, 
                       max_length:int=5000,
                       masking_rate:float=0.05,
                       learning_rate:float=1e-6,
                       **kwargs):
        super().__init__(**kwargs)

        ## Store configuration arguments
        self.tokenization_method = tokenization_method
        self.embedding_dim = embedding_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.max_length = max_length
        self.masking_rate = masking_rate
        self.learning_rate = learning_rate

        ## Define the tokenizing and positional embedding
        # self.tokenizing_layer = DNATokenizer(tokenization_method=tokenization_method)
        # self.vocabulary = self.tokenizing_layer.vocabulary
        # self.vocab_size = len(self.vocabulary)

        # self.token_masker = layers.Dropout(rate=masking_rate)
        # self.embedding_layer = PositionalEmbedding(self.vocab_size, embedding_dim, max_length=max_length)

        ## Define the compute layers
        # self.encoder_layers = [
        #     TransformerBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads, 
        #                      key_dim=key_dim, dropout_rate=dropout_rate) 
        #     for _ in range(encoder_layers)
        # ]
        # self.decoder_layers = [
        #     TransformerBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads, 
        #                     key_dim=key_dim, dropout_rate=dropout_rate)
        #     for _ in range(decoder_layers)
        # ]

        # self.final_layer = layers.Dense(self.vocab_size, activation="softmax")

        ## Build the sub-models
        # Encoder
        # input_sequence = layers.Input(shape=(), dtype="string")
        # tokens = self.tokenizing_layer(input_sequence)
        # float_tokens = tf.cast(tokens, "float32")
        # masked_tokens = self.token_masker(float_tokens)
        # enc_z = self.embedding_layer(masked_tokens)
        # for enc_layer in self.encoder_layers:
        #     enc_z = enc_layer(enc_z)
        # self.encoder = models.Model(inputs=[input_sequence], outputs=[enc_z], name="encoder")

        self.encoder = SequenceEncoder(tokenization_method, encoder_layers, embedding_dim,
                                       key_dim, num_heads, dropout_rate, ff_dim,
                                       masking_rate, max_length)
        
        self.vocab_size = self.encoder.vocab_size
        self.vocabulary = self.encoder.vocabulary

        # Decoder
        # input_latent = layers.Input(shape=(None, embedding_dim))
        # dec_z = layers.Identity()(input_latent)
        # for dec_layer in self.decoder_layers:
        #     dec_z = dec_layer(dec_z)
        # self.decoder = models.Model(inputs=[input_latent], outputs=[dec_z], name="decoder")

        self.decoder = SequenceDecoder(decoder_layers, embedding_dim, key_dim, num_heads,
                                       dropout_rate, ff_dim, self.vocab_size)

        ## Compile the model
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.levenshtein_metric = LevenshteinDistance(self.vocabulary)
        track_metrics = ["acc",
                         self.levenshtein_metric]

        # self.autoencoder.compile(optimizer=opt, loss="categorical_crossentropy", metrics=track_metrics)
        # self.build(input_shape=input_sequence.shape)
        self.build(input_shape=(None,1))
        self.compile(optimizer=self.opt, loss="categorical_crossentropy", metrics=track_metrics)

    def call(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def tokenize(self, x, one_hot=True):
        ## Convert a DNA sequence to a sequence of tokens, optionally one-hot encoded
        # tokens = self.tokenizer(x)
        # tokens = self.tokenizing_layer(x)
        tokens = self.encoder.tokenizing_layer(x)
        if one_hot:
            return tf.one_hot(tokens, depth=self.vocab_size)
        else:
            return tokens
    
    def train(self, 
              data:tf.data.Dataset, 
              val_data:tf.data.Dataset,
              batch_size:int, 
              epochs:int, 
              *callbacks):

        ## Define and format the reconstruction targets as a dataset
        y = data.map(self.tokenize)
        y_val = val_data.map(self.tokenize)

        ## Create the training dataset
        x = tf.data.Dataset.zip(data,y)
        x = x.shuffle(buffer_size=100*batch_size)
        x = x.padded_batch(batch_size)
        x = x.prefetch(tf.data.AUTOTUNE)

        ## Create the validation dataset
        val_x = tf.data.Dataset.zip(val_data, y_val)
        val_x = val_x.padded_batch(batch_size)
        val_x = val_x.prefetch(tf.data.AUTOTUNE)
            
        ## Train the model
        # H = self.autoencoder.fit(x, validation_data=val_x, epochs=epochs, callbacks=callbacks)
        H = self.fit(x, validation_data=val_x, epochs=epochs, callbacks=callbacks)
        return H
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "tokenization_method":self.tokenization_method,
            "embedding_dim":self.embedding_dim,
            "encoder_layers":self.encoder_layers,
            "decoder_layers":self.decoder_layers,
            "key_dim":self.key_dim,
            "num_heads":self.num_heads,
            "dropout_rate":self.dropout_rate,
            "ff_dim":self.ff_dim,
            "max_length":self.max_length,
            "masking_rate":self.masking_rate,
            "learning_rate":self.learning_rate
        }
        return {**base_config, **config}