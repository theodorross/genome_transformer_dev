import tensorflow as tf
from keras import layers
from keras import models
import numpy as np

# from utils import PositionalEmbedding, TransformerBlock, DNATokenizer, LevenshteinDistance
# from utils.PositionalEmbedding import PositionalEmbedding
# from utils.TransformerBlock import TransformerBlock
# from utils.DNATokenizer import DNATokenizer
from utils.TrainingUtils import *
from utils.SequenceEncoder import SequenceEncoder
from utils.SequenceDecoder import SequenceDecoder



@tf.keras.saving.register_keras_serializable()
class GeneTransformer(models.Model):
    '''
    Transformer model for embedding gene sequences
    - default values taken from "Attention is All You Need": https://arxiv.org/pdf/1706.03762
    
    @TODO: 
        - codon tokenization doesn't quite work right, not sure why
        - fix sequence masking during attention
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
                       n_sequence_tokens:int=2,
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
        self.n_sequence_tokens = n_sequence_tokens

        if tokenization_method.lower() == "nucleotide":
            # self.sequence_token = "x"
            self.sequence_token = None
        elif tokenization_method.lower() == "codon":
            # self.sequence_token = "seq"
            self.sequence_token = None

        ## Build the sub-models
        # Encoder
        self.encoder = SequenceEncoder(tokenization_method, encoder_layers, embedding_dim,
                                       key_dim, num_heads, dropout_rate, ff_dim,
                                       masking_rate, self.sequence_token, n_sequence_tokens, max_length)
        
        self.vocab_size = self.encoder.vocab_size
        self.vocabulary = self.encoder.vocabulary

        # Decoder
        self.decoder = SequenceDecoder(decoder_layers, embedding_dim, key_dim, num_heads,
                                       dropout_rate, ff_dim, self.vocab_size, n_sequence_tokens,
                                       max_length)
        

        ## Compile the model
        # Define the optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Define the objective function
        # loss = MaskedSparseCategoricalCrossentropy(mask_category=0)
        loss = "sparse_categorical_crossentropy"

        # Define performance metrics to track
        levenshtein_metric = LevenshteinDistance(self.vocabulary)
        masked_accuracy = MaskedAccuracy(mask_category=0)
        track_metrics = [masked_accuracy,
                         levenshtein_metric]

        self.build(input_shape=(None,))
        self.compile(optimizer=opt, loss=loss, metrics=track_metrics)

    def call(self, x, **kwargs):
        x = tf.cast(x, tf.string)
        ## Pass through the encoder
        z = self.encoder(x, **kwargs)
        ## Pass through the decoder
        return self.decoder(z, **kwargs)

    def encode(self, x, **kwargs):
        return self.encoder.predict(x, **kwargs)

    def decode(self, z, **kwargs):
        return self.decoder.predict(z, **kwargs)
    
    def tokenize(self, x, one_hot=False):
        ## Add a batch dimension if needed
        # if x.shape.rank < 1:
        if len(x.shape) == 0:
            x = tf.expand_dims(x, axis=0)
        ## Convert a DNA sequence to a sequence of tokens, optionally one-hot encoded
        tokens = self.encoder.tokenizing_layer(x)
        if one_hot:
            return tf.one_hot(tokens, depth=self.vocab_size)
        else:
            return tf.squeeze(tokens)
        # return tokens
        
    def preprocess_genes(self, data:tf.data.Dataset):
        ## Add the sequence character to the beginning of each gene sequence
        prepend_char = lambda x: self.sequence_token + x
        return data.map(prepend_char)
    
    def _random_mask(self, s):
        ## Randomly remove a fraction of characters from a string
        lens = tf.strings.length(s)
        n_mask = tf.math.floor( self.masking_rate*tf.cast(lens, tf.float32) )
        n_mask = tf.cast(n_mask, dtype=tf.int32)
        keep_indices = tf.random.shuffle( tf.range(lens) )[n_mask:]
        keep_indices = tf.squeeze( tf.sort( keep_indices ) )
        new_s = tf.strings.substr(s, pos=keep_indices, len=tf.ones(tf.shape(keep_indices), dtype=tf.int32))
        new_s = tf.strings.reduce_join(new_s)
        return new_s
    
    def _preprocess_dataset(self, data:tf.data.Dataset, mask:bool=False) -> tf.data.Dataset:
        ## Map the dataset to a label dataset and randomly mask the inputs
        y = data.map(self.tokenize)
        if mask:
            x = data.map(self._random_mask)
            return tf.data.Dataset.zip(x,y)
        else:
            return tf.data.Dataset.zip(data,y)
    
    def train(self, 
              data:tf.data.Dataset, 
              val_data:tf.data.Dataset,
              batch_size:int, 
              epochs:int, 
              *callbacks,
              **kwargs):
        
        # ## Define and format the reconstruction targets as a dataset
        # y = data.map(self.tokenize)
        # val_y = val_data.map(self.tokenize)

        # ## Create the training dataset
        # x = tf.data.Dataset.zip(data,y)
        # x = x.shuffle(buffer_size=100*batch_size)
        # # x = x.padded_batch(batch_size)
        # x = x.batch(batch_size)
        # x = x.prefetch(tf.data.AUTOTUNE)

        # ## Create the validation dataset
        # val_x = tf.data.Dataset.zip(val_data, val_y)
        # # val_x = val_x.padded_batch(batch_size)
        # val_x = val_x.batch(batch_size)
        # val_x = val_x.prefetch(tf.data.AUTOTUNE)

        # H = self.fit(x, validation_data=val_x, epochs=epochs, callbacks=callbacks, **kwargs)
        # return H.history

        ## Preprocess the input data for training
        _training = self._preprocess_dataset(data)
        _training = _training.shuffle(buffer_size=100*batch_size)
        _training = _training.batch(batch_size)
        _training = _training.prefetch(tf.data.AUTOTUNE)

        _validation = self._preprocess_dataset(val_data)
        _validation = _validation.batch(batch_size)
        _validation = _validation.prefetch(tf.data.AUTOTUNE)
        
        ## Train the model
        H = self.fit(_training, validation_data=_validation, epochs=epochs, callbacks=callbacks, **kwargs)
        return H.history

    

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
            "learning_rate":self.learning_rate,
            "n_sequence_tokens":self.n_sequence_tokens
        }
        return {**base_config, **config}
    
