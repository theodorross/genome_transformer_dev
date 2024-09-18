import tensorflow as tf
from keras import layers
from keras import models

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
        - try to fix accuracy and Levenshtein distance metrics
            - I think this has to do with masking/padding (accuracy and loss done)
        - test with various embedding dimmensionalities
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

        ## Build the sub-models
        # Encoder
        self.encoder = SequenceEncoder(tokenization_method, encoder_layers, embedding_dim,
                                       key_dim, num_heads, dropout_rate, ff_dim,
                                       masking_rate, max_length)
        
        self.vocab_size = self.encoder.vocab_size
        self.vocabulary = self.encoder.vocabulary

        # Decoder
        self.decoder = SequenceDecoder(decoder_layers, embedding_dim, key_dim, num_heads,
                                       dropout_rate, ff_dim, self.vocab_size)

        ## Compile the model
        # Define the optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Define the objective function
        loss = MaskedSparseCategoricalCrossentropy(mask_category=0)

        # Define performance metrics to track
        levenshtein_metric = LevenshteinDistance(self.vocabulary)
        masked_accuracy = MaskedAccuracy(mask_category=0)
        track_metrics = [masked_accuracy,
                         levenshtein_metric]

        self.build(input_shape=(None,1))
        self.compile(optimizer=opt, loss=loss, metrics=track_metrics)

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def tokenize(self, x, one_hot=False):
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
    



def masked_categorical_crossentropy(label, pred):
    ## Compute the loss
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    # loss = loss_object(label, pred)
    loss = tf.keras.losses.sparse_categorical_crossentropy(label, pred, from_logits=False)

    ## Mask the computed loss
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = tf.multiply(loss, mask)

    ## Compute the mean of the loss across the masked values
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


# def masked_accuracy(label, pred):
#     ## Compute the predicted categories 
#     pred = tf.argmax(pred, axis=2)
#     # label = tf.argmax(label, axis=2)
#     label = tf.cast(label, pred.dtype)

#     ## Check for matches between predictions and labels
#     # match = label == pred
#     match = tf.equal(label, pred)

#     ## Mask the matches based
#     # mask = label != 0
#     mask = tf.not_equal(label, 0)
#     # match = match & mask
#     match = tf.logical_and(match, mask)

#     ## Compute the accuracy 
#     match = tf.cast(match, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(match)/tf.reduce_sum(mask)

