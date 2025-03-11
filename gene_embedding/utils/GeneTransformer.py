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



@tf.keras.utils.register_keras_serializable()
class GeneTransformer(models.Model):
    '''
    Transformer model for embedding gene sequences
    - default values taken from "Attention is All You Need": https://arxiv.org/pdf/1706.03762
    - no longer resembles a normal transformer...
    '''

    def __init__(self, tokenization_method:str, 
                       embedding_dim:int=4,
                       latent_dim:int=512,
                       encoder_layers:int=6, 
                       decoder_layers:int=6, 
                       key_dim:int=64, 
                       num_heads:int=8, 
                       dropout_rate:float=0.1, 
                       ff_dim:int=2048, 
                       max_length:int=5000,
                       decode_length:int=5000,
                       masking_rate:float=0.05,
                       learning_rate:float=1e-6,
                       n_sequence_tokens:int=2,
                       **kwargs):
        super(GeneTransformer, self).__init__(**kwargs)

        ## Store configuration arguments
        self.tokenization_method = tokenization_method
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        # self.max_length = max_length
        self.masking_rate = masking_rate
        self.learning_rate = learning_rate
        self.n_sequence_tokens = n_sequence_tokens

        if tokenization_method.lower() == "nucleotide":
            self.max_length = max_length
            self.decode_length = decode_length
        elif tokenization_method.lower() == "codon":
            self.max_length = int(np.ceil(max_length/3))
            self.decode_length = int(np.ceil(decode_length/3))

        ## Build the sub-models        
        # Encoder
        self.encoder = SequenceEncoder(tokenization_method=self.tokenization_method, 
                                       encoder_layers=self.encoder_layers, 
                                       embedding_dim=self.embedding_dim, 
                                       latent_dim=self.latent_dim,
                                       key_dim=self.key_dim, 
                                       num_heads=self.num_heads,
                                       dropout_rate=self.dropout_rate, 
                                       ff_dim=self.ff_dim,
                                       masking_rate=self.masking_rate, 
                                       n_sequence_tokens=self.n_sequence_tokens, 
                                       max_length=self.max_length,
                                       decode_length=self.decode_length)
        self.vocab_size = self.encoder.vocab_size
        self.vocabulary = self.encoder.vocabulary

        # Decoder
        self.decoder = SequenceDecoder(decoder_layers=self.decoder_layers, 
                                       embedding_dim=self.embedding_dim, 
                                       latent_dim=self.latent_dim, 
                                       key_dim=self.key_dim, 
                                       num_heads=self.num_heads,
                                       dropout_rate=self.dropout_rate, 
                                       ff_dim=self.ff_dim, 
                                       vocab_size=self.vocab_size, 
                                       n_sequence_tokens=self.n_sequence_tokens,
                                       max_length=self.max_length,
                                       decode_length=self.decode_length)


        ## Compile the model
        # Define the optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # opt = "adam"
        # opt2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # opt3 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Define the objective function
        # self.loss = MaskedSparseCategoricalCrossentropy(mask_category=0)
        self.loss = "sparse_categorical_crossentropy"

        # Define performance metrics to track
        # levenshtein_metric = LevenshteinDistance(self.vocabulary)
        # masked_accuracy = MaskedAccuracy(mask_category=0)
        # track_metrics = [masked_accuracy,
        #                  levenshtein_metric]
        track_metrics = "accuracy"

        # self.encoder.compile(optimizer=opt2, loss=loss, metrics=track_metrics, weighted_metrics=[])
        # self.decoder.compile(optimizer=opt3, loss=loss, metrics=track_metrics, weighted_metrics=[])
        self.compile(optimizer=opt, loss=self.loss, metrics=track_metrics, weighted_metrics=[])

        ## Run a dummy input through the model
        dummy_in = tf.convert_to_tensor([["atgatgatg"]])
        self(dummy_in)


    def build(self, input_shape):
        super().build(input_shape)



    def call(self, x, **kwargs):
        ## Pass through the encoder
        z = self.encoder(x, **kwargs)
        ## Pass through the decoder
        y = self.decoder(z, **kwargs)
        return y


    def encode(self, x, **kwargs):
        return self.encoder.predict(x, **kwargs)

    def decode(self, z, **kwargs):
        return self.decoder.predict(z, **kwargs)


    def update_decode_length(self, new_length):
        ## Compute the new decoding length
        if self.tokenization_method.lower() == "nucleotide":
            self.decode_length = new_length
        elif self.tokenization_method.lower() == "codon":
            self.decode_length = int(np.ceil(new_length/3))

        ## Set sub-objects to have the new decoding length
        self.encoder._update_decode_length(self.decode_length)
        self.decoder._update_decode_length(self.decode_length)


    def tokenize(self, x, one_hot=False):
        ## Add a batch dimension if needed
        # if x.shape.rank < 1:
        if len(x.shape) == 0:
            x = tf.expand_dims(x, axis=0)
        ## Convert a DNA sequence to a sequence of tokens, optionally one-hot encoded
        tokens = self.encoder.tokenizing_layer(x)
        # tokens = tf.ensure_shape(tokens, [None, None])
        if one_hot:
            tokens = tf.squeeze( tf.one_hot(tokens, depth=self.vocab_size), axis=0 )
            # tokens = tf.one_hot(tokens, depth=self.vocab_size)
            return tokens
        else:
            tokens = tf.squeeze(tokens, axis=0)
            # print("tokenize debug:", x.shape, tokens.shape)
            return tokens
        


    def _align_data_to_devices(self, data:tf.data.Dataset, batch_size:int, n_replicas:int=1, verbose:bool=True) -> tuple[int,tf.data.Dataset]:
        ## Determine the cardinality of the input dataset
        if data.cardinality() > 0:
            cardinality = data.cardinality()
        else:
            cardinality = 0
            for _ in enumerate(data):
                cardinality += 1

        ## Define a set of alternative options for the batch sizes and ensure the batch sizes 
        ## are greater than the number of devices
        batch_size_offsets = np.arange(-5, 6)*n_replicas
        batch_size_options = batch_size + batch_size_offsets
        batch_size_options = batch_size_options[batch_size_options > n_replicas] 
        
        ## Select a batch size that requires discarding the fewest validation samples
        discard_options = cardinality % batch_size_options
        new_batch_size = batch_size_options[np.argmin(discard_options)]
        # print("DEBUG:")
        # print("n_replicas:", n_replicas)
        # print("batch_size_options:", batch_size_options)
        # print("discard_options:   ", discard_options)
        # print("argmin:", np.argmin(discard_options))
        # print("new_batch_size:", new_batch_size)

        ## Discard samples until the dataset size is evenly divisible by the new batch size
        needed_discards = min(discard_options)
        samples_to_keep = cardinality - needed_discards
        new_data = data.take(samples_to_keep)
        
        if verbose:
            # print(f"Resetting batch size from {batch_size} to {new_batch_size} to fit dataset of length {cardinality}")
            print(f"Changing batch size and dataset cardinality, will lose {needed_discards} validation samples of {cardinality}")
            print(f"\tdata cardinality: {cardinality} -> {samples_to_keep}")
            print(f"\tbatch_size: {batch_size} -> {new_batch_size}")
        return new_batch_size, new_data
    


    def _preprocess_dataset(self, data:tf.data.Dataset, weights:tf.lookup.StaticHashTable=None) -> tf.data.Dataset:
        ## Map the dataset to a label dataset and randomly mask the inputs
        y = data.map(self.tokenize)
        # temp = tf.keras.Sequential()
        # temp.add(layers.Input(shape=(50,)))
        # y = y.map(temp)
        # _y = data.map(lambda x: self.tokenize(x, one_hot=True))
        if weights is None:
            return tf.data.Dataset.zip(data,y)
            # return tf.data.Dataset.zip(_y,y)
        else:
            w = y.map(weights.lookup)
            return tf.data.Dataset.zip(data,y,w)
            # return tf.data.Dataset.zip(_y,y,w)
        


    def _compute_token_weights(self, data:tf.data.Dataset, sample:int=0):
        ## Compute training weights for each token based on the token frequencies
        ## Tokenize and optionally shuffle the input dataset
        _data = data.map(lambda x: self.tokenize(x, one_hot=True))
        if sample!=0:
            _data = _data.shuffle(sample)

        ## Compute the occurance of each token in the dataset
        counts = np.zeros(self.encoder.vocab_size)
        for ix,g in enumerate(_data):
            if ix+1 == sample:
                break
            counts += g.numpy().sum(axis=0).squeeze()

        ## Compute weights for tokens with non-zero presence
        _w = ((ix+1) * self.decode_length) / (counts[counts!=0] * sum(counts!=0))
        # Use 1 for the weight of absent tokens
        weights = np.ones(counts.shape)
        weights[counts!=0] = _w

        ## Define the weights as both a dict and a StaticHashTable
        weight_dict = {ix:w for ix,w in enumerate(weights)}
        weight_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                list(weight_dict.keys()),
                list(weight_dict.values()),
                key_dtype=tf.int64,
                value_dtype=tf.float64
            ),
            default_value=1
        )
        return weight_dict, weight_table
    


    def train(self, 
              data:tf.data.Dataset, 
              val_data:tf.data.Dataset,
              batch_size:int, 
              epochs:int, 
              callbacks:list,
              num_devices:int=1,
              **kwargs):

        ## Compute token frequencies to inform class weights
        _,weight_table = self._compute_token_weights(data)

        ## Preprocess the input data for training
        _training = self._preprocess_dataset(data, weight_table)
        # _training = self._preprocess_dataset(data)
        # _training = _training.shuffle(buffer_size=_training.cardinality())
        _training = _training.batch(batch_size).cache()
        _training = _training.prefetch(tf.data.AUTOTUNE)

        _validation = self._preprocess_dataset(val_data, weight_table)
        # _validation = self._preprocess_dataset(val_data)
        _validation_batch, _validation = self._align_data_to_devices(_validation, batch_size, num_devices)
        _validation = _validation.batch(_validation_batch).cache()
        _validation = _validation.prefetch(tf.data.AUTOTUNE)

        # print("generator debug:")
        # print(_training.element_spec)
        # for x,y,w in _training:
        #     print(x.shape)
        #     print(y.shape)
        #     print(w.shape)
        #     break
        # exit()
        
        ## Train the model
        # H = self.fit(_training, validation_data=_validation, epochs=epochs, callbacks=callbacks, **kwargs)
        H = self.fit(_training, epochs=epochs)

        return H.history

    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "tokenization_method":self.tokenization_method,
            "embedding_dim":self.embedding_dim,
            "latent_dim":self.latent_dim,
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
    
