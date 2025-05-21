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
                       functional_loss:float=1.0,
                       reconstruction_loss:float=1.0,
                       clustering_loss:float=1.0,
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
        
        # Linear Classifier head
        self.classifier = models.Sequential(name="category_classifier")
        self.classifier.add(layers.Input(shape=(latent_dim*n_sequence_tokens,)))
        self.classifier.add(layers.Dense(23, activation="sigmoid"))
        
        ## Run a dummy input through the model
        dummy_in = layers.Input((), dtype=tf.string)
        dummy_mid = self.encoder(dummy_in)
        self.decoder(dummy_mid)
        self(dummy_in)

        ## Compile the model
        # Define the optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # opt = "adam"
        # opt2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # opt3 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Define the objective function
        self.reconstruction_loss = MaskedSparseCategoricalCrossentropy(mask_category=0, name="reconstruction")
        self.triplet_loss = OnlineMarginTripletLoss(margin=15, name="clustering")
        self.category_loss = MaskedBinaryCrossentropy(name="COG_category")
        # self.reconstruction_loss = "sparse_categorical_crossentropy"

        # Define performance metrics to track
        levenshtein_metric = LevenshteinDistance(self.vocabulary)
        masked_accuracy = MaskedAccuracy(mask_category=0, name="reconstruction_accuracy")
        category_accuracy = MaskedBinaryAccuracy(name="COG_category_accuracy")
        recon_metrics = [masked_accuracy,
                         levenshtein_metric]
        latent_metrics = []
        classifier_metrics = [category_accuracy]
        loss_weights = [clustering_loss, 
                        functional_loss,
                        reconstruction_loss]
        metrics = [latent_metrics, classifier_metrics, recon_metrics]

        # self.encoder.compile(optimizer=opt2, loss=loss, metrics=track_metrics, weighted_metrics=[])
        # self.decoder.compile(optimizer=opt3, loss=loss, metrics=track_metrics, weighted_metrics=[])
        # self.compile(optimizer=opt, loss=self.loss, metrics=[MaskedAccuracy(mask_category=0)])
        # self.compile(optimizer=opt, loss=self.loss, metrics=['accuracy'], weighted_metrics=[])
        self.compile(optimizer=opt, 
                     loss=[self.triplet_loss, self.category_loss, self.reconstruction_loss],
                     loss_weights=loss_weights,
                     metrics=metrics)
        # self.compile(optimizer=opt, 
        #              loss=self.reconstruction_loss, 
        #              metrics=recon_metrics)


    def build(self, input_shape):
        super().build(input_shape)



    def call(self, x, **kwargs):
        ## Pass through the encoder
        z = self.encoder(x, **kwargs)
        ## Classify the category
        z_cat = self.classifier(z)
        ## Pass through the decoder
        y = self.decoder(z, **kwargs)
        return z,z_cat,y
        # return y


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
        if len(x.shape) == 0:
            x = tf.expand_dims(x, axis=0)
        ## Convert a DNA sequence to a sequence of tokens, optionally one-hot encoded
        tokens = self.encoder.tokenizing_layer(x)
        if one_hot:
            tokens = tf.squeeze( tf.one_hot(tokens, depth=self.vocab_size), axis=0 )
            return tokens
        else:
            tokens = tf.squeeze(tokens, axis=0)
            return tokens
    


    def preprocess_dataset(self, data:tf.data.Dataset, batch_size:int, validation_data:tf.data.Dataset=None, weighted:bool=True) -> tf.data.Dataset:
        ## Map the dataset to a label dataset and randomly mask the inputs
        train_genes = data.map(lambda g,d,c: g)
        train_domains = data.map(lambda g,d,c: d)
        train_categs = data.map(lambda g,d,c:c)
        train_y = train_genes.map(self.tokenize)
        new_train_y = tf.data.Dataset.zip(train_domains, train_categs, train_y)
        if validation_data is not None:
            val_genes = validation_data.map(lambda g,d,c: g)
            val_domains = validation_data.map(lambda g,d,c: d)
            val_categs = validation_data.map(lambda g,d,c: c)
            val_y = val_genes.map(self.tokenize)
            new_val_y = tf.data.Dataset.zip(val_domains, val_categs, val_y)

        # If class weighs are to be used
        if weighted:
            _,weights = self._compute_token_weights(train_genes)
            w = train_y.map(weights.lookup)
            new_data = tf.data.Dataset.zip(train_genes, new_train_y, w)
            if validation_data is not None:
                new_val_data = tf.data.Dataset.zip(val_genes, new_val_y, w)

        # If no class weights will be used
        else:
            new_data = tf.data.Dataset.zip(train_genes, new_train_y)
            if validation_data is not None:
                new_val_data = tf.data.Dataset.zip(val_genes, new_val_y)

        ## Shuffle and batch the training data
        shuffle_buffer = min(new_data.cardinality(), batch_size*1000)
        new_data = new_data.shuffle(buffer_size=shuffle_buffer)
        new_data = new_data.batch(batch_size=batch_size, drop_remainder=False).repeat()
        if validation_data is not None:     # Only batch the validation data

            # Find the cardinality of the validation data
            new_val_card = 0
            for _ in new_val_data:
                new_val_card += 1
            # If there are fewer validation samples than batch size, use a smaller batch size
            if batch_size > new_val_card:
                val_batch_size = new_val_card
            else:
                val_batch_size = batch_size
            new_val_data = new_val_data.batch(batch_size=val_batch_size, drop_remainder=True)
            return new_data, new_val_data
        else:
            return new_data
        


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
    

    # def fit(self, x=None, y=None, batch=None, epochs=1, verbose='auto', callbacks=None, validation_split=0.0, validation_data=None,
    #         shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1):
    # def fit(self, x:tf.data.Dataset, validation_data:tf.data.Dataset=None, callbacks=None, steps_per_epoch:int=None, initial_epoch:int=0):
    #     '''
    #     Custom training loop for triplet loss
    #     '''
        


    #     return
    

    
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
    
