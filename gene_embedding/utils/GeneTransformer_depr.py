import tensorflow as tf
from keras.models import Model
from keras import layers
from keras.utils import serialize_keras_object, deserialize_keras_object
# from keras.utils import to_categorical
import itertools
import numpy as np


@tf.keras.utils.register_keras_serializable()
class GeneTransformer(Model):
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
                       batch_size:int=None, 
                       max_length:int=5000,
                       masking_rate:float=0.05,
                       learning_rate:float=1e-6,
                       **kwargs):
        super().__init__(**kwargs)

        ## Define parameters
        self.tokenization_method = tokenization_method
        self.embedding_dim = embedding_dim
        self.n_encoder_layers = encoder_layers
        self.n_decoder_layers = decoder_layers
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.masking_rate = masking_rate
        self.learning_rate = learning_rate


        ## Define the tokenizer 
        assert tokenization_method.lower() in {"nucleotide","codon"}

        if tokenization_method.lower() == "nucleotide":
            vocab = ['a','t','c','g']
            split_method = "character"
        elif tokenization_method.lower() == "codon":
            vocab = ["".join(c) for c in itertools.product("atcg", repeat=3)]
            split_method = self.codon_splitter

        self.tokenizing_layer = layers.TextVectorization(split=split_method, vocabulary=vocab)
        self.vocabulary = self.tokenizing_layer.get_vocabulary()
        self.vocab_size = len(self.vocabulary)
        # self.embedding_layer = layers.Embedding(self.vocab_size, embedding_dim, mask_zero=True)
        self.token_masker = layers.Dropout(rate=masking_rate)
        self.embedding_layer = PositionalEmbedding(self.vocab_size, embedding_dim, max_length=max_length)


        ## Define a lookup table for converting back to sequences
        self.lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=range(self.vocab_size),
                values=self.vocabulary,
                key_dtype="int32",
                value_dtype=tf.string
            ),
            default_value=self.vocabulary[0]
        )


        ## Define the layers
        self.encoder_layers = [
            TransformerBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads, 
                             key_dim=key_dim, dropout_rate=dropout_rate) 
            for _ in range(encoder_layers)
        ]
        self.decoder_layers = [
            TransformerBlock(output_dim=embedding_dim, ff_dim=ff_dim, num_heads=num_heads, 
                            key_dim=key_dim, dropout_rate=dropout_rate)
            for _ in range(decoder_layers)
        ]
        
        # self.pooling = layers.GlobalAveragePooling1D()
        self.final_layer = layers.Dense(self.vocab_size, activation="softmax")


        ## Build the models
        # Encoder
        input_sequence = layers.Input(shape=(), dtype="string")
        tokens = self.tokenizing_layer(input_sequence)
        float_tokens = tf.cast(tokens, "float32")
        masked_tokens = self.token_masker(float_tokens)
        enc_z = self.embedding_layer(masked_tokens)
        for enc_layer in self.encoder_layers:
            enc_z = enc_layer(enc_z)
        self.encoder = Model(inputs=[input_sequence], outputs=[enc_z], name="encoder")

        # Decoder
        input_latent = layers.Input(shape=(None, embedding_dim))
        dec_z = layers.Identity()(input_latent)
        for dec_layer in self.decoder_layers:
            dec_z = dec_layer(dec_z)
        self.decoder = Model(inputs=[input_latent], outputs=[dec_z], name="decoder")

        # Tokenizer
        self.tokenizer = Model(inputs=[input_sequence], outputs=[tokens], name="tokenizer")

        # Autoencoder
        z = self.encoder(input_sequence)
        x_hat = self.decoder(z)
        x_hat = self.final_layer(x_hat)
        self.autoencoder = Model(inputs=[input_sequence], outputs=[x_hat], name="gene_autoencoder")

        ## Compile the model
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.levenshtein_metric = LevenshteinDistance(self.vocabulary)
        track_metrics = ["acc",
                         self.levenshtein_metric]

        # self.autoencoder.compile(optimizer=opt, loss="categorical_crossentropy", metrics=track_metrics)
        self.build(input_shape=input_sequence.shape)
        self.compile(optimizer=self.opt, loss="categorical_crossentropy", metrics=track_metrics)


    @tf.function
    def codon_splitter(self, x):
        # Determine the lengths of the input sequences
        _str_lens = tf.strings.length(x)
        maxlen = tf.reduce_max(_str_lens)

        # Initialize arrays defining codon start indices
        _range = tf.range(0,maxlen,3, dtype="int32")[:,None]
        _pos = tf.repeat(_range, repeats=tf.shape(x), axis=1)

        # Mask out positions and lengths greater than a given sequence length
        mask = tf.less(_pos, _str_lens)
        _pos = tf.multiply(_pos, tf.cast(mask, "int32"))
        _len = tf.multiply(3, tf.cast(mask, "int32"))
        codons = tf.strings.substr(x, _pos, _len)
        return tf.transpose(codons)
    
    
    def tokenize(self, x, one_hot=True):
        ## Convert a DNA sequence to a sequence of tokens, optionally one-hot encoded
        tokens = self.tokenizer(x)
        if one_hot:
            return tf.one_hot(tokens, depth=self.vocab_size)
        else:
            return tokens
        
    def untokenize(self, x, one_hot=True, join=True):
        ## Convert a sequence of tokens to a DNA sequence
        if one_hot:
            x = tf.argmax(x, axis=-1, output_type="int32")
        genes = self.lookup_table.lookup(x)
        if join:
            genes = tf.strings.reduce_join(genes, axis=-1)
        return genes


    def call(self, x):
        return self.autoencoder(x)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def get_config(self):
        base_config = super().get_config()
        layer_config = {_l.name:serialize_keras_object(_l) for _l in self.layers}
        aux_config = self.levenshtein_metric.get_config()
        return {**base_config, **layer_config, **aux_config}

    def train(self, data:tf.data.Dataset, val_data:tf.data.Dataset,
              batch_size:int, epochs:int, *callbacks):

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



class LevenshteinDistance(tf.keras.metrics.Metric):

    def __init__(self, vocabulary, name="mean_levenshtein_distance", **kwargs):
        super().__init__(name=name, **kwargs)
        
        ## Define a lookup table for converting back to sequences
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=range(self.vocab_size),
                values=self.vocabulary,
                key_dtype="int32",
                value_dtype=tf.string
            ),
            default_value=self.vocabulary[0]
        )

        ## Define the distance variables
        self.levenshtein_dist = self.add_variable(
            shape=(),
            initializer='zeros',
            name='levenshtein_distance'
        )

    def untokenizer(self, x):
        ## Convert a sequence of tokens to a DNA sequence
        x = tf.argmax(x, axis=-1, output_type="int32")
        genes = self.lookup_table.lookup(x)
        return genes


    def update_state(self, y_true, y_pred, **kwargs):
        tokens_true = self.untokenizer(y_true)
        tokens_pred = self.untokenizer(y_pred)
        sparse_true = tf.sparse.from_dense(tokens_true)
        sparse_pred = tf.sparse.from_dense(tokens_pred)
        dists = tf.edit_distance(sparse_pred, sparse_true)
        self.levenshtein_dist.assign(tf.reduce_mean(dists))

    def result(self):
        return self.levenshtein_dist
    
    def get_config(self):
        base_config = super().get_config()
        aux_config = {"vocabulary":self.vocabulary, "name":self.name}
        return {**base_config, **aux_config}
    
    @classmethod
    def from_config(cls, config):
        vocabulary = config.pop("vocabulary")
        name = config.pop("name")
        return cls(vocabulary, name, **config)



class TransformerBlock(layers.Layer):
    '''
    Transformer model for whole genome assemblies.
    '''
    def __init__(self, output_dim:int, ff_dim:int, num_heads:int, key_dim:int, dropout_rate:float=0.1, **kwargs):
        super().__init__(**kwargs)
        
        ## Define needed paramters
        self.output_dim = output_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate

        ## Attention layer
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        ## Normalization layers
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        ## Feed forward layers
        self.ff1 = layers.Dense(ff_dim, activation="relu")
        self.ff2 = layers.Dense(output_dim)

        ## Dropout layers
        if dropout_rate != 0:
            self.dropout1 = layers.Dropout(rate=dropout_rate)
            self.dropout2 = layers.Dropout(rate=dropout_rate)


    def call(self, inputs):
        ## Multi-head attention
        attn_output = self.attn(inputs, inputs)
        if self.dropout_rate != 0:
            attn_output = self.dropout1(attn_output)

        ## Add and norm
        out1 = self.norm1(inputs + attn_output)

        ## Feed-forward
        ffn_output = self.ff1(out1)
        if self.dropout_rate != 0:
            ffn_output = self.dropout2(ffn_output)
        ffn_output = self.ff2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)
        return out2

    def build(self, input_shape):
        return
    
    def compute_output_shape(self, input_shape):
        b,s,_ = input_shape
        return (b, s, self.ff_dim)
    
    # @classmethod
    # def from_config(cls, config):
    #     attn_config = config.pop("multihead-attention")
    #     sublayer = tf.keras.utils.deserialize_keras_object(sublayer_config)
    #     return cls(sublayer, **config)
    
    # @classmethod
    def get_config(self):
        base_config = super().get_config()
        config = {
            "output_dim":self.output_dim,
            "ff_dim":self.ff_dim,
            "num_heads":self.num_heads,
            "key_dim":self.key_dim,
            "dropout_rate":self.dropout_rate,
            "multihead-attention":serialize_keras_object(self.attn),
            "norm1":serialize_keras_object(self.norm1),
            "norm2":serialize_keras_object(self.norm2),
            "feedforward1":serialize_keras_object(self.ff1),
            "feedforward2":serialize_keras_object(self.ff2),
            "dropout1":serialize_keras_object(self.dropout1),
            "dropout2":serialize_keras_object(self.dropout2)
        }
        return {**base_config, **config}
    



def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, None]     # (seq, 1)
    depths = np.arange(depth)[None, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)





class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, max_length=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=max_length, depth=embedding_dim)

    # def compute_mask(self, *args, **kwargs):
    #     return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


