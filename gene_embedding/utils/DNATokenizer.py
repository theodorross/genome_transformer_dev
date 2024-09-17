import tensorflow as tf
import numpy as np
from keras import layers
import itertools
import pandas as pd



@tf.keras.saving.register_keras_serializable()
class DNATokenizer(layers.Layer):

    def __init__(self, tokenization_method, **kwargs):
        super().__init__(**kwargs)

        ## Enfore supported tokenization method
        assert tokenization_method.lower() in {"nucleotide","codon"}
        self.tokenization_method = tokenization_method

        ## Define vocabularies
        if tokenization_method.lower() == "nucleotide":
            vocab = ['a','t','c','g']
            split_method = "character"
        elif tokenization_method.lower() == "codon":
            vocab = ["".join(c) for c in itertools.product("atcg", repeat=3)]
            split_method = self.codon_splitter

        ## Define the layer
        self.tokenizing_layer = layers.TextVectorization(split=split_method, vocabulary=vocab)
        self.vocabulary = self.tokenizing_layer.get_vocabulary()


    def call(self, x):
        return self.tokenizing_layer(x)
    
    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "tokenization_method":self.tokenization_method
        }
        return {**base_config, **config}


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


