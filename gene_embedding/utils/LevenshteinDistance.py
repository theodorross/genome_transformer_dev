import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
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
        self.levenshtein_dist = self.add_weight(
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
        config = {
            "vocabulary":self.vocabulary, 
            "name":self.name
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        vocabulary = config.pop("vocabulary")
        name = config.pop("name")
        return cls(vocabulary, name, **config)