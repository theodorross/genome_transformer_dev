import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class LevenshteinDistance(tf.keras.metrics.Metric):

    def __init__(self, vocabulary, mask_token=0, name="mean_levenshtein_distance", **kwargs):
        super().__init__(name=name, **kwargs)
        
        ## Define a lookup table for converting back to sequences
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.mask_token = mask_token
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
            name='levenshtein_dist',
            dtype=tf.float32
        )

    def untokenizer(self, x, from_one_hot=False):
        ## Convert a sequence of tokens to a DNA sequence
        if from_one_hot:
            x = tf.argmax(x, axis=-1, output_type="int32")
        else:
            x = tf.cast(x, tf.int32)
        genes = self.lookup_table.lookup(x)
        return genes

    def update_state(self, y_true, y_pred, **kwargs):
        ## Compute a mask from the labels
        mask = tf.not_equal(y_true, self.mask_token)

        ## Mask the predictions
        y_pred = tf.argmax(y_pred, axis=2)
        mask = tf.cast(mask, y_pred.dtype)
        y_pred = tf.multiply(y_pred, mask)

        ## Convert the predictions and labels to text
        tokens_true = self.untokenizer(y_true)
        tokens_pred = self.untokenizer(y_pred)

        ## Convert the text to sparse matrices
        sparse_true = tf.sparse.from_dense(tokens_true)
        sparse_pred = tf.sparse.from_dense(tokens_pred)

        ## Compute the Levenshtein distance and normalize it by the number of replicas
        dists = tf.edit_distance(sparse_pred, sparse_true)
        # try:
        #     n_replicas = tf.distribute.get_replica_context().num_replicas_in_sync
        #     self.levenshtein_dist.assign(tf.reduce_mean(dists) / n_replicas)
        # except:
        self.levenshtein_dist.assign(tf.reduce_mean(dists))

    def result(self):
        return tf.math.multiply( self.levenshtein_dist, 1 )
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "vocabulary":self.vocabulary, 
            "mask_token":self.mask_token,
            "name":self.name
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        vocabulary = config.pop("vocabulary")
        mask_token = config.pop("mask_token")
        name = config.pop("name")
        return cls(vocabulary, mask_token, name, **config)
    

    


@tf.keras.utils.register_keras_serializable()
class MaskedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, mask_category=0, name="masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)

        self._direction = "up"
        self.mask_category = mask_category
        self.acc = self.add_variable(
            shape=(),
            initializer='zeros',
            name='masked_acc',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, **kwargs):
        
        ## Compute predicted categories
        pred = tf.argmax(y_pred, axis=-1)
        label = tf.cast(y_true, pred.dtype)

        ## Find category matches
        matches = tf.equal(label, pred)

        ## Mask the matches
        mask = tf.not_equal(label, self.mask_category)
        matches = tf.logical_and(matches, mask)

        ## Compute the accuracy
        matches = tf.cast(matches, tf.float32)
        mask = tf.cast(mask, tf.float32)

        ## Normalize the accuracy by the number of replicas
        # acc = tf.reduce_sum(matches)/tf.reduce_sum(mask)
        acc = tf.math.divide(tf.reduce_sum(matches), tf.reduce_sum(mask))
        # try:
        #     n_replicas = tf.distribute.get_replica_context().num_replicas_in_sync
        #     self.acc.assign( tf.divide(acc, n_replicas) )
        # except:
        self.acc.assign( acc )

    def reset_state(self):
        self.acc.assign(0)
    
    def result(self):
        return tf.math.multiply_no_nan( self.acc, 1 )
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "mask_category":self.mask_category, 
            "name":self.name
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        mask_category = config.pop("mask_category")
        name = config.pop("name")
        return cls(mask_category, name, **config)





@tf.keras.utils.register_keras_serializable()
class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):

    def __init__(self, mask_category=0, name="masked_sparse_categorical_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mask_category = mask_category

    def call(self, y_true, y_pred):
        ## Compute the base loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)

        ## Mask the computed loss
        mask = tf.not_equal(y_true, self.mask_category)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = tf.multiply(loss, mask)

        ## Compute the mean of the loss across the masked values
        loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "mask_category":self.mask_category, 
            "name":self.name
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        mask_category = config.pop("mask_category")
        name = config.pop("name")
        return cls(mask_category, name, **config)