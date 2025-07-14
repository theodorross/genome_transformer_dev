import tensorflow as tf
import sklearn as sk
import numpy as np
import wandb



def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of shape `[n, m]`.
      mask: 2-D Boolean `Tensor` of shape `[n, m]`.
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, tf.cast(mask, data.dtype)), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums


def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of shape `[n, m]`.
      mask: 2-D Boolean `Tensor` of shape `[n, m]`.
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, tf.cast(mask, data.dtype)), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums


def _masked_random(data, mask, dim=1):

    # Define a linear coordinate vector for each row
    row_idx = tf.linspace(0, tf.shape(mask)[1]-1, tf.shape(mask)[1])
    row_idx = tf.expand_dims( tf.cast(row_idx, dtype=tf.int32), axis=1)

    ## Define a function to randomly choose a masked element from a vector
    def choose_fn(vec):
        logits = tf.cast(vec, tf.float32) / tf.reduce_sum(tf.cast(vec, tf.float32))
        logits = tf.expand_dims(tf.math.log(logits + 1e-10), axis=0)
        choice = tf.random.categorical(logits=logits, num_samples=1, dtype=tf.int32)
        return tf.squeeze(choice)
    
    ## Apply choose_fn to each row of the mask matrix
    col_idx = tf.map_fn(choose_fn, tf.cast(mask, tf.int32))
    col_idx = tf.expand_dims(col_idx, axis=1)

    # Concatenate them into a row/column coordiante array
    coords = tf.concat([row_idx, col_idx], axis=1)

    # Gather the defined indices
    out = tf.gather_nd(data, coords)

    return tf.expand_dims(out, axis=1)



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
    



@tf.keras.utils.register_keras_serializable()
class OnlineMarginTripletLoss(tf.keras.losses.Loss):

    def __init__(self, margin, p_norm=2, name="triplet_clustering", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin
        self.p_norm = p_norm

    def call(self, y_true, y_pred):
        ## Assume a decently large batch size of y_pred
      
        # Compute pairwise distances
        z0 = tf.expand_dims(y_pred, axis=0)
        z1 = tf.expand_dims(y_pred, axis=1)
        z_dif = tf.square( tf.norm(z0-z1, ord=self.p_norm, axis=-1) )

        # Compute masks for samples with shared features
        b0 = tf.expand_dims(y_true, axis=0)
        b1 = tf.expand_dims(y_true, axis=1)
        pos_matches = tf.greater( tf.reduce_sum( tf.multiply(b0, b1), axis=-1), 0 )
        neg_matches = tf.logical_not(pos_matches)

        # Force the diagonal to be zero for the positive match mask
        eye = tf.eye(tf.shape(y_true)[0], dtype=bool)
        inv_eye = tf.logical_not(eye)
        pos_matches = tf.logical_and(pos_matches, inv_eye)

        # Compute the hard positive and negative distances for each anchor
        hard_pos_dists = _masked_maximum(z_dif, pos_matches)
        # hard_neg_dists = _masked_minimum(z_dif, neg_matches)
        rand_neg_dists = _masked_random(z_dif, neg_matches)

        # Only use anchors with a positive match
        pos_idx = tf.where( tf.greater(hard_pos_dists, 0) )
        hard_pos_dists = tf.gather_nd(hard_pos_dists, pos_idx)
        # hard_neg_dists = tf.gather_nd(hard_neg_dists, pos_idx)
        rand_neg_dists = tf.gather_nd(rand_neg_dists, pos_idx)
        
        # Compute the margined difference between the mean postive-pair and negative-pair distances
        # loss = tf.reduce_mean(hard_pos_dists - hard_neg_dists + self.margin)
        loss = tf.reduce_mean(hard_pos_dists - rand_neg_dists + self.margin)
        loss = tf.maximum(loss, 0)
        loss = tf.keras.ops.nan_to_num(loss, nan=0.0)   # just in case there are no positive distances, force the value to 0
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "margin":self.margin,
            "p_norm":self.p_norm, 
            "name":self.name
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        margin = config.pop("margin")
        p_norm = config.pop("p_norm")
        name = config.pop("name")
        return cls(margin, p_norm, name, **config)





@tf.keras.utils.register_keras_serializable()
class MaskedBinaryCrossentropy(tf.keras.losses.Loss):

    def __init__(self, name="masked_binary_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        
        ## Compute the base loss
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)

        ## Mask the computed loss to ignore samples with no predicted category
        cat_sum = tf.reduce_sum(y_true, axis=-1)
        mask = tf.not_equal(cat_sum, 0)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = tf.multiply(loss, mask)

        ## Compute the mean of the loss across the masked values
        n_categs = tf.cast(tf.shape(y_true)[-1], tf.float32)
        loss = tf.divide( tf.reduce_sum(loss), tf.reduce_sum(mask)*n_categs )
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "name":self.name
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        name = config.pop("name")
        return cls(name, **config)
    


@tf.keras.utils.register_keras_serializable()
class MaskedBinaryAccuracy(tf.keras.metrics.Metric):

    def __init__(self, threshold=0.5, name="masked_binary_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)

        self._direction = "up"
        self.threshold = 0.5
        self.acc = self.add_variable(
            shape=(),
            initializer='zeros',
            name='masked_acc',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, **kwargs):
        
        ## Compute predicted categories
        pred = tf.cast(y_pred > self.threshold, y_pred.dtype)
        label = tf.cast(y_true, pred.dtype)

        ## Find category matches
        matches = tf.equal(label, pred)

        ## Mask the matches to ignore samples with zero positive labels
        cat_sum = tf.reduce_sum(y_true, axis=-1, keepdims=True)
        mask = tf.not_equal(cat_sum, 0)
        matches = tf.logical_and(matches, mask)
        matches = tf.reduce_all(matches, axis=-1)

        ## Compute the accuracy
        matches = tf.cast(matches, tf.float32)
        mask = tf.cast(mask, tf.float32)

        ## Normalize the accuracy by the number of replicas
        acc = tf.math.divide(tf.reduce_sum(matches), tf.reduce_sum(mask))
        self.acc.assign( acc )

    def reset_state(self):
        self.acc.assign(0)
    
    def result(self):
        return tf.math.multiply_no_nan( self.acc, 1 )
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "threshold":self.threshold, 
            "name":self.name
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        threshold = config.pop("threshold")
        name = config.pop("name")
        return cls(threshold, name, **config)



class LinearEvaluationProtocol(tf.keras.callbacks.Callback):

    def __init__(self, validation_freq=1, validation_data=None):
        super().__init__()
        self.data = validation_data
        self.validation_freq = validation_freq
        self.acc_metric = MaskedBinaryAccuracy(name=None)
        self.loss_metric = MaskedBinaryCrossentropy(name=None)

    def set_validation_data(self, validation_data):
        self.data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        ## Only perform on validation frequency
        if (epoch+1) % self.validation_freq == 0:

            ## Get embeddings and labels for the entire validation dataset
            z = self.model.encoder.predict(self.data, verbose=0)
            y = np.concatenate([_y[1][1] for _y in self.data], axis=0)
            pred_y = np.zeros(y.shape)

            ## Get linear classification prediction probabilities for each category
            for categ in range(y.shape[1]):
                categ_y = y[:,categ]
                if categ_y.sum() != 0:
                    categ_preds = sk.linear_model.LogisticRegression().fit(z,categ_y).predict_proba(z)[:,1]
                    pred_y[:,categ] = categ_preds

            ## Compute the accuracy and loss following the protocol in MaskedBinaryAccuracy above
            # matches = np.all(pred_y == y, axis=1)   # All predictions for a sample match labels
            # mask = y.sum(axis=1) != 0       # Mask out samples that have no functional predictions
            # matches = np.logical_and(matches, mask)
            # accuracy = np.mean(matches)
            accuracy = self.acc_metric(y, pred_y)
            loss = self.loss_metric(y, pred_y)

            # Store and log the accuracy
            logs["val_COG_category_accuracy"] = accuracy
            logs["val_COG_category_loss"] = loss
            try:
                wandb.log({"epoch/val_COG_category_accuracy": accuracy}, step=epoch)
                wandb.log({"epoch/val_COG_category_loss": loss}, step=epoch)
            except:
                pass






if __name__=="__main__":
    
    # testloss = MaskedBinaryCrossentropy()
    # testacc = MaskedBinaryAccuracy()

    # testprot = LinearEvaluationProtocol()

    # # test_pred = tf.convert_to_tensor([[0.01,0.05,0.9, 0.4, 0.75],[0.4,0.02,0.13,0.58,0.92]])
    # # test_true = tf.convert_to_tensor([[0,0,1,0,1], [0,0,0,0,0]])
    # test_pred = tf.convert_to_tensor([[0.01,0.05,0.9, 0.4, 0.75]])
    # test_true = tf.convert_to_tensor([[0,0,1,0,1]])

    # print("loss:    ", testloss(test_true, test_pred).numpy())
    # print("accuracy:", testacc(test_true, test_pred).numpy())

    # 11581137
    
    # print(testmat)
    # print(mask)
    # print(testmat * tf.cast(mask, testmat.dtype))
    # print(_masked_maximum(testmat, mask))
    # print(_masked_minimum(testmat, mask))
    # print(_masked_random(testmat, notmask))

    # test gradients
    with tf.GradientTape() as tape:
        testmat = tf.random.uniform((5,5))
        mask = tf.convert_to_tensor([[2,1,1,2,3]])
        mask = tf.equal(mask, tf.transpose(mask))
        notmask = tf.logical_not(mask)
        mask = tf.logical_and(mask, tf.logical_not(tf.eye(5, dtype=bool)))

        rnd_choice = _masked_random(testmat, notmask)
        # rnd_choice = _masked_minimum(testmat, notmask)
        print(rnd_choice.shape)
        exit()

        rnd_choice_sum = tf.reduce_sum(rnd_choice)

        print(testmat)
        print(rnd_choice)
        print(rnd_choice_sum)
        grad = tape.gradient(rnd_choice_sum, testmat)
        print("gradient debug:")
        print(grad)

        pass
    
    pass