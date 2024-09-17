import tensorflow as tf
import numpy as np
from keras import layers


@tf.function
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



@tf.keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, max_length=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length

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
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "vocab_size":self.vocab_size,
            "embedding_dim":self.embedding_dim,
            "max_length":self.max_length
        }
        return {**base_config, **config}
    

