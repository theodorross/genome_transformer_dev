import tensorflow as tf
import numpy as np
from keras import layers


@tf.function
def positional_encoding(length, depth):
    # depth = depth/2

    positions = np.arange(length)[:, None]              # (seq, 1)
    
    # depths = np.arange(depth)[None, :]/depth    # (1, depth)
    sin_depths = np.arange(depth/2)                     # (1, floor(depth/2))
    sin_angle_rates = 10000**sin_depths                 # (1, floor(depth/2))
    sin_angle_rads = positions / sin_angle_rates        # (pos, floor(depth/2))

    cos_depths = np.arange(depth//2)                    # (1, floor(depth/2) - 1)
    cos_angle_rates = 10000**cos_depths                 # (1, floor(depth/2) - 1)
    cos_angle_rads = positions / cos_angle_rates        # (pos, floor(depth/2) - 1)

    pos_encoding = np.concatenate(
        [np.sin(sin_angle_rads), np.cos(cos_angle_rads)],
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
        print("DEBUG:", self.pos_encoding.shape, (max_length, embedding_dim))

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
    

