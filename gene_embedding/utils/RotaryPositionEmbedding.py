import tensorflow as tf
import keras
import numpy as np


class RotaryPositionEncoding(keras.layers.Layer):
    '''
    Rotational position encoding according to "ROFORMER: ENHANCED TRANSFORMER WITH ROTARY
    POSITION EMBEDDING" -- https://arxiv.org/pdf/2104.09864
    '''

    def __init__(self, num_positions: int, 
                 embedding_dim: int, 
                 **kwargs):
        super().__init__(**kwargs)

        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")

        self.embedding_dim = embedding_dim
        self.num_positions = num_positions
        self.supports_masking = True

        ## Define the angle vector
        d_idx = tf.range(self.embedding_dim//2)
        theta = 10000 ** ((-1*d_idx)/self.embedding_dim)
        reps = tf.fill(theta.shape, 2)
        theta = tf.repeat(theta, reps)

        ## Define the position vector
        m = tf.range(self.num_positions, dtype=theta.dtype)

        ## Define the positonally encoded angles with an outer product
        self.m_theta = tf.einsum("s,d->sd", m,theta)

    # def compute_output_shape(self, input_shape):
    #     # return super().compute_output_shape(*args, **kwargs)
    #     return input_shape

    def compute_mask(self, inputs, mask=None):
        return super().compute_mask(inputs, mask)

    def _permute_for_rotation(self, x):
        '''
        Permute an input vector for efficient rotation
            [x1,x2,x3,x4,...] -> [-x2,x1,-x4,x3,...]
        '''

        ## Define the alternating indices array
        idx = tf.range(self.embedding_dim, dtype=tf.int32)
        evs = idx%2==0
        ods = idx%2!=0
        idx = (idx+1)*tf.cast(evs, idx.dtype) + (idx-1)*tf.cast(ods, idx.dtype)

        ## Define the alternating signs array
        sgn = tf.ones(self.embedding_dim, dtype=x.dtype)
        sgn = sgn*tf.cast(evs,sgn.dtype) - sgn*tf.cast(ods,sgn.dtype)

        ## Apply the alternating indices and signs
        return tf.gather( tf.multiply(x,sgn), idx, axis=2)


    def call(self, x: tf.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        # print("\nROTARY DEBUGGING:")
        seq_len = tf.shape(x)[1]
        _angles = tf.slice(self.m_theta, begin=[0,0], size=[seq_len,self.embedding_dim])[None,...]
        _angles = tf.cast(_angles, x.dtype)
        x_shuffle = self._permute_for_rotation(x)
        _t1 = tf.multiply(x, tf.math.cos(_angles))
        _t2 = tf.multiply(x_shuffle,tf.math.sin(_angles))
        return tf.add(_t1, _t2)
    