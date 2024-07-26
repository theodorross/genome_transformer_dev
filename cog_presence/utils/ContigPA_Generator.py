import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import math


class ContigPA_Generator(keras.utils.Sequence):

    def __init__(self, data_df:pd.DataFrame, label_df:pd.DataFrame, batch_size=8, max_contigs=100, 
                 shuffle_sequences=True, shuffle_contigs=True, random_state=2341,
                 **kwargs):
        super().__init__(**kwargs)

        self.data_df = data_df
        self.label_df = label_df
        self.batch_size = batch_size
        self.max_contigs = max_contigs
        self.shuffle_contigs = shuffle_contigs
        self.suffle_sequences = shuffle_sequences
        self.random_state = random_state
        self.sequences = label_df.index.tolist()

        self.n_batches = math.ceil(len(self.sequences) / self.batch_size)

        np.random.seed(random_state)


    def on_epoch_end(self):
        '''Shuffle the order of the sequences on epoch end'''
        if self.suffle_sequences:
            np.random.shuffle(self.sequences)

    
    def __len__(self):
        '''Return the number of batches'''
        return self.n_batches
    
    def __get_sequence(self, querry_seq):
        '''Return individual sample'''
        ## Get the data array
        # querry_seq = self.sequences[index]
        arr = self.data_df.loc[querry_seq,:].to_numpy()

        ## Shuffle the rows of the data array if desired
        if self.shuffle_contigs:
            np.random.shuffle(arr)

        ## Pad the data array to the max sequence length
        if arr.shape[0] < self.max_contigs:
            pad_width = self.max_contigs - arr.shape[0]
            zero_arr = np.zeros((pad_width, arr.shape[1]))
            arr = np.concatenate([arr, zero_arr], axis=0)
        elif arr.shape[0] > self.max_contigs:
            raise("OOPSIE DOODLE NEED TO INCREASE 'max_contigs'")
        
        return arr

    
    def __getitem__(self, index):
        '''Return batch corresponding to "index"'''
        ## Get the sequence ids for the current batch
        lo_idx = index*self.batch_size
        hi_idx = min(lo_idx+self.batch_size, len(self.sequences))
        querry_seqs = self.sequences[lo_idx:hi_idx]

        ## Retrieve the data arrays for each sequence
        arr = [self.__get_sequence(q) for q in querry_seqs]
        arr = np.stack(arr, axis=0)

        ## Get the labels array
        label = self.label_df.loc[querry_seqs].to_numpy()

        return tf.convert_to_tensor(arr), tf.convert_to_tensor(label)


    
