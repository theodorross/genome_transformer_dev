import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import math


class GeneSeq_Generator(keras.utils.Sequence):


    def __init__(self, file_path, token_type, **kwargs):
        super().__init__(**kwargs)
        self.token_type = token_type
        
