import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf
import keras
import wandb
import argparse
# import json
# import datetime
# import pickle
# from tensorflow.keras import Layers
from utils import *

print("tensorflow version:", tf.__version__)
print("keras version:", keras.__version__, tf.keras.__version__)
print("wandb version:", wandb.__version__)


def clip_gene(length):
    def clipper(gene):
        return tf.strings.substr(gene, 3, length)
    return clipper



def concatenate_datasets(*datasets) -> tf.data.Dataset:
    ## Function to concatenate a list of dataset objects into one dataset
    out_ds = datasets[0].concatenate(datasets[1])
    if len(datasets) > 2:
        for ds in datasets[2:]:
            out_ds = out_ds.concatenate(ds)
    return out_ds



def train_gene_ae(*config):

    ## parse the model configuration
    model_config = {
        'tokenizing_method': config.get("tokenizing_method", 'nucleotide'),
        'embedding_dim': config['embedding_dim'],
        'latent_dim': config['latent_dim'],
        'encoder_layers': config['encoder_layers'],
        'decoder_layers': config['decoder_layers'],
        'key_dim': config["key_dim"],
        'num_heads': config['num_heads'],
        'dropout_rate': config['dropout_rate'],
        'ff_dim': config['ff_dim'],
        'max_length' :config['max_length'],
        'decode_length': config['decode_length'],
        'masking_rate': config['masking_rate'],
        'learning_rate': config['learning_rate'],
        'n_sequence_tokens': config['n_sequence_tokens'],
    }

    ## Parse the training configuration
    training_config = {
        "patience": config['patience'],
        # "cross_folds": config['cross_folds'],
        "batch_size": config['batch_size'],
        "epochs": config['epochs'],
        "learning_rate_decay": config['learning_rate_decay'],
        "learning_rate_decay_start": config['learning_rate_decay_start'],
        "seq_length_steps": config['seq_length_steps']
    }


    return



if __name__ == "__main__":
    keras.config.disable_traceback_filtering()

    # sweep_config = {
    #     "method":"random",
    #     "metric": {'goal':'minimize', 'name':'val_loss'}
    #     "parameters":{
    #         "n-sequence-tokens"
    #     }
    # }