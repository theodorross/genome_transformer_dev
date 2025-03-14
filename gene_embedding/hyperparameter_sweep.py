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



def train_gene_ae(config):

    '''
    Parse the input configuration
    '''
    ## parse the model configuration
    model_config = {
        'tokenization_method': config.get("tokenization_method", 'nucleotide'),
        'embedding_dim': config['embedding_dim'],
        'latent_dim': config['latent_dim'],
        'encoder_layers': config['encoder_layers'],
        'decoder_layers': config['decoder_layers'],
        'key_dim': config["key_dim"],
        'num_heads': config['num_heads'],
        'dropout_rate': config['dropout_rate'],
        'ff_dim': config['latent_dim'],             ## Ensure latent_dim and ff_dim are equal
        'max_length' :config['max_length'],
        'decode_length': config['decode_length'],
        'masking_rate': config['masking_rate'],
        'learning_rate': config['learning_rate'],
        'n_sequence_tokens': config['n_sequence_tokens'],
    }

    '''
    Load the gene data
    '''
    ## Define the data path depending ont the dataset and platform
    platform = config.get('platform', 'LUMI')
    if platform.lower() == "local":
        data_dir = "../data/gene_sequences"
    if platform.lower() == "springfield":
        data_dir = "/storage/data/e_faecium/gene_embedding"
    if platform.upper() == "LUMI":
        data_dir = "/project/project_465001381/rosstheo/genome_transformer_dev/data/gene_sequences"

    ## Load the dataset and remove genes over the max sequence length
    datapath = f"{data_dir}/train_unique_gene_seqs.txt"
    gene_dataset = tf.data.TextLineDataset(datapath)
    gene_dataset = gene_dataset.map(clip_gene(config["max_length"]))


    '''
    Define training callbacks
    '''
    wandb_callback = wandb.keras.WandbMetricsLogger()
    early_stopper = keras.callbacks.EarlyStopping(patience=config['patience'],
                                                  restore_best_weights=True)
    callbacks = [early_stopper, wandb_callback]

    if (config['learning_rate_decay'] is not None) and (config['learning_rate_decay_start'] is not None):
        # schedule_func = lambda e,lr: lr*config['learning_rate_decay'] if (e%50==49 and e>config['learning_rate_decay_start']) else lr*1.0
        def schedule_func(e, lr):
            if (config['learning_rate_decay'] is not None) and (config['learning_rate_decay_start'] is not None):
                if (e%50==49) and (e>config['learning_rate_decay_start']):
                    lr*config['learning_rate_decay']
                else:
                    return lr*1.0
            else:
                return lr*1.0
    
        lr_scheduler = keras.callbacks.LearningRateScheduler(schedule_func)
        callbacks.append(lr_scheduler)

    ## Define incrementing gene lengths for training
    decode_lengths = np.linspace(config['decode_length'], config['max_length'],
                                 config['seq_length_steps'], dtype=int)
    

    '''
    Define and train the model
    '''
    ## Define the model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        gene_ae = GeneTransformer(**model_config)

    ## Split the dataset into training and validation
    dataset_cuts = [gene_dataset.shard(5, k) for k in range(5)]
    validation_genes = dataset_cuts[-1]
    training_genes = dataset_cuts[:-1]
    training_genes = concatenate_datasets(*training_genes)

    ## Loop through the desired gene lengths
    _epoch_count = 0
    history = {}
    for ix,gene_length in enumerate(decode_lengths):

        ## Filter the dataset by the specified gene length
        _training = training_genes.filter(lambda x: tf.strings.length(x) < gene_length)
        _validation = validation_genes.filter(lambda x: tf.strings.length(x) < gene_length)

        ## Preprocess the data
        _training, _validation = gene_ae.preprocess_dataset(_training,
                                                            config['batch_size'],
                                                            _validation)
        
        ## Fit the model to the data
        _epochs = config['epochs'] // len(decode_lengths)        # number of epochs per decode length
        _hist = gene_ae.fit(_training, validation_data=_validation, epochs=_epochs*(ix+1),
                            callbacks=callbacks, verbose=1, initial_epoch=_epoch_count)
        _epoch_count += len(_hist.history['loss'])

        ## Store the training history
        for key,val in _hist.history.items():
            if key in history.keys():
                history[key] += val
            else:
                history[key] = val

    ## Save the model
    gene_ae.save(f"models/geneAE_{wandb.run.name}.keras")

    ## Return the latest validation loss 
    # return history['val_loss'][-1]
    _eval = gene_ae.evaluate(_validation, return_dict=True)
    print("Final validation loss:", _eval['loss'])
    print("\n\n")
    return _eval["loss"]


def main():
    wandb.init(project="gene-encoder")
    val_loss = train_gene_ae(wandb.config)
    wandb.log({"val_loss":val_loss})



if __name__ == "__main__":
    keras.config.disable_traceback_filtering()

    sweep_config = {
        "method":"random",
        "metric": {'goal':'minimize', 'name':'val_loss'},
        "parameters":{
            "embedding_dim": {'values': [4,8]},
            'latent_dim': {'values': [16, 32, 64]},
            'encoder_layers': {'values': [4, 8, 10, 20]},
            'decoder_layers': {'values': [1, 2, 4]},
            'key_dim': {'values': [16, 32]},
            'num_heads': {'values': [12, 24]},
            'dropout_rate': {'values': [0.0]},
            'max_length': {'values': [300]},
            'decode_length': {'values': [300]},
            'masking_rate': {'values': [0.0]},
            'learning_rate': {'values': [1e-3]},
            "n_sequence_tokens": {'values': [2,4,8,16]},
            'patience': {'values': [100]},
            'batch_size': {'values': [256]},
            'epochs': {'values': [2000]},
            'learning_rate_decay': {'values':[0.95]},
            'learning_rate_decay_start': {'values': [100, 200]},
            'seq_length_steps': {'values': [1]}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project="gene-encoder")
    wandb.agent(sweep_id, function=main, count=2)


    # ## parse the model configuration
    # model_config = {
    #     'tokenizing_method': config.get("tokenizing_method", 'nucleotide'),
    #     'embedding_dim': config['embedding_dim'],
    #     'latent_dim': config['latent_dim'],
    #     'encoder_layers': config['encoder_layers'],
    #     'decoder_layers': config['decoder_layers'],
    #     'key_dim': config["key_dim"],
    #     'num_heads': config['num_heads'],
    #     'dropout_rate': config['dropout_rate'],
    #     'ff_dim': config['ff_dim'],
    #     'max_length' :config['max_length'],
    #     'decode_length': config['decode_length'],
    #     'masking_rate': config['masking_rate'],
    #     'learning_rate': config['learning_rate'],
    #     'n_sequence_tokens': config['n_sequence_tokens'],
    # }

    # ## Parse the training configuration
    # training_config = {
    #     "patience": config['patience'],
    #     # "cross_folds": config['cross_folds'],
    #     "batch_size": config['batch_size'],
    #     "epochs": config['epochs'],
    #     "learning_rate_decay": config['learning_rate_decay'],
    #     "learning_rate_decay_start": config['learning_rate_decay_start'],
    #     "seq_length_steps": config['seq_length_steps']
    # }