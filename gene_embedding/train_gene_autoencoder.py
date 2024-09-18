import numpy as np
import tensorflow as tf
import keras
import wandb
import argparse
import json
import os
# import datetime
# import pickle

from utils import *



def concatenate_datasets(*datasets) -> tf.data.Dataset:
    ## Function to concatenate a list of dataset objects into one dataset
    out_ds = datasets[0].concatenate(datasets[1])
    if len(datasets) > 2:
        for ds in datasets[2:]:
            out_ds = out_ds.concatenate(ds)
    return out_ds


if __name__ == "__main__":

    '''
    Initialize run parameters
    '''
    parser = argparse.ArgumentParser()
    ## System parameters
    parser.add_argument("--platform", default="local", type=str, help="Hardware platform used for job training.", required=False)
    parser.add_argument("--dataset", default="full", choices=["dev","full"], type=str, help="Which datset to use.", required=False)

    ## Model architecture hyperparameters
    parser.add_argument("--tokenization", default="nucleotide", choices=["nucleotide","codon"], type=str, help="Units to tokenize for processiing.", required=False)
    parser.add_argument("--embedding-dim", "-z", default=512, type=int, help="Dimensionality of the contig embedding vectors. One of ['nucleotide','codon'],", required=False)
    parser.add_argument("--encoder-layers", default=6, type=int, help="Number of transformer-blocks in the encoder.")
    parser.add_argument("--decoder-layers", default=6, type=int, help="Number of transformer-blocks in the decoder.")
    parser.add_argument("--key-dim", default=64, type=int, help="Dimension of the key, query, and value vectors in multi-head attention units.")
    parser.add_argument("--num-heads", default=8, type=int, help="Number of attention heads in multi-head attention units.")
    parser.add_argument("--dropout-rate", default=0.1, type=float, help="Dropout rate used in feed-forward layers.")
    parser.add_argument("--ff-dim", default=2048, type=int, help="Dimensionality of the hidden feed-forward layer in the transformer blocks.")

    ## Training hyperparameters
    parser.add_argument("--masking-rate", default=0.05, type=float, help="Probability of masking each input token during training.")
    parser.add_argument("--learning-rate", default=1e-6, type=float, help="Learning rate for the optimizer.", required=False)
    parser.add_argument("--batch-size", "-b", default=4, type=int, help="Batch size for training.", required=False)
    parser.add_argument("--epochs", default=100, type=int, help="Maximum number of training epochs to perform.", required=False)
    parser.add_argument("--patience", default=25, type=int, help="Patience for early stopping.", required=False)
    parser.add_argument("--cross-folds", default=5, type=int, help="Number of cross-folds for validation of training.", required=False)
    parser.add_argument("--max-seq-length", default=5000, type=int, help="Maximum sequence length to use during training.")
    args = parser.parse_args()

    '''
    Initialize wandb
    '''
    sys_config = {"platform":args.platform,
                  "dataset":args.dataset}
    
    model_config = {"tokenization_method":args.tokenization,
                    "embedding_dim":args.embedding_dim,
                    "encoder_layers":args.encoder_layers,
                    "decoder_layers":args.decoder_layers,
                    "key_dim":args.key_dim,
                    "num_heads":args.num_heads,
                    "dropout_rate":args.dropout_rate,
                    "ff_dim":args.ff_dim,
                    "max_length":args.max_seq_length,
                    "masking_rate":args.masking_rate,
                    "learning_rate":args.learning_rate}
    
    training_config = {"patience":args.patience,
                       "cross_folds":args.cross_folds,
                       "batch_size":args.batch_size,
                       "epochs":args.epochs}
    
    wandb_config = sys_config | model_config | training_config

    wandb.init(
        project="gene-encoder",
        config=wandb_config,
        sync_tensorboard=True
    )

    '''
    Load the unique gene sequences and get rid of genes longer than 5 kb
    '''
    ## Define the data path depending ont the dataset and platform
    if args.platform.lower() == "local":
        data_dir = "../data/gene_sequences"
    if args.platform.lower() == "springfield":
        data_dir = "/storage/data/e_faecium/gene_embedding"
    
    if args.dataset.lower() == "dev":   
        datapath = f"{data_dir}/unique_dna_seqs_dev.txt"
    elif args.dataset.lower() == "full":
        datapath = f"{data_dir}/train_unique_gene_seqs.txt"


    ## Load the dataset and remove genes over the max sequence length
    gene_dataset = tf.data.TextLineDataset(datapath)
    gene_dataset = gene_dataset.filter(lambda x: tf.strings.length(x) <= args.max_seq_length)

    ## Cut the dataset into k cross-folds
    dataset_cuts = [gene_dataset.shard(args.cross_folds, k) for k in range(args.cross_folds)]


    '''
    Define training callbacks
    '''
    wandb_callback = wandb.keras.WandbMetricsLogger()
    early_stopper = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=args.patience,
                                                  restore_best_weights=True)
    callbacks = [wandb_callback, early_stopper]


    '''
    Define a new model to train for each cross-validation fold
    '''
    training_histories = {}

    for k in range(args.cross_folds):

        ## Define the training and test datsets
        validation_fold = dataset_cuts[k]
        training_fold = dataset_cuts[:k] + dataset_cuts[k+1:]
        training_fold = concatenate_datasets(*training_fold)

        ## Initialize the model
        gene_ae = GeneTransformer(**model_config)

        ## Train the model
        fold_history = gene_ae.train(training_fold, validation_fold, args.batch_size, args.epochs, *callbacks)

        ## Store the training history
        training_histories[f"Fold {k}"] = fold_history.history

        ## Save the model
        os.mkdir(f"models/geneAE_{wandb.run.name}_fold{k}")
        gene_ae.save(f"models/geneAE_{wandb.run.name}_fold{k}")

        # now_str = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")
        # gene_ae.save(f"models/geneAE_{wandb.run.name}_fold{k}_{now_str}.keras")
        # os.mkdir(f"models/geneAE_{wandb.run.name}_fold{k}_{now_str}")
        # gene_ae.save(f"models/geneAE_{wandb.run.name}_fold{k}_{now_str}")

        break



    ## Save the histories
    with open(f"training_histories/geneAE_{wandb.run.name}.json","w") as f:
        json.dump(training_histories, f)

    # now_str = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")
    # with open(f"training_histories/geneAE_{wandb.run.name}_{now_str}.json","w") as f:
    #     json.dump(training_histories, f)

