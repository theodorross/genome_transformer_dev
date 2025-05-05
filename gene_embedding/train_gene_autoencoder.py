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

from utils.GeneTransformer import GeneTransformer
from utils.TrainingUtils import MaskedSparseCategoricalCrossentropy, MaskedAccuracy, OnlineMarginTripletLoss
# from utils import *

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


if __name__ == "__main__":
    keras.config.disable_traceback_filtering()

    '''
    Initialize run parameters
    '''
    parser = argparse.ArgumentParser()
    ## System parameters
    parser.add_argument("--platform", default="local", type=str, help="Hardware platform used for job training.", required=False)
    parser.add_argument("--dataset", default="full", choices=["dev","full"], type=str, help="Which datset to use.", required=False)

    ## Model architecture hyperparameters
    parser.add_argument("--tokenization", default="nucleotide", choices=["nucleotide","codon"], type=str, help="Units to tokenize for processing. One of ['nucleotide','codon'].", required=False)
    parser.add_argument("--latent-dim", "-z", default=32, type=int, help="Dimensionality of the representation space.", required=False)
    parser.add_argument("--embedding-dim", default=8, type=int, help="Dimensionality of token embedding vectors.", required=False)
    parser.add_argument("--encoder-layers", default=2, type=int, help="Number of transformer-blocks in the encoder.")
    parser.add_argument("--decoder-layers", default=1, type=int, help="Number of transformer-blocks in the decoder.")
    parser.add_argument("--key-dim", default=16, type=int, help="Dimension of the key, query, and value vectors in multi-head attention units.")
    parser.add_argument("--num-heads", default=8, type=int, help="Number of attention heads in multi-head attention units.")
    parser.add_argument("--dropout-rate", default=0.1, type=float, help="Dropout rate used in feed-forward layers.")
    parser.add_argument("--ff-dim", default=32, type=int, help="Dimensionality of the hidden feed-forward layer in the transformer blocks.")
    parser.add_argument("--n-sequence-tokens", default=4, type=int, help="Number of latent sequence tokens to use.")
    parser.add_argument("--decode-length", default=100, type=int, help="Number of sequence tokens to use during reconstruction at the beginning of training.")
    parser.add_argument("--masking-rate", default=0.05, type=float, help="Probability of masking each input token during training.")
    parser.add_argument("--learning-rate", default=1e-6, type=float, help="Learning rate for the optimizer.", required=False)
    parser.add_argument("--max-seq-length", default=100, type=int, help="Maximum sequence length to use during training.")

    ## Training hyperparameters
    parser.add_argument("--learning-rate-decay", default=None, type=float, help="Decay rate for learning rate schedule.")
    parser.add_argument("--learning-rate-decay-start", default=None, type=float, help="Epoch to begin learning rate decay.")
    parser.add_argument("--batch-size", "-b", default=4, type=int, help="Batch size for training.", required=False)
    parser.add_argument("--epochs", default=100, type=int, help="Maximum number of training epochs to perform.", required=False)
    parser.add_argument("--patience", default=25, type=int, help="Patience for early stopping.", required=False)
    parser.add_argument("--cross-folds", default=5, type=int, help="Number of cross-folds for validation of training.", required=False)
    parser.add_argument("--seq-length-steps", default=1, type=int, help="Number of linear steps for increasing the sequence length from decode-length to max-seq-length.")
    args = parser.parse_args()

    '''
    Initialize wandb
    '''
    sys_config = {"platform":args.platform,
                  "dataset":args.dataset}
    
    model_config = {"tokenization_method":args.tokenization,
                    "latent_dim":args.latent_dim,
                    "embedding_dim":args.embedding_dim,
                    "encoder_layers":args.encoder_layers,
                    "decoder_layers":args.decoder_layers,
                    "key_dim":args.key_dim,
                    "num_heads":args.num_heads,
                    "dropout_rate":args.dropout_rate,
                    "ff_dim":args.ff_dim,
                    "max_length":args.max_seq_length,
                    "masking_rate":args.masking_rate,
                    "learning_rate":args.learning_rate,
                    "n_sequence_tokens":args.n_sequence_tokens,
                    "decode_length":args.decode_length}
    
    training_config = {"patience":args.patience,
                       "cross_folds":args.cross_folds,
                       "batch_size":args.batch_size,
                       "epochs":args.epochs,
                       "learning_rate_decay":args.learning_rate_decay,
                       "learning_rate_decay_start":args.learning_rate_decay_start,
                       "seq_length_steps":args.seq_length_steps}
    

    ## Instantiate the multi-device training strategy
    # strategy = tf.distribute.MirroredStrategy()
    # if args.batch_size % strategy.num_replicas_in_sync != 0:
    #     raise ValueError(f"batch_size must be evenly divisible by the number of devices in use:\n\tbatch_size: {args.batch_size}\n\tnumber of devices: {strategy.num_replicas_in_sync}")
    # print(f"\nNumber of devices: {strategy.num_replicas_in_sync}\n")

    ## Initialize wandb
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
    # if args.platform.lower() == "local":
    #     data_dir = "../data/gene_sequences"
    # if args.platform.lower() == "springfield":
    #     data_dir = "/storage/data/e_faecium/gene_embedding"
    # if args.platform.upper() == "LUMI":
    #     data_dir = "/project/project_465001381/rosstheo/genome_transformer_dev/data/gene_sequences"
    
    # if args.dataset.lower() == "dev":   
    #     datapath = f"{data_dir}/unique_dna_seqs_dev.txt"
    # elif args.dataset.lower() == "full":
    #     datapath = f"{data_dir}/train_unique_gene_seqs.txt"


    ## Load the dataset and remove genes over the max sequence length
    # gene_dataset = tf.data.TextLineDataset(datapath)
    gene_dataset = tf.data.Dataset.load("../data/gene_sequences/training_dataset")
    if args.dataset == "full":
        gene_dataset = gene_dataset.filter(lambda g,d,c: tf.strings.length(g) <= args.max_seq_length).cache()
    elif args.dataset == "dev":
        print("'dev' dataset is depreciated.")
        exit()
        gene_dataset = gene_dataset.map(clip_gene(args.max_seq_length))

    ## Cut the dataset into k cross-folds
    dataset_cuts = [gene_dataset.shard(args.cross_folds, k) for k in range(args.cross_folds)]


    '''
    Define training callbacks
    '''
    wandb_callback = wandb.keras.WandbMetricsLogger()
    early_stopper = keras.callbacks.EarlyStopping(patience=args.patience,
                                                  restore_best_weights=True)
    callbacks = [wandb_callback, early_stopper]

    if (args.learning_rate_decay is not None) and (args.learning_rate_decay_start is not None):
        if args.dataset.lower() == "dev":
            schedule_func = lambda e,lr: lr*0.95 if (e%50==49 and e>args.learning_rate_decay_start) else lr*1.0
        else:
            schedule_func = lambda e,lr: lr*0.95 if (e%50==49 and e>args.learning_rate_decay_start) else lr*1.0
        lr_scheduler = keras.callbacks.LearningRateScheduler(schedule_func)
        callbacks.append(lr_scheduler)

    '''
    Define a new model to train for each cross-validation fold
    '''
    ## Define incrementing gene lengths for training
    decode_lengths = np.linspace(args.decode_length, args.max_seq_length, args.seq_length_steps, dtype=int)

    ## Initialize a dictionary for storing training histories of each fold
    training_histories = {}

    ## Loop through training folds
    for k in range(args.cross_folds):
        print()

        ## Define the training and test datsets
        validation_fold = dataset_cuts[k]
        training_fold = dataset_cuts[:k] + dataset_cuts[k+1:]
        training_fold = concatenate_datasets(*training_fold)

        ## Initialize the model
        # with strategy.scope():
        #     gene_ae = GeneTransformer(**model_config)
        gene_ae = GeneTransformer(**model_config)
        print(gene_ae.summary())
        

        ## Initialize a training history
        fold_history = {}

        ## Loop through the desired gene lengths
        _epoch_count = 0
        for ix,gene_length in enumerate(decode_lengths):
            print(f"\nTraining on genes of {gene_length} tokens and smaller...")
            gene_ae.update_decode_length(gene_length)

            ## Filter the datasets
            _training_fold = training_fold.filter(lambda g,d: tf.strings.length(g) < gene_length)
            _validation_fold = validation_fold.filter(lambda g,d: tf.strings.length(g) < gene_length)

            ## Preprocess the datasets
            _training_fold, _validation_fold = gene_ae.preprocess_dataset(_training_fold, 
                                                                          batch_size=args.batch_size, 
                                                                          validation_data=_validation_fold)
            
            # count = 0
            # for tup in _training_fold:
            #     print(len(tup), tup[0].shape, (tup[1][0].shape, tup[1][1].shape), tup[2].shape)
            #     count += 1
            #     if count > 5:
            #         break

            ## Train the model
            _epochs = args.epochs // len(decode_lengths)        # number of epochs per decode length
            _hist = gene_ae.fit(_training_fold, validation_data=_validation_fold, epochs=_epochs*(ix+1), 
                                callbacks=callbacks, verbose=1, initial_epoch=_epoch_count)
            print(_hist.history.keys())
            _epoch_count += len(_hist.history["loss"])
            
            ## Store the training history
            for key,val in _hist.history.items():
                if key in fold_history.keys():
                    fold_history[key] += _hist[key]
                else:
                    fold_history[key] = _hist[key]

            ## Save the interstitial model after each step up in size
            if not os.path.exists(f"models/geneAE_{wandb.run.name}_fold{k}"):
                os.mkdir(f"models/geneAE_{wandb.run.name}_fold{k}")
            # os.mkdir(f"models/geneAE_{wandb.run.name}_fold{k}/{gene_length}_tokens")
            gene_ae.save(f"models/geneAE_{wandb.run.name}_fold{k}/{gene_length}_tokens.keras")


        ## Print a sample reconstruction
        print("Training reconstructions")
        sample_genes = next(training_fold.batch(5).as_numpy_iterator())
        sample_preds = gene_ae.predict(sample_genes)
        recon_tokens = np.argmax(sample_preds, axis=-1)
        recon_chars = np.asarray(gene_ae.encoder.vocabulary)[recon_tokens]
        recon_genes = ["".join(recon_chars[ix]).upper() for ix in range(sample_genes.shape[0])]

        for ix in range(5):
            print()
            recon_str = ""
            for t,r in zip(sample_genes[ix].decode("ASCII"), recon_genes[ix]):
                if t != r: recon_str += f"\033[0;31m{r}\033[0m"
                else: recon_str += f"\033[0;32m{r}\033[0m"
            print(sample_genes[ix].decode("ASCII"))
            print(recon_str)



        print("\nValidation reconstructions")
        sample_genes = next(validation_fold.batch(5).as_numpy_iterator())
        sample_preds = gene_ae.predict(sample_genes)
        recon_tokens = np.argmax(sample_preds, axis=-1)
        recon_chars = np.asarray(gene_ae.encoder.vocabulary)[recon_tokens]
        recon_genes = ["".join(recon_chars[ix]).upper() for ix in range(sample_genes.shape[0])]

        for ix in range(5):
            print()
            recon_str = ""
            for t,r in zip(sample_genes[ix].decode("ASCII"), recon_genes[ix]):
                if t != r: recon_str += f"\033[0;31m{r}\033[0m"
                else: recon_str += f"\033[0;32m{r}\033[0m"
            print(sample_genes[ix].decode("ASCII"))
            print(recon_str)

        ## Store the training history
        training_histories[f"Fold {k}"] = fold_history

        ## Save the model
        # if not os.path.exists(f"models/geneAE_{wandb.run.name}_fold{k}"):
        #     os.mkdir(f"models/geneAE_{wandb.run.name}_fold{k}")
        gene_ae.save(f"models/geneAE_{wandb.run.name}_fold{k}.keras")

        
        break

    print()
    wandb.finish()

    ## Save the histories
    # with open(f"training_histories/geneAE_{wandb.run.name}.json","w") as f:
    #     json.dump(training_histories, f)

