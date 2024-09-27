import tensorflow as tf
import numpy as np
import keras
import nltk
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from utils.GeneTransformer import GeneTransformer


def clip_gene(gene):
    # print(gene)
    return tf.strings.substr(gene, 0, 35)


def concatenate_datasets(*datasets) -> tf.data.Dataset:
    ## Function to concatenate a list of dataset objects into one dataset
    out_ds = datasets[0].concatenate(datasets[1])
    if len(datasets) > 2:
        for ds in datasets[2:]:
            out_ds = out_ds.concatenate(ds)
    return out_ds


if __name__ == "__main__":

    '''
    Define the model parameters
    '''
    ## - embedding_dim appears to be the limiting factor. Adding more class tokens seems to help alleviate this
    config = {"embedding_dim":4,
            "encoder_layers":1,
            'decoder_layers':1,
            'key_dim':16,
            'num_heads':16,
            'ff_dim':64,
            'learning_rate':1e-3,
            'n_sequence_tokens':20,
            'max_length':40}


    '''
    Define the dataset
    '''
    ## Load the genes
    gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs_dev.txt")

    ## Shorten them dramatically
    gene_dataset = gene_dataset.map(clip_gene)

    ## Add the class token
    # gene_dataset = gene_ae.preprocess_genes(gene_dataset)
    # gene_dataset = gene_ae.preprocess_genes(gene_dataset)

    ## Cut the datset into training and validation
    dataset_cuts = [gene_dataset.shard(5, k) for k in range(5)]


    '''
    Train a model on each split
    '''
    ## Initialize a dataframe for plotting
    df_list = []

    for k in range(1):
        
        ## Get this fold's data
        val_genes = dataset_cuts[k]
        train_genes = dataset_cuts[:k] + dataset_cuts[k+1:]
        train_genes = concatenate_datasets(*train_genes)

        ## Define the model
        gene_ae = GeneTransformer("nucleotide", **config)
        print(gene_ae.summary())

        ## Train the model
        callback = tf.keras.callbacks.EarlyStopping(patience=20)
        train_history = gene_ae.train(train_genes, val_genes, 8, 50, callback, verbose=2)

        ## Save the training history
        hist_df = pd.DataFrame(train_history)
        hist_df.index.name = "epoch"
        hist_df.reset_index(inplace=True)
        hist_df["fold"] = k
        df_list.append(hist_df)

    ## Combine the training history dataframes
    df = pd.concat(df_list)
    mean_df = df.groupby('epoch').mean()
    min_df = df.groupby('epoch').min()
    max_df = df.groupby('epoch').max()

    ## Plot the training history
    fig,ax = plt.subplots(1,3, figsize=(9,3))
    cols = ['loss','masked_accuracy','mean_levenshtein_distance']
    for c,a in zip(cols,ax):
        a.plot(mean_df.index, mean_df[c], label='training')
        a.plot(mean_df.index, mean_df[f'val_{c}'], label='validation')
        a.fill_between(mean_df.index, max_df[c], min_df[c], label='training', alpha=0.5)
        a.fill_between(mean_df.index, max_df[f'val_{c}'], min_df[f'val_{c}'], label='validation', alpha=0.5)
        a.set_title(c)

    ax[-1].legend()
    filestr = f"z{config['embedding_dim']}-s{config['n_sequence_tokens']}-e{config['encoder_layers']}"
    filestr += f"-d{config['decoder_layers']}-h{config['num_heads']}"
    filestr += f"-k{config['key_dim']}-ff{config['ff_dim']}-lr{config['learning_rate']}"
    # fig.suptitle(f"{filestr}\ncontext")
    fig.suptitle(f"{filestr}\nseqmask")

    plt.tight_layout()
    # plt.savefig(f"figures/small_test_context_{filestr}.png")
    # plt.savefig(f"figures/small_test_seqmask_{filestr}.png")

    '''
    Test the reconstructions
    '''
    ## Isolte the validation genes
    # val_gene_seqs = next(val_genes.batch(60).as_numpy_iterator())
    val_gene_seqs = next(val_genes.batch(5).as_numpy_iterator())
    
    ## Pass through the model
    encodes = gene_ae.encode(val_gene_seqs)
    decodes = gene_ae.decode(encodes)

    ## Reconstruct the sequences
    recon_tokens = np.argmax(decodes, axis=-1)
    recon_chars = np.asarray(gene_ae.encoder.vocabulary)[recon_tokens]
    
    recon_genes = ["".join(recon_chars[ix]).upper() for ix in range(val_gene_seqs.shape[0])]
    
    for ix in range(5):
        print()
        print(val_gene_seqs[ix].decode("ASCII"))
        print(recon_genes[ix])
    
    
    plt.show()
