import tensorflow as tf
import numpy as np
import keras
import nltk
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import entropy

from sklearn.decomposition import PCA

from utils.GeneTransformer import GeneTransformer
from utils.TrainingUtils import MaskedSparseCategoricalCrossentropy


def clip_gene(gene):
    # num = tf.random.categorical( tf.math.log([[.2,.2,.2,.2,.2]]), num_samples=1, dtype=tf.int32) + 30
    # num = tf.squeeze(num)
    # num = 30
    return tf.strings.substr(gene, 0, 50)


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
    config = {"embedding_dim":6,
              "latent_dim":16,
              'n_sequence_tokens':16,
              "encoder_layers":5,
              'decoder_layers':3,
              'key_dim':128,
              'num_heads':16,
              'ff_dim':256,
              'max_length':50,
              'masking_rate':0.1,
              'learning_rate':1e-3}


    '''
    Define the dataset
    '''
    ## Generate a fake datset
    # genes = ["".join(np.random.choice(["A","T","C","G"], np.random.choice([30,31,32,33,34,35],1))) for _ in range(300)]
    # gene_dataset = tf.data.Dataset.from_tensor_slices(genes)


    ## Load the genes
    # gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs_dev.txt")
    # gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs.txt")
    gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/rsmD_alleles.txt")
    gene_dataset = gene_dataset.concatenate(tf.data.TextLineDataset("../data/gene_sequences/coaD_alleles.txt"))
    # gene_dataset = gene_dataset.concatenate(tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs_dev.txt"))

    # test = np.asarray(list(gene_dataset.as_numpy_iterator()))
    # print("Debug:", gene_dataset.cardinality().numpy())
    # print(test.shape)

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
    ## If not training --> for debugging
    # gene_ae = GeneTransformer("nucleotide", **config)
    # val_genes = dataset_cuts[0]
    # train_genes = dataset_cuts[1:]

    ## Initialize a dataframe for plotting
    df_list = []

    for k in range(5):
        
        ## Get this fold's data
        val_genes = dataset_cuts[k]
        train_genes = dataset_cuts[:k] + dataset_cuts[k+1:]
        train_genes = concatenate_datasets(*train_genes)

        ## Define the model
        gene_ae = GeneTransformer("nucleotide", **config)
        print(gene_ae.summary())
        # print(gene_ae.decoder.summary())

        ## Train the model
        # plip = lambda e,lr: lr if e<50 else lr/5
        plip = lambda e,lr: lr*tf.exp(-0.1) if e%50==49 else lr
        # plip = lambda e,lr: config["learning_rate"]*tf.exp(-0.005*e)
        # def plip(e, lr):
        #     if e < 100: return 1e-3
        #     else: return 1e-4
        callback = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
        lrsched = tf.keras.callbacks.LearningRateScheduler(plip)
        train_history = gene_ae.train(train_genes, val_genes, 4, 500, callback, lrsched, verbose=2)

        ## Save the training history
        hist_df = pd.DataFrame(train_history)
        hist_df.index.name = "epoch"
        hist_df.reset_index(inplace=True)
        hist_df["fold"] = k
        df_list.append(hist_df)

        break


    '''
    Plot the training metrics
    '''
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
    filestr = f"z{config['embedding_dim']}-s{config['n_sequence_tokens']}-l{config['latent_dim']}"
    filestr += f"-e{config['encoder_layers']}-d{config['decoder_layers']}-h{config['num_heads']}"
    filestr += f"-k{config['key_dim']}-ff{config['ff_dim']}-lr{config['learning_rate']}"
    # fig.suptitle(f"{filestr}\ncontext")
    # fig.suptitle(f"{filestr}\nseqmask")
    fig.suptitle(f"{filestr}\nperciever")
    

    plt.tight_layout()
    # plt.savefig(f"figures/small_test_context_{filestr}.png")
    # plt.savefig(f"figures/small_test_seqmask_{filestr}.png")
    # plt.savefig(f"figures/small_test_query_stuff_{filestr}.png")
    # plt.savefig(f"figures/small_test_perciever_{filestr}_includestart_.png")

    '''
    Test the reconstructions
    '''
    ## Isolte the validation genes
    # val_gene_seqs = next(val_genes.shuffle(val_genes.cardinality()).batch(60).as_numpy_iterator())
    val_gene_seqs = next(train_genes.shuffle(train_genes.cardinality()).batch(5).as_numpy_iterator())
    
    ## Pass through the model
    encodes = gene_ae.encode(val_gene_seqs)
    decodes = gene_ae.decode(encodes)

    ## Reconstruct the sequences
    recon_tokens = np.argmax(decodes, axis=-1)
    recon_chars = np.asarray(gene_ae.encoder.vocabulary)[recon_tokens]
    recon_genes = ["".join(recon_chars[ix]).upper() for ix in range(val_gene_seqs.shape[0])]

    ## Plot the prediction probabilities
    fo,ao = plt.subplots(1,5, figsize=(7,7))
    for ix in range(5):
        print()
        # print(len(val_gene_seqs[ix].decode("ASCII")))
        # print(len(recon_genes[ix]))
        print(val_gene_seqs[ix].decode("ASCII"))
        print(recon_genes[ix])

        ao[ix].imshow(decodes[ix,...], vmin=0, vmax=1)
        ao[ix].set_yticks(range(decodes.shape[1]), labels=list(val_gene_seqs[ix].decode("ASCII")))
        ao[ix].set_xticks(range(6), labels=np.asarray(gene_ae.encoder.vocabulary), rotation=90)
    ao[0].set_ylabel("True Nucleotide")
    fo.tight_layout()


    '''
    Plot position-wise accuracy
    '''
    ## Get the validation genes as sequnces and arrays
    # val_gene_seqs = next(val_genes.shuffle(val_genes.cardinality()).batch(val_genes.cardinality()).as_numpy_iterator())
    val_gene_seqs = np.array( list(val_genes.shuffle(val_genes.cardinality()).as_numpy_iterator()) )
    # val_gene_seqs = next(val_genes.batch(5).as_numpy_iterator())
    val_gene_chars = np.array( [list(_v.decode('ASCII').lower()) for _v in val_gene_seqs] )

    ## Predict the validation genes
    # preds = gene_ae.predict(val_gene_seqs)
    preds = gene_ae(val_gene_seqs, training=False)
    recon_tokens = np.argmax(preds, axis=-1)
    recon_chars = np.asarray(gene_ae.encoder.vocabulary)[recon_tokens]

    ## Compute the coordinate-level accuracy and entropy
    pos_acc = np.mean(recon_chars==val_gene_chars, axis=0)
    pos_frq = np.stack([np.mean(val_gene_chars==ch, axis=0) for ch in ["a","t","c","g"]], axis=0 )
    pos_true_H = -np.sum(pos_frq * np.log2(pos_frq+1e-10), axis=0)
    pos_pred_H = np.mean( -np.sum(preds * np.log2(preds+1e-10), axis=2), axis=0)

    print(pos_acc.shape, pos_true_H.shape, pos_pred_H.shape)


    fig1, a1 = plt.subplots(2,2, figsize=(6,6))
    a1[0,0].plot(pos_acc, "-o")
    a1[0,0].set_xlabel("Position")
    a1[0,0].set_ylabel("Accuracy")
    a1[0,0].set_ylim([0,1])

    a1[0,1].plot(pos_true_H, "-o")
    a1[0,1].set_xlabel("Position")
    a1[0,1].set_ylabel("Entropy (bits)")

    a1[1,0].scatter(pos_acc, pos_true_H)
    a1[1,0].set_xlabel("Accuracy")
    a1[1,0].set_ylabel("True Nucleotide Entropy")

    a1[1,1].scatter(pos_pred_H, pos_true_H)
    a1[1,1].set_xlabel("Mean Prediction Entropy")
    a1[1,1].set_ylabel("True Nucleotide Entropy")

    fig1.suptitle("Position-wise metrics")
    fig1.tight_layout()
    
    plt.show()
