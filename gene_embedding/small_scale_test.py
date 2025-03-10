import os
import tensorflow as tf
import numpy as np
import keras
# import nltk
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import entropy

from sklearn.decomposition import PCA

from utils.GeneTransformer import GeneTransformer
from utils.TrainingUtils import MaskedSparseCategoricalCrossentropy, MaskedAccuracy

from tensorflow.python.client import device_lib

def clip_gene(len):
    # num = tf.random.categorical( tf.math.log([[.2,.2,.2,.2,.2]]), num_samples=1, dtype=tf.int32) + 30
    # num = tf.squeeze(num)
    # num = 30
    def _clip(gene):
        return tf.strings.substr(gene, 3, len)
    return _clip


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
    config = {'embedding_dim':8,
              "latent_dim":32,
              'n_sequence_tokens':8,
              'encoder_layers':2,
              'decoder_layers':2,
              'dropout_rate':0,
              'key_dim':8,
              'num_heads':12,
              'ff_dim':64,
              'max_length':50,
              'decode_length':50,
              'masking_rate':0.00,
              'learning_rate':5e-3}
    
    ## Instantiate the multi-GPU training strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"\nNumber of devices: {strategy.num_replicas_in_sync}\n")
    print(type(tf.distribute.get_replica_context().num_replicas_in_sync))
    # print(device_lib.list_local_devices())

    keras.config.disable_traceback_filtering()


    '''
    Define the dataset
    '''
    ## Generate a fake datset
    # genes = ["".join(np.random.choice(["A","T","C","G"], np.random.choice([30,31,32,33,34,35],1))) for _ in range(300)]
    # gene_dataset = tf.data.Dataset.from_tensor_slices(genes)

    # print("\033[93m test \033[0m")
    # exit()


    ## Load the genes
    # gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs_dev.txt")
    gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs.txt")
    # gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/rsmD_alleles.txt")
    # gene_dataset = gene_dataset.concatenate(tf.data.TextLineDataset("../data/gene_sequences/coaD_alleles.txt"))
    # gene_dataset = gene_dataset.concatenate(tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs_dev.txt"))

    # test = np.asarray(list(gene_dataset.as_numpy_iterator()))
    # print("Debug:", gene_dataset.cardinality().numpy())
    # print(test.shape)

    ## Shorten them dramatically
    # gene_dataset = gene_dataset.map(clip_gene)

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

    for k in range(5):
        
        ## Get this fold's data
        _val_genes = dataset_cuts[k]
        _train_genes = dataset_cuts[:k] + dataset_cuts[k+1:]
        _train_genes = concatenate_datasets(*_train_genes)

        ## Define the model
        gene_ae = GeneTransformer("nucleotide", **config)
        print(gene_ae.summary())
        print(gene_ae.encoder.summary())
        print(gene_ae.decoder.summary())


        ## Loop for increasing gene lengths
        train_history = {}
        # for ix, gene_len in enumerate([50,75,100,125]):
        _epoch_count = 0
        for ix, gene_len in enumerate([50]):
            gene_ae.update_decode_length(gene_len)

            ## Trim the genes
            # val_genes = _val_genes.map(clip_gene(25))
            # train_genes = _train_genes.map(clip_gene(25))
            val_genes = _val_genes.filter(lambda x: tf.strings.length(x) < gene_len)
            train_genes = _train_genes.filter(lambda x: tf.strings.length(x) < gene_len)

            ## Train the model
            # plip = lambda e,lr: lr*tf.exp(-0.1) if (e%250==249 and e>1000) else lr
            plip = lambda e,lr: lr*tf.exp(-0.1) if (e%250==249) else lr
            callback = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
            lrsched = tf.keras.callbacks.LearningRateScheduler(plip)
            epochs = 5
            # epochs = 2

            _hist = gene_ae.train(train_genes, val_genes, 5, epochs+_epoch_count, callbacks=[callback, lrsched], 
                                  num_devices=strategy.num_replicas_in_sync, verbose=1, initial_epoch=_epoch_count)
            _epoch_count += len(_hist["loss"])
            exit()

            ## Store the training history
            for key,val in _hist.items():
                if key in train_history.keys():
                    train_history[key] += _hist[key]
                else:
                    train_history[key] = _hist[key]


        ## Save the training history
        hist_df = pd.DataFrame(train_history)
        hist_df.index.name = "epoch"
        hist_df.reset_index(inplace=True)
        hist_df["fold"] = k
        df_list.append(hist_df)
        
        ## Plot attentions scores
        # x = next(train_genes.batch(1).as_numpy_iterator())
        # _ = gene_ae(x)
        # a_e1 = gene_ae.encoder.cross_attn_layer.last_attention.numpy()
        # # a_d1 = gene_ae.decoder.transformer_layer.last_attention.numpy()
        # a_d1 = gene_ae.decoder.transformer_layer.last_attention.numpy()
        # f,a = plt.subplots(1,2)
        # cbar = a[0].imshow(a_e1[0,0,:,:])
        # a[0].set_title("encoder cross attn score")
        # plt.colorbar(cbar, ax=a[0])
        # cbar = a[1].imshow(a_d1[0,0,:,:])
        # a[1].set_title("decoder cross attn score")
        # plt.colorbar(cbar, ax=a[1])
        # plt.show()

        break

    '''
    Test something
    '''
    # m_acc = MaskedAccuracy()
    for val_seqs in val_genes.batch(3):
        break

    print(val_seqs)
    preds = gene_ae(val_seqs)
    # print(type(preds._keras_mask))
    print(preds)
    print(tf.keras.backend.get_keras_mask(preds))

    # print(v_x)
    # print(v_y)
    # v_pred = gene_ae.predict(v_x)

    # m_acc.update_state(v_y, v_pred)

    # print("\nFINAL_OUT")
    # print(m_acc.acc.numpy)

    exit()

    '''
    Plot the training metrics
    '''
    ## Combine the training history dataframes
    df = pd.concat(df_list)
    print(df)
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
    val_gene_tokens = gene_ae.tokenize(val_gene_seqs)
    val_gene_chars = np.asarray(gene_ae.encoder.vocabulary)[val_gene_tokens]
    
    ## Pass through the model
    encodes = gene_ae.encoder(val_gene_seqs)
    print("ENCODES:", encodes.shape)
    decodes = gene_ae.decoder(encodes)

    ## Plot the encoded values
    fe,ae = plt.subplots(1, 5, figsize=(10,3))
    for ix in range(5):
        _b = ae[ix].imshow(tf.transpose(encodes[ix]))
        # plt.colorbar(_b, ax=ae[ix])
        ae[ix].set_title(f"input {ix}")
        # ae[ix].set_xlabel("latent_dim")
    # ae[0].set_ylabel("latent_tokens")
    ae[0].set_ylabel("latent_dim")
    fe.suptitle("Encodings")

    ## Reconstruct the sequences
    recon_tokens = np.argmax(decodes, axis=-1)
    # recon_tokens = np.argmax(encodes, axis=-1)
    recon_chars = np.asarray(gene_ae.encoder.vocabulary)[recon_tokens]
    recon_genes = ["".join(recon_chars[ix]).upper() for ix in range(val_gene_seqs.shape[0])]

    ## Plot the prediction probabilities
    fo,ao = plt.subplots(5,1, figsize=(10,5))
    for ix in range(5):
        print()
        # print(len(val_gene_seqs[ix].decode("ASCII")))
        # print(len(recon_genes[ix]))
        recon_str = ""
        for t,r in zip(val_gene_seqs[ix].decode("ASCII"), recon_genes[ix]):
            if t != r: recon_str += f"\033[0;31m{r}\033[0m"
            else: recon_str += f"\033[0;32m{r}\033[0m"
            # print(t,r)
        print(val_gene_seqs[ix].decode("ASCII"))
        # print(recon_genes[ix])
        print(recon_str)


        ao[ix].imshow(tf.transpose(decodes[ix,...]), vmin=0, vmax=1)
        # ao[ix].imshow(encodes[ix,...], vmin=0, vmax=1)
        ao[ix].set_xticks(range(decodes.shape[1]), labels=val_gene_chars[ix,:])
        # ao[ix].set_yticks(range(encodes.shape[1]), labels=val_gene_chars[ix,:])
        ao[ix].set_yticks(range(len(gene_ae.encoder.vocabulary)), labels=np.asarray(gene_ae.encoder.vocabulary))
    ao[-1].set_xlabel("True Nucleotide")
    # fo.tight_layout()


    '''
    Plot position-wise accuracy
    '''
    ## Get the validation genes as sequnces and arrays
    # val_gene_seqs = next(val_genes.shuffle(val_genes.cardinality()).batch(val_genes.cardinality()).as_numpy_iterator())
    val_gene_seqs = np.array( list(train_genes.as_numpy_iterator()) )
    # val_gene_seqs = next(val_genes.batch(5).as_numpy_iterator())
    val_gene_tokens = gene_ae.tokenize(val_gene_seqs)
    val_gene_chars = np.asarray(gene_ae.encoder.vocabulary)[val_gene_tokens]

    ## Predict the validation genes
    # preds = gene_ae.predict(val_gene_seqs)
    preds = gene_ae(val_gene_seqs, training=False)
    recon_tokens = np.argmax(preds, axis=-1)
    recon_chars = np.asarray(gene_ae.encoder.vocabulary)[recon_tokens]

    ## Compute the coordinate-level accuracy and entropy
    pos_acc = np.mean(recon_chars==val_gene_chars, axis=0)
    pos_frq = np.stack([np.mean(val_gene_chars==token, axis=0) for token in gene_ae.encoder.vocabulary], axis=0 )
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
    
    '''
    Show all plots
    '''
    plt.show()
