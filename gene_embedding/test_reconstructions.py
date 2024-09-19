import tensorflow as tf
import numpy as np
import keras
import nltk
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.GeneTransformer import GeneTransformer
from utils.TransformerBlock import TransformerBlock
from utils.TrainingUtils import *
from utils.PositionalEmbedding import PositionalEmbedding
from utils.SequenceDecoder import SequenceDecoder
from utils.SequenceEncoder import SequenceEncoder


if __name__ == "__main__":

    '''
    Load the desired model to test
    '''
    custom_objs = {"GeneTransformer":GeneTransformer,
                   "TransformerBlock":TransformerBlock,
                   "LevenshteinDistance":LevenshteinDistance,
                   "MaskedAccuracy":MaskedAccuracy,
                   "MaskedSparseCrossentropy":MaskedSparseCategoricalCrossentropy,
                   "PositionalEmbedding":PositionalEmbedding,
                   "SequenceEncoder":SequenceEncoder,
                   "SequenceDecoder":SequenceDecoder}
    # gene_ae = keras.models.load_model("models/geneAE_lemon-pyramid-1_fold0", custom_objects=custom_objs)
    # gene_ae = keras.models.load_model("models/geneAE_super-wildflower-66_fold0", custom_objects=custom_objs)    ## mask_zero = True
    # gene_ae = keras.models.load_model("models/geneAE_super-wildflower-66_fold0", custom_objects=custom_objs)    ## mask_zero = False
    # gene_ae = keras.models.load_model("models/geneAE_super-salad-79_fold0", custom_objects=custom_objs)    ## Masked Levenshtein
    # gene_ae = keras.models.load_model("models/geneAE_smart-disco-122_fold0", custom_objects=custom_objs)    ## Codons
    # gene_ae = keras.models.load_model("models/geneAE_honest-field-133_fold0", custom_objects=custom_objs)   
    gene_ae = keras.models.load_model("models/geneAE_major-brook-134_fold0", custom_objects=custom_objs)   

    # gene_ae = GeneTransformer("nucleotide", embedding_dim=32, encoder_layers=2, decoder_layers=2,
    #                           key_dim=8, num_heads=4, ff_dim=256)
    print(gene_ae.encoder.summary())
    # print(gene_ae.decoder.summary())
    print(gene_ae.summary())

    vocab_arr = np.array(gene_ae.vocabulary)

    # test_model = tf.keras.models.Sequential([
    #     tf.keras.layers.Input(shape=64),
    #     tf.keras.layers.Dense(32, "relu"),
    #     tf.keras.layers.Dense(2, "softmax")
    # ])

    # test_in = np.random.random((1,64))
    # test_out = test_model.predict(test_in)
    # print(test_out)
    # print(gene_ae.vocabulary)
    lev_dist = LevenshteinDistance(gene_ae.vocabulary)
    


    '''
    Load the desired dataset to test
    '''
    gene_dataset = gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs_dev.txt")

    df = pd.DataFrame(columns=["distance","length"])
    embedding_arr = np.zeros((300, 2))

    for ix,x in tqdm(gene_dataset.padded_batch(1).enumerate(), total=300):
        # print()

        ## Test the encoder
        zs,zv = gene_ae.encoder.predict(x, verbose=False)
        # print(zv == np.mean(zs, axis=1))
        embedding_arr[ix,:] = zv

        fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        ax = fig.add_subplot()
        tokens = gene_ae.tokenize(x)
        for t in np.unique(tokens):
            m = np.squeeze(tokens == t)
            # ax.scatter(zs[0,m,0], zs[0,m,1], zs[0,m,2], label=vocab_arr[t])
            ax.scatter(zs[0,m,0], zs[0,m,1], label=vocab_arr[t])
        plt.legend()
        plt.show()
        

        ## Parse the input
        inp = x.numpy()[0].decode('ASCII')
        # print("Input lengths:")
        # print([len(s) for s in inp])
        # print(inp)

        ## Compute model predictions
        y = gene_ae.predict(x, verbose=0)

        ## Reconstruct the predicted sequences
        # print("Reconstruction:")
        # print(y[0,-10:,:])
        y_indices = np.argmax(y, axis=-1).squeeze()
        # print(y_indices[:,-10:])
        letters = vocab_arr[y_indices]
        # print(letters[:,-10:])

        recon = "".join(letters).upper()
        # print(inp == recon)
        # if inp != recon: print("Imperfect reconstruction")
        dist = nltk.edit_distance(inp,recon)
        # print(inp[:125] + "...")
        # print(recon[:125] + "...")
        # print(dist)

        # ## Test the levenshtein distance metric
        # y_true = gene_ae.tokenize(x)
        # lev_dist.update_state(y_true, y)
        # print("Test metric:", lev_dist.result().numpy())

        df.loc[ix.numpy(), :] = [dist, len(inp)]
        # if ix.numpy() > 3: break


    print(df)
    plt.scatter(df["length"], df["distance"])
    plt.xlabel("gene length")
    plt.ylabel("input-output Leveshtein distance")
    # # plt.show()


    ## plot embeddings
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(embedding_arr[:,0], embedding_arr[:,1], embedding_arr[:,2])
    plt.show()

