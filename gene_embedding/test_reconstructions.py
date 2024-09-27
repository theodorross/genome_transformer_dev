import tensorflow as tf
import numpy as np
import keras
import nltk
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

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
  
    # gene_ae = keras.models.load_model("models/geneAE_ancient-sun-1_fold0", custom_objects=custom_objs)
    # # gene_ae = keras.models.load_model("models/geneAE_ruby-wind-2_fold0", custom_objects=custom_objs)
    # # gene_ae = keras.models.load_model("models/geneAE_vivid-star-3_fold0", custom_objects=custom_objs)
    gene_ae = keras.models.load_model("models/geneAE_devout-universe-16_fold0", custom_objects=custom_objs)
    # gene_ae = keras.models.load_model("models/geneAE_magic-wood-14_fold0", custom_objects=custom_objs)

    vocab_arr = np.array(gene_ae.vocabulary)



    # gene_ae = GeneTransformer("nucleotide", embedding_dim=2, encoder_layers=1, decoder_layers=1,
    #                           key_dim=5, num_heads=1, ff_dim=16)
    # # # print(gene_ae.encoder.summary())
    # print(gene_ae.decoder.summary())
    # print(gene_ae.summary())

    # vocab_arr = np.array(gene_ae.vocabulary)
    # print("vocab length:", len(vocab_arr))
    # print(vocab_arr)
    # # print(gene_ae.decoder.transformer_layers[0].attn.weights)

    # ## Dummy sequencess
    # # test_input_1 = tf.constant(["xATGC", "xGCCGT"])
    # # test_input_2 = tf.constant(["xxxxAC", "xxxxGT", "xxxxCT"])
    # test_input_2 = tf.constant(["xATCAC", "xGACGT", "xTACCT"])

    # # print(gene_ae.encoder.embedding_layer.compute_mask(test_input_1))
    # # print(tf.strings.length(test_input_1))

    # ## Test the decoder masking
    # # z1 = gene_ae.encode(test_input_1)
    # z2 = gene_ae.encode(test_input_2)
    # print(z2.shape)

    # ## Print the decoding mask
    # m = gene_ae.get_decoder_mask(z2)
    # print("Decoder mask")
    # print(m)

    # ## Add constants
    # # mask = np.zeros((1,z1.shape[1],1))
    # # mask[:,0,:] = 0
    # # mask[:,1,:] = 0
    # # mask[:,2,:] = 1
    # # mask[:,3,:] = 0
    # # mask[:,4,:] = 0
    # # mask[:,5,:] = 0
    # # _z1 = z1 + mask
    # # _z2 = z2 + mask

    # ## Add noise
    # mask = np.ones((1,z2.shape[1],z2.shape[2]))
    # noise = np.random.normal(size=mask.shape)
    # # noise[:,[0,1,2,3],:] = 0
    # noise[:,[0],:] = 0
    # # _z1 = z1 + noise
    # _z2 = z2 + noise
    # print("\nComparing z vectors")
    # print(z2[0] == _z2[0])

    # # d1 = tf.transpose( gene_ae.decode(z1), perm=(0,2,1) )
    # # _d1 = tf.transpose( gene_ae.decode(_z1), perm=(0,2,1) )
    # d2 = tf.transpose( gene_ae.decode(z2), perm=(0,2,1) )
    # _d2 = tf.transpose( gene_ae.decode(_z2), perm=(0,2,1) )

    # # print("\nINPUT 1")
    # # print(d1[0]==_d1[0])
    # # # print(d1[0])
    # # # print(_d1[0])
    # # # print(d1[0]-_d1[0])
    # # # print(tf.reduce_max(tf.abs(d1-_d1)))
    # print("\nINPUT 2")
    # print(d2[0]==_d2[0])

    # print("\nComparing columns -> these should all be false")
    # print(d2[0,:,0] == d2[0,:,1])
    # print(d2[0,:,1] == d2[0,:,2])
    # print(d2[0,:,2] == d2[0,:,3])
    # print(d2[0,:,3] == d2[0,:,4])
    # print(d2[0,:,4] == d2[0,:,5])
    # print(d2[0])
    # print(_d2[0])
    # # print(d2[0]-_d2[0])
    # # print(tf.reduce_max(tf.abs(d2-_d2)))
    

    # ## Look at the attention weights
    # # w = gene_ae.decoder.tra


    '''
    Load the desired dataset to test
    '''
    gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs_dev.txt")
    gene_dataset = gene_ae.preprocess_genes(gene_dataset)

    df = pd.DataFrame(columns=["distance","length"])

    count = min(300, 150)
    embedding_arr = np.zeros((count, gene_ae.embedding_dim))
    gene_lens = np.zeros(count)

    # print(gene_ae.decoder.mask_token)


    for ix,x in tqdm(gene_dataset.padded_batch(1).enumerate(), total=count):
        if ix.numpy() >= count: break

        # print()
        
        ## Parse the input
        inp = x.numpy()[0].decode('ASCII')

        ## Compute model predictions
        z = gene_ae.encode(x)
        
        # noise = tf.random.normal(shape=z.shape)
        # n_mask = np.ones((1,z.shape[1],1))
        # n_mask[:,0,:] = 0
        # # _z = z*(1-n_mask) + n_mask*noise
        # _z = z + n_mask * noise

        # _z = z * np.cos(np.arange(z.shape[1])[None,:,None])
        # _z = z + np.sin(np.arange(z.shape[1])[None,:,None])
        # _z = z + np.arange(z.shape[1])[None,:,None]

        # print(z.shape)
        # print(tf.reduce_mean(_z - z, axis=2).numpy()[0,:10])

        y = gene_ae.decode(z)
        # print(z.shape)
        

        ## Store embeddings
        embedding_arr[ix,:] = z[0,0,:]
        gene_lens[ix] = len(inp)

        ## Reconstruct the predicted sequences
        y_indices = np.argmax(y, axis=-1).squeeze()
        letters = vocab_arr[y_indices]
        recon = "".join(letters).upper()

        ## Compute the levenshtein distance
        print("input:", inp.upper())
        print("recon:", recon)
        print("input:", inp.upper()[:60])
        print("recon:", recon[:60])
        dist = nltk.edit_distance(inp.upper(),recon)
        print(f"{dist} / {len(inp)}")
        df.loc[ix.numpy(), :] = [dist, len(inp)]

    # # print(df)
    # plt.scatter(df["length"], df["distance"])
    # plt.xlabel("gene length")
    # plt.ylabel("input-output Leveshtein distance")
    # # plt.show()


    ## plot embeddings
    fig = plt.figure()

    # pca_embeddings = PCA(n_components=3).fit_transform(embedding_arr)
    # ax = fig.add_subplot(projection="3d")
    # pos = ax.scatter(pca_embeddings[:,0], pca_embeddings[:,1], pca_embeddings[:,2], c=gene_lens)
    # ax.set_zlabel("e3")

    pca_embeddings = PCA(n_components=2).fit_transform(embedding_arr)
    ax = fig.add_subplot()
    pos = ax.scatter(pca_embeddings[:,0], pca_embeddings[:,1], c=gene_lens)

    ax.set_xlabel("e1")
    ax.set_ylabel("e2")
    fig.colorbar(pos, ax=ax, label="Gene Length")
    plt.show()

