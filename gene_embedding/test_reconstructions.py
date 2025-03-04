import tensorflow as tf
import numpy as np
import keras
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from utils.GeneTransformer import GeneTransformer
from utils.TransformerBlock import TransformerDecoderBlock, TransformerEncoderBlock
from utils.TrainingUtils import *
from utils.PositionalEmbedding import PositionalEmbedding
from utils.SequenceDecoder import SequenceDecoder
from utils.SequenceEncoder import SequenceEncoder


if __name__ == "__main__":

    '''
    Load the desired model to test
    '''
    custom_objs = {"GeneTransformer":GeneTransformer,
                   "TransformerEncoderBlock":TransformerEncoderBlock,
                   "TransformerDecoderBlock":TransformerDecoderBlock,
                   "LevenshteinDistance":LevenshteinDistance,
                   "MaskedAccuracy":MaskedAccuracy,
                   "MaskedSparseCrossentropy":MaskedSparseCategoricalCrossentropy,
                   "PositionalEmbedding":PositionalEmbedding,
                   "SequenceEncoder":SequenceEncoder,
                   "SequenceDecoder":SequenceDecoder}
  
    gene_ae = keras.models.load_model("models/geneAE_lemon-fog-260_fold0", custom_objects=custom_objs)

    vocab_arr = np.array(gene_ae.vocabulary)


    '''
    Load the desired dataset to test
    '''
    gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/train_unique_gene_seqs.txt")

    ## Filter by gene length
    gene_length = 300
    gene_dataset = gene_dataset.filter(lambda x: tf.strings.length(x) < gene_length).cache()
    # gene_dataset = gene_ae._preprocess_dataset(gene_dataset)

    # df = pd.DataFrame(columns=["distance","length"])

    # count = min(300, 150)
    # embedding_arr = np.zeros((count, gene_ae.embedding_dim))
    # gene_lens = np.zeros(count)

    # print(gene_ae.decoder.mask_token)


    # for ix,x in tqdm(gene_dataset.padded_batch(5).enumerate()):
    for ix,x in gene_dataset.padded_batch(5).enumerate():

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
        

        ## Store embeddings
        # embedding_arr[ix,:] = z[0,0,:]
        # gene_lens[ix] = len(inp)

        ## Reconstruct the predicted sequences
        # sample_genes = next(validation_fold.batch(5).as_numpy_iterator())
        # sample_preds = gene_ae.predict(x)
        recon_tokens = np.argmax(y, axis=-1)
        recon_chars = np.asarray(gene_ae.encoder.vocabulary)[recon_tokens]
        recon_genes = ["".join(recon_chars[ix]).upper() for ix in range(x.numpy().shape[0])]

        for ix in range(5):
            print()
            recon_str = ""
            for t,r in zip(x.numpy()[ix].decode("ASCII"), recon_genes[ix]):
                if t != r: recon_str += f"\033[0;31m{r}\033[0m"
                else: recon_str += f"\033[0;32m{r}\033[0m"
            print(x.numpy()[ix].decode("ASCII"))
            print(recon_str)

        exit()

        y_indices = np.argmax(y, axis=-1).squeeze()
        letters = vocab_arr[y_indices]
        recon = "".join(letters).upper()

        ## Compute the levenshtein distance
        print("input:", inp.upper())
        print("recon:", recon)
        print("input:", inp.upper()[:60])
        print("recon:", recon[:60])
        exit()
        # dist = nltk.edit_distance(inp.upper(),recon)
        # print(f"{dist} / {len(inp)}")
        # df.loc[ix.numpy(), :] = [dist, len(inp)]

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

