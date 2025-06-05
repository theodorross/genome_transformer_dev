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
  
    # model_path = "models/checkpoints/rare-sunset-4761_fold0/0562.checkpoint.keras"
    model_path = "models/checkpoints/worldly-elevator-5104_fold0/0199.checkpoint.keras"
    gene_ae = keras.models.load_model(model_path, custom_objects=custom_objs)
    print(gene_ae.summary())

    vocab_arr = np.array(gene_ae.vocabulary)


    '''
    Load the desired dataset to test
    '''
    # gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/train_unique_gene_seqs.txt")
    gene_dataset = tf.data.Dataset.load("../data/gene_sequences/training_dataset")

    ## Filter by gene length
    gene_length = 5000
    # gene_dataset = gene_dataset.filter(lambda x: tf.strings.length(x) < gene_length).cache()
    gene_dataset = gene_dataset.filter(lambda g,d,c: tf.strings.length(g) < gene_length)
    gene_dataset = gene_ae.preprocess_dataset(gene_dataset, batch_size=5, weighted=False, shuffle=True)

    # df = pd.DataFrame(columns=["distance","length"])

    # count = min(300, 150)
    # embedding_arr = np.zeros((count, gene_ae.embedding_dim))
    # gene_lens = np.zeros(count)

    # print(gene_ae.decoder.mask_token)


    # for ix,x in tqdm(gene_dataset.padded_batch(5).enumerate()):
    for ix,_x in gene_dataset.enumerate():

        # print()
        x,_ = _x

        out = gene_ae.predict(x)
        z = out[0]
        y = out[2]
        # print(z.shape)
        print("L2 norms:", np.linalg.norm(z, ord=2, axis=1))
        print("z maxs:", np.max(z, axis=1))
        print("z mins:", np.min(z, axis=1))
        print('z means:', np.mean(z, axis=0))
        print('z std:', np.std(z, axis=0).max())
        # print(z@z.T)

        plt.imshow(y[0,:30,:])
        plt.show()

        ## Reconstruct the predicted sequences
        recon_tokens = np.argmax(y, axis=-1)
        print("recon_tokens", recon_tokens.shape)
        recon_chars = np.asarray(gene_ae.encoder.vocabulary)[recon_tokens]
        recon_genes = ["".join(recon_chars[ix]).upper() for ix in range(x.numpy().shape[0])]

        # in_str = x.numpy()[0].decode("ASCII")
        # print(x.numpy()[0].decode("ASCII")[:30])
        # # print(recon_tokens[0][:30])
        # # print(recon_tokens[0][-30:])
        # print(np.unique(list(in_str), return_counts=True))
        # print(np.unique(recon_tokens[0], return_counts=True))
        # print(vocab_arr)
        # exit()

        for ix in range(5):
            print()
            codon_pos_errs = np.array([0,0,0])
            recon_str = ""
            p_count = 0
            for t,r in zip(x.numpy()[ix].decode("ASCII"), recon_genes[ix]):
                if t != r: recon_str += f"\033[0;31m{r}\033[0m"
                else: 
                    recon_str += f"\033[0;32m{r}\033[0m"
                    codon_pos_errs[p_count%3] += 1
                p_count += 1
            print(x.numpy()[ix].decode("ASCII"))
            print(recon_str)
            print("codon position errors:", codon_pos_errs)

        exit()

        # y_indices = np.argmax(y, axis=-1).squeeze()
        # letters = vocab_arr[y_indices]
        # recon = "".join(letters).upper()

        # ## Compute the levenshtein distance
        # print("input:", inp.upper())
        # print("recon:", recon)
        # print("input:", inp.upper()[:60])
        # print("recon:", recon[:60])
        # exit()
        # dist = nltk.edit_distance(inp.upper(),recon)
        # print(f"{dist} / {len(inp)}")
        # df.loc[ix.numpy(), :] = [dist, len(inp)]

    # # # print(df)
    # # plt.scatter(df["length"], df["distance"])
    # # plt.xlabel("gene length")
    # # plt.ylabel("input-output Leveshtein distance")
    # # # plt.show()


    # ## plot embeddings
    # fig = plt.figure()

    # # pca_embeddings = PCA(n_components=3).fit_transform(embedding_arr)
    # # ax = fig.add_subplot(projection="3d")
    # # pos = ax.scatter(pca_embeddings[:,0], pca_embeddings[:,1], pca_embeddings[:,2], c=gene_lens)
    # # ax.set_zlabel("e3")

    # pca_embeddings = PCA(n_components=2).fit_transform(embedding_arr)
    # ax = fig.add_subplot()
    # pos = ax.scatter(pca_embeddings[:,0], pca_embeddings[:,1], c=gene_lens)

    # ax.set_xlabel("e1")
    # ax.set_ylabel("e2")
    # fig.colorbar(pos, ax=ax, label="Gene Length")
    # plt.show()

