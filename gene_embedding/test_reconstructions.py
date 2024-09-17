import tensorflow as tf
import numpy as np
import keras
from utils.GeneTransformer_dep import *
import nltk



if __name__ == "__main__":

    '''
    Load the desired model to test
    '''
    # custom_objs = {"TransformerBlock":TransformerBlock,
    #                "LevenshteinDistance":LevenshteinDistance,
    #                "PositionalEmbedding":PositionalEmbedding}
    # gene_ae = keras.models.load_model("models/geneAE_driven-jazz-93_fold0", custom_objects=custom_objs)
    gene_ae = GeneTransformer("nucleotide")
    print(gene_ae.summary())

    print(gene_ae.ff_dim)

    '''
    Load the desired dataset to test
    '''
    # gene_dataset = gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs_dev.txt")

    # for x in gene_dataset.batch(4):
    #     print(x[0].numpy())
    #     y = gene_ae.encoder.predict(x)
    #     print(y.shape)

    #     break




    pass

