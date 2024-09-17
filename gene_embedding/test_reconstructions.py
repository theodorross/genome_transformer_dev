import tensorflow as tf
import numpy as np
import keras
from utils.GeneTransformer import GeneTransformer
# import nltk



if __name__ == "__main__":

    '''
    Load the desired model to test
    '''
    # custom_objs = {"TransformerBlock":TransformerBlock,
    #                "LevenshteinDistance":LevenshteinDistance,
    #                "PositionalEmbedding":PositionalEmbedding}
    # gene_ae = keras.models.load_model("models/geneAE_driven-jazz-93_fold0", custom_objects=custom_objs)
    gene_ae = GeneTransformer("nucleotide", embedding_dim=32, encoder_layers=2, decoder_layers=2,
                              key_dim=8, num_heads=4, ff_dim=256)
    # print(gene_ae.encoder.summary())
    # print(gene_ae.decoder.summary())
    print(gene_ae.summary())

    # test_model = tf.keras.models.Sequential([
    #     tf.keras.layers.Input(shape=64),
    #     tf.keras.layers.Dense(32, "relu"),
    #     tf.keras.layers.Dense(2, "softmax")
    # ])

    # test_in = np.random.random((1,64))
    # test_out = test_model.predict(test_in)
    # print(test_out)


    '''
    Load the desired dataset to test
    '''
    gene_dataset = gene_dataset = tf.data.TextLineDataset("../data/gene_sequences/unique_dna_seqs_dev.txt")

    for x in gene_dataset.batch(1):
        print(x[0].numpy())
        y = gene_ae.predict(x)
        print(np.argmax(y, axis=-1))

        break




    pass

