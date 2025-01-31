import tensorflow as tf
from utils import GeneSeq_Generator
import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm
import itertools
import datetime

from utils.GeneTransformer_depr import GeneTransformer



def concatenate_datasets(*datasets) -> tf.data.Dataset:
    out_ds = datasets[0].concatenate(datasets[1])
    if len(datasets) > 2:
        for ds in datasets[2:]:
            out_ds = out_ds.concatenate(ds)
    return out_ds



if __name__=="__main__":

    '''
    Load the unique gene sequences and get rid of genes longer than 5 kb
    '''
    datapath = "../data/gene_sequences/unique_dna_seqs_dev.txt"
    # datapath = "../data/gene_sequences/train_unique_gene_seqs.txt"
    gene_dataset = tf.data.TextLineDataset(datapath)
    # gene_dataset = gene_dataset.filter(lambda x: tf.strings.length(x) <= 5000)


    ## Define validation split
    cuts = [gene_dataset.shard(5,k) for k in range(5)]

    val_split = cuts[0]
    train_split = cuts[1:]
    train_split = concatenate_datasets(*train_split)
    

    # leftds, rightds = keras.utils.split_dataset(gene_dataset, left_size=0.8)

    # test_ds = tf.data.Dataset.range(25)
    # cuts = [gene_dataset.shard(5,k) for k in range(5)]


    # for k in range(5):

    #     val = cuts[k]
    #     _t = cuts[:k] + cuts[k+1:]
    #     test = concatenate_datasets(*_t)



    
    '''
    Test forward passes
    '''
    # model = GeneTransformer("codon", embedding_dim=32, key_dim=16, 
    #                         num_heads=4, dropout_rate=0.25, ff_dim=8)
    model = GeneTransformer("nucleotide", embedding_dim=32, encoder_layers=1, 
                            decoder_layers=1, key_dim=16, num_heads=4, 
                            dropout_rate=0.1, ff_dim=8)
    # model = GeneTransformer("nucleotide")


    print(model.autoencoder.summary())
    print(datetime.datetime.now().strftime("%d.%m.%Y_%H.%M"))
    # hist = model.train(train_split, val_split, batch_size=2, epochs=2)

    # print(hist.history.keys())

    # plt.plot(hist.history["loss"])
    # plt.show()



    # test_seqs = []
    # # for ix,g in enumerate(gene_dataset.batch(4)):
    # for ix,g in enumerate(gene_dataset):
    #     g = g[None,...]
    #     # print(g.shape)
    #     # y = model.autoencoder(g)
    #     t = model.tokenize(g, one_hot=False)
    #     _g = model.untokenize(t, one_hot=False)
    #     print(g[0])
    #     print(t[0,...])
    #     print(_g[0])
    #     # print(y.shape, t.shape)

    #     break

    # _,pred = model.predict(test_seqs)
    # true = model.tokenize(test_seqs)
    # true = tf.keras.utils.to_categorical(true, num_classes=66)
    # print(true.shape)

    # model.train(gene_dataset)



    # test_loss = tf.losses.categorical_crossentropy(y_true=true, y_pred=pred)
    # print(test_loss.shape)

    # print(test_seqs)
    # test_seqs = tf.convert_to_tensor(test_seqs)
    # # print(test_seqs)

    # tokens = model.tokenize(test_seqs)
    # # tokens = tf.expand_dims(tokens, 2)
    # print(tokens.shape)

    # vecs = model.encoder(test_seqs)
    # vecs,outs = model.call(test_seqs)
    # print(vecs.shape)
    # print(np.argmax(outs[0,:4,:], axis=1))
    # print(model.summary())

    # print(tf.reduce_sum(outs[0,0,:]))
    # model.fit(gene_dataset)


    
    '''
    Test tokenizers
    '''
    # # # test = [['1',2,3], [4,'5'], [b'6',7,8,9]]
    # # # test = [[b'ATG', b'GTA', b'GAA', b'AGA'], [b'TTA', b'TTT'], [b'AGT', b'TTA', b'GCA', b'AAA', b'TAA']]
    # # # print(tf.keras.utils.pad_sequences(test, dtype=object, padding='post', value=""))

    # # # codon_splitter = lambda x: [x[i:i+3] for i in range(0, tf.strings.length(x), 3)]
    # # # @tf.keras.saving.register_keras_serializable("codon_splitter")


    # # # def codon_splitter(x):
    # # #     y = []
    # # #     for gene in x:
    # # #         _codons = [tf.strings.substr(gene, i, 3) for i in tf.range(tf.strings.length(gene), delta=3)]
    # # #         y.append(tf.convert_to_tensor(_codons))
    # # #     return tf.keras.utils.pad_sequences(y, dtype=object, padding="post", value="")
    

    # def codon_splitter(x):
    #     # Determine the lengths of the input sequences
    #     _str_lens = tf.strings.length(x)
    #     maxlen = tf.reduce_max(_str_lens)

    #     # Initialize arrays defining codon start indices and lengths
    #     # _pos = tf.tile(tf.range(0,maxlen,3, dtype="int32")[:,None], [1,len(x)])
    #     _pos = tf.repeat(tf.range(0,maxlen,3, dtype="int32")[:,None], repeats=len(x), axis=1)
    #     # _len = 3

    #     # Mask out positions and lengths greater than a given sequence length
    #     mask = tf.less(_pos, _str_lens)
    #     _pos = tf.multiply(_pos, tf.cast(mask, "int32"))
    #     _len = tf.multiply(3, tf.cast(mask, "int32"))
    #     codons = tf.strings.substr(x, _pos, _len)

    #     return tf.transpose(codons)
    #     # return tf.strings.split(x, "A")
    #     # print(x[0])
    #     # return tf.split(x[0], 3)
    

    # # test_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=4, char_level=True)
    # # test_tokenizer = tf.keras.layers.TextVectorization(split="character", vocabulary=["a","t","c","g"], output_mode="int")

    # codons = ["".join(c) for c in itertools.product("atcg", repeat=3)]
    # test_tokenizer = tf.keras.layers.TextVectorization(split=codon_splitter, vocabulary=codons, output_mode="int")
    # # print("DEBUG:", test_tokenizer.get_vocabulary())
    # print(len(test_tokenizer.get_vocabulary()))

    # lens = [15,6,12,9]
    # count = 1
    # mods = []
    # for batch in gene_dataset.batch(4).as_numpy_iterator():
    #     _batch = np.array([s[:l] for s,l in zip(batch,lens)])
    #     print(_batch)

    #     print(codon_splitter(_batch))

    #     # g = [gene[0:12],gene[15:21],gene[-15:]]
    #     # # g = gene
    #     # print()
    #     # print("g:", g)
    #     # print(codon_splitter(g))
    #     # # print(test_tokenizer(g))
    #     # # print(test_tokenizer.texts_to_matrix([g]))
    #     # # mods.append(len(gene)%3)

        
    #     break
    #     # if count == 3: break
    #     # count += 1
