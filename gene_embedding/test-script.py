import tensorflow as tf
from utils import GeneSeq_Generator


def isolate_nucleotides(val):
    return(tf.split(",")[5])



if __name__=="__main__":

    datapath = "../data/gene_data.csv"

    # with open(datapath,"r") as f:
    #     for l in f.readlines():
    #         print(l)

    test = tf.data.TextLineDataset(datapath).map(isolate_nucleotides)

    c = 0
    for a in test:
        print(a)
        c += 1
        if c > 50:
            break

    pass