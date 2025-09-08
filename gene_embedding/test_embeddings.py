import numpy as np
import tensorflow as tf
import keras
import os
from utils.GeneTransformer import GeneTransformer

from tqdm import tqdm


def concatenate_datasets(*datasets) -> tf.data.Dataset:
    ## Function to concatenate a list of dataset objects into one dataset
    out_ds = datasets[0].concatenate(datasets[1])
    if len(datasets) > 2:
        for ds in datasets[2:]:
            out_ds = out_ds.concatenate(ds)
    return out_ds



if __name__ == "__main__":
    ''' 
    Define paths to models
    '''
    ## Models to test
    c_name = "lilac-violet-5132"        # done
    f_name = "wise-plasma-5134"         # done
    fc_name = "bumbling-wind-5135"      # done
    done_names = [f_name, c_name, fc_name]

    r_name = "ancient-morning-5137"     # checkpoint
    rc_name = "glorious-silence-5133"   # checkpoint
    fr_name = "peachy-hill-5135"        # checkpoint
    frc_name = "hopeful-violet-5138"    # checkpoint
    frac_name = "curious-totem-5139"    # checkpoint
    chkpt_names = [r_name, rc_name, fr_name, frc_name, frac_name]

    ## Check the model files are stored locally 
    for _n in done_names:
        n_path = f"models/geneAE_{_n}_fold0.keras"
        if not os.path.exists(n_path):
            print(f"{n_path} not found locally")

    ## Check if the checkpoint files are stored locally
    for _n in chkpt_names:
        checkpoints = os.listdir(f"models/checkpoints/{_n}_fold0")
        if len(checkpoints) == 0:
            print(f"{_n} has no local checkpoints")


    '''
    Load the data to test
    '''
    gene_dataset = tf.data.Dataset.load("../data/gene_sequences/training_dataset")
    gene_dataset = gene_dataset.filter(lambda g,d,c: tf.strings.length(g) <= 5000)
    dataset_cuts = [gene_dataset.shard(5, k) for k in range(5)]
    # train_set = concatenate_datasets(*dataset_cuts[:-1])
    val_set = dataset_cuts[-1]

    '''
    Load the model
    '''
    usemodel = frac_name
    chkpt_num = os.listdir(f"models/checkpoints/{usemodel}_fold0/")[0]
    model_path = f"models/checkpoints/{usemodel}_fold0/{chkpt_num}"

    # gene_ae = keras.models.load_model(model_path)


    '''
    Compute the embeddings
    '''
    # train_set, val_set = gene_ae.preprocess_dataset(train_set, batch_size=5, validation_data=val_set, shuffle=False)
    # print(val_set.cardiality)
    # print(gene_ae.encoder.max_length)
    # z = gene_ae.encode(val_set.batch(5))
    # print(z.shape)

    ix = 0 
    genes = []
    domains = []
    cog_cats = []
    for tup in tqdm(val_set):
        # genes.append(tup[0])
        # domains.append(tup[1])
        # cog_cats.append(tup[2])
        ix += 1

    print(ix)
