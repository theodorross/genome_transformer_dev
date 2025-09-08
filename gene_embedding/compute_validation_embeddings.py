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


def generate_embeddings(model_path:str, data:tf.data.Dataset):

    gene_ae = keras.models.load_model(model_path)
    z = gene_ae.encode(data, batch_size=128)

    return z


if __name__ == "__main__":

    ''' 
    Define paths to models
    '''
    ## Models to test
    c_name = "lilac-violet-5132"        # done
    f_name = "wise-plasma-5134"         # done
    fc_name = "bumbling-wind-5135"      # done
    done_names = {'f':f_name, 'c':c_name, 'fc':fc_name}

    r_name = "ancient-morning-5137"     # checkpoint
    rc_name = "glorious-silence-5133"   # checkpoint
    fr_name = "peachy-hill-5135"        # checkpoint
    frc_name = "hopeful-violet-5138"    # checkpoint
    frac_name = "curious-totem-5139"    # checkpoint
    chkpt_names = {'r':r_name, 'rc':rc_name, 'fr':fr_name, 'frc':frc_name, 'frac':frac_name}

    model_paths = {}

    ## Check the model files are stored locally 
    for _l,_n in done_names.items():
        n_path = f"models/geneAE_{_n}_fold0.keras"
        if not os.path.exists(n_path):
            print(f"{n_path} not found locally")
        else:
            model_paths[_l] = n_path

    ## Check if the checkpoint files are stored locally
    for _l,_n in chkpt_names.items():
        checkpoints = os.listdir(f"models/checkpoints/{_n}_fold0")
        if len(checkpoints) == 0:
            print(f"{_n} has no local checkpoints")
        else:
            chkpt_files = os.listdir(f"models/checkpoints/{_n}_fold0")
            latest_chkpt = sorted(chkpt_files)[-1]
            model_paths[_l] = f"models/checkpoints/{_n}_fold0/{latest_chkpt}"


    '''
    Load the data to test
    '''
    gene_dataset = tf.data.Dataset.load("../data/gene_sequences/training_dataset")
    gene_dataset = gene_dataset.filter(lambda g,d,c: tf.strings.length(g) <= 5000)
    dataset_cuts = [gene_dataset.shard(5, k) for k in range(5)]
    train_set = concatenate_datasets(*dataset_cuts[:-1])
    val_set = dataset_cuts[-1]

    ## Preprocess the dataset
    _gene_ae = keras.models.load_model(model_paths['c'])
    train_set, val_set = _gene_ae.preprocess_dataset(train_set, 128, validation_data=val_set, weighted=False, shuffle=False)


    '''
    Compute the embeddings
    '''
    for _l,path in model_paths.items():

        _z = generate_embeddings(path, val_set)
        np.save(f"embeddings/{_l}_{path.split('/')[-1]}.npz", _z)


