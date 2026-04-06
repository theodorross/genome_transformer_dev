import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import altair as alt
import os
import pandas as pd

import argparse



def eval_cog_preds(model, dataset):
    '''
    Evaluate the COG category predictions of the gene autoencoder model.
    
    Inputs:
        model ()
    '''

    print(type(model))
    print(type(dataset))


    return






if __name__ == "__main__":
    keras.config.disable_traceback_filtering()

    '''
    Initialize run parameters
    '''
    parser = argparse.ArgumentParser()

    ## System parameters
    parser.add_argument("--model-path", default="models/geneAE_curious-totem-5139_fold0.keras", type=str, help="Which keras model to load and test.", required=False)
    parser.add_argument("--dataset-path", default="../data/test-species/tensorflow_datasets/enterococcus-avium_dataset", type=str, help="Dataset to evalueate on.", required=False)
    args = parser.parse_args()


    '''
    Load the specified model and dataset
    '''
    ## Load the model
    gene_ae = tf.keras.models.load_model(args.model_path)
    # gene_ae = None

    ## Load the dataset
    ds = tf.data.Dataset.load(args.dataset_path)
    
    # Filter the dataset to only keep genes up to 5000 nucleotides long
    gene_dataset = ds.filter(lambda g,d,c: tf.strings.length(g) <= 5000)


    '''
    Evalueate COG category predictions
    '''
    eval_cog_preds(gene_ae, ds)






