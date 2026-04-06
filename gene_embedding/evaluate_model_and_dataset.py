import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import altair as alt
import os
import pandas as pd
from tqdm import tqdm

import time

import argparse



def eval_cog_preds(model, dataset):
    '''
    Evaluate the COG category predictions of the gene autoencoder model.
    
    Inputs:
        model (GeneTransformer) : keras model for the gene transformer model.
        dataset (tensorflow.data.Dataset) : tensorflow dataset containing the genes, COG categories, and protein domains.
    '''

    ## Initialize the lists to store data
    genes = []
    cog_categories = []

    ## Populate the lists
    for t in tqdm(dataset):
        # dataset.append(g)
        genes.append(t[0])
        cog_categories.append(t[1])

    for x in t:
        print(x)
    exit()

    ## Compute the predictions
    _geneseqs = tf.data.Dataset.from_tensor_slices(genes)
    _,cog_preds,_ = model.predict(_geneseqs)

    ## Define prediction and label arrays
    cog_preds = (cog_preds > 0.5).astype(int)
    correct_mask = np.all(cog_preds == cog_categories, axis=1)
    print(f"genes completely correct:\t\t {correct_mask.sum():5d}\t ({correct_mask.mean()*100:3.2f}%)")

    ## Find all unique combination of COG category
    categs = np.asarray([c for c in 'ABCDEFGHIJKLMNOPQTUVWYZ'])
    true_combo_vec, pred_combo_vec = [], []



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
    print("loading model...", end=" ")
    t1 = time.time()
    gene_ae = tf.keras.models.load_model(args.model_path)
    print(f"done. ({time.time()-t1:0.4f})")
    # gene_ae = None

    ## Load the dataset
    print("loading dataset...", end=" ")
    t1 = time.time()
    ds = tf.data.Dataset.load(args.dataset_path)
    print(f"done. ({time.time()-t1:0.4f})")

    # Filter the dataset to only keep genes up to 5000 nucleotides long and 
    gene_dataset = ds.filter(lambda g,d,c: tf.strings.length(g) <= 5000)

    # Preprocess the dataset
    print("preprocessing...", end=" ")
    t1 = time.time()
    gene_dataset = gene_ae.preprocess_dataset(gene_dataset, batch_size=4, weighted=False, shuffle=False)
    print(f"done. ({time.time()-t1:0.4f})")


    '''
    Evalueate COG category predictions
    '''
    eval_cog_preds(gene_ae, gene_dataset)






