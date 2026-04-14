import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import altair as alt
import os
import pandas as pd
from tqdm import tqdm
import time
import faiss
from sklearn import metrics

# from scipy.stats import normaltest
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LogisticRegression
# import Levenshtein

import argparse

alt.data_transformers.disable_max_rows()



def eval_cog_preds(preds, labels):
    '''
    Evaluate the COG category predictions of the gene autoencoder model.
    
    Inputs:
        model (GeneTransformer) : keras model for the gene transformer model.
        dataset (tensorflow.data.Dataset) : tensorflow dataset containing the genes, COG categories, and protein domains.
    '''

    ## Define prediction and label arrays
    preds = (preds > 0.5).astype(int)
    correct_mask = np.all(preds == labels, axis=1)
    
    ## Isolate genes that had functional labels
    labeled_mask = labels.sum(axis=1) > 0
    labeled_preds = preds[labeled_mask]
    labeled_labels = labels[labeled_mask].astype(int)

    full_correct = np.all(labeled_preds == labeled_labels, axis=1)
    not_full_corr_mask = ~np.all(labeled_preds == labeled_labels, axis=1)
    labelled_no_pred = labeled_preds.sum(axis=1) == 0

    ## Create a csv for saving summary information
    csv_str = "metric,total,percent\n"
    csv_str += f"total genes,{len(preds)},\n"

    csv_str += "genes completely correct,"
    csv_str += f"{full_correct.sum()},"
    csv_str += f"{full_correct.mean()*100}\n"

    csv_str += "genes with labels,"
    csv_str += f"{(labels.sum(axis=1)!=0).sum()},"
    csv_str += f"{(labels.sum(axis=1)!=0).mean()*100}\n"

    csv_str += "genes with no prediction,"
    csv_str += f"{(preds.sum(axis=1)==0).sum()},"
    csv_str += f"{(preds.sum(axis=1)==0).mean()*100}\n"

    csv_str += "genes with labels and no prediction,"
    csv_str += f"{labelled_no_pred.sum()},"
    csv_str += f"{labelled_no_pred.mean()*100}\n"

    ## Find all unique combination of COG category
    categs = np.asarray([c for c in 'ABCDEFGHIJKLMNOPQTUVWYZ'])
    categ_df = pd.DataFrame(index=categs, columns=["Count", "F1", "Precision", "Recall", "Accuracy"])
    categ_df.index.name = "Category"

    for c in range(labels.shape[1]):
        _true = labeled_labels[:,c]
        cat_letter = categs[c]

        if _true.sum() != 0:
            # _pred = val_cat_preds[:,c]
            _pred = labeled_preds[:,c]
    
            categ_df.loc[cat_letter,"Count"] = _true.sum()
            categ_df.loc[cat_letter,"F1"] = metrics.f1_score(_true,_pred)
            categ_df.loc[cat_letter,"Precision"] = metrics.precision_score(_true,_pred)
            categ_df.loc[cat_letter,"Recall"] = metrics.recall_score(_true,_pred)
            categ_df.loc[cat_letter,"Accuracy"] = metrics.accuracy_score(_true,_pred)   

    return csv_str, categ_df



def eval_reconstructions(preds, labels, vocab):
    '''
    Evaluate the gene reconstructions.
    '''
    ## Define the vocabulary and the reconstruction indices
    pred_idx = np.argmax(preds, axis=2)

    ## Compute the gene lengths and the reconstruction accuracies per gene
    true_tokens = pred_idx == labels
    mask = labels != 0
    gene_lens = mask.sum(axis=1)
    gene_accs = true_tokens.sum(axis=1) / gene_lens

    # fig,ax = plt.subplots(1,1)
    # ax.hist(gene_accs, bins=50)
    # ax.set_xlabel("Recon. Acc.")
    # # fig.savefig("temp.png")

    ## Compute the GC content of each gene
    c_mask = labels == 4
    g_mask = labels == 5
    gc_mask = c_mask | g_mask
    gc_frac = gc_mask.sum(axis=1) / mask.sum(axis=1)

    ## Create a histogram of the reconstruction accuracies
    df = pd.DataFrame({"recon_acc":gene_accs,
                       "gene_lens":gene_lens,
                       "gc_frac":gc_frac})
    base = alt.Chart(df)
    bar_hist = base.mark_bar().encode(
        x = alt.X("recon_acc:Q", title="Reconstruction Accuracy", bin=alt.Bin(maxbins=30)),
        y = alt.Y("count()")
    )
    len_plot = base.mark_circle().encode(
        x = alt.X("gene_lens:Q", title="Gene Length (nt)"),
        y = alt.Y("recon_acc:Q", title="Reconstruction Accuracy")
    )
    gc_plot = base.mark_circle().encode(
        x = alt.X("gc_frac:Q", title="GC Content"),
        y = alt.Y("recon_acc:Q", title="Reconstruction Accuracy")
    )  
    out_plot = bar_hist | len_plot | gc_plot
    
    return df, out_plot



def eval_domains(z, y):
    '''
    Evaluate the domain clusterings.
    '''

    ## Initialize FAISS
    faiss_index = faiss.IndexFlatL2(2048)
    faiss_index.add(z)

    ## Loop through each domain to compute centers
    domain_centers = np.zeros((y.shape[1],2048))
    domain_counts = []
    nearest_neighbor_matching_fracs = []
    
    # for dx in tqdm(range(y.shape[1])[-50:]):
    for dx in tqdm(range(y.shape[1])):

        ## Find the genes containing this domain
        domain_mask = y[:,dx].astype(bool)
        n_domains = int( domain_mask.sum() )

        if n_domains != 0:
            ## Find the mean/center of this domain's embeddings
            domain_mean = z[domain_mask,:].mean(axis=0)[None,:]

            ## Find the n_domains closest embeddings to the center
            D,I = faiss_index.search(domain_mean, n_domains)

            ## Determine if the n_domains nearest neighbors contain the query domain
            label_match_flags = y[np.squeeze(I), dx]
            nearest_neighbor_matching_fracs.append(np.mean(label_match_flags))
            domain_counts.append(n_domains)

    ## Make an output plot
    # Define the dataframe to plot
    df = pd.DataFrame({'count':domain_counts, 
                       'frac':nearest_neighbor_matching_fracs})

    # Define the chart bases and scales
    base = alt.Chart(df)
    bar_base = base.mark_bar(binSpacing=0)
    
    frac_scale = alt.Scale(domain=(0,1))
    count_scale = alt.Scale(domain=(0,max(domain_counts)*1.05))
    
    # Create the scatterplot
    points = base.mark_circle().encode(
        x = alt.X('count:Q', title="Total domain instances").scale(count_scale, type="log"),
        y = alt.Y('frac:Q', title="Fraction included in nearest neighbors").scale(frac_scale)
    )

    # Create a hitogram of the counts
    top_bar = bar_base.encode(
        x = alt.X('count:Q').scale(count_scale, type="log").bin(
            maxbins=30, extent=count_scale.domain
        ).title(""),
        y = alt.Y('count():Q').stack(None).title("")
    ).properties(height=60)

    # Create a histogram of the fractions
    right_bar = bar_base.encode(
        y = alt.Y('frac:Q').scale(frac_scale).bin(
            maxbins=30, extent=frac_scale.domain
        ).title(""),
        x = alt.X('count():Q').stack(None).title("")
    ).properties(width=60)

    # Format the final chart
    chart = top_bar & (points | right_bar)

    return df, chart





if __name__ == "__main__":
    keras.config.disable_traceback_filtering()

    '''
    Initialize run parameters
    '''
    parser = argparse.ArgumentParser()

    ## System parameters
    parser.add_argument("--model-path", default="models/geneAE_curious-totem-5139_fold0.keras", type=str, help="Which keras model to load and test.", required=False)
    parser.add_argument("--dataset-path", default="../data/test-species/tensorflow_datasets/vagococcus-avium_dataset", type=str, help="Dataset to evalueate on.", required=False)
    # # defdir = "/scratch/project_465002309/rosstheo/genome_transformer_dev/data/gene_sequences/training_dataset"
    # defdir = "/scratch/project_465002309/rosstheo/genome_transformer_dev/data/gene_sequences/test_dataset"
    # parser.add_argument("--dataset-path", default=defdir, type=str, help="Dataset to evalueate on.", required=False)
    args = parser.parse_args()
    print("Model:", args.model_path)
    print("Dataset:", args.dataset_path)


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
    # gene_dataset = [gene_dataset.shard(5, k) for k in range(5)][-1]     # Only for training dataset
    _,gene_dataset = gene_ae.preprocess_dataset(gene_dataset, validation_data=gene_dataset, batch_size=64, weighted=False, shuffle=False)
    print(f"done. ({time.time()-t1:0.4f})")

    
    '''
    Collect label data
    '''
    ## Initialize the lists to store data
    # val_genes = []
    cog_labels = []
    domain_labels = []
    reconstruction_targets = []
    
    ## Populate the lists
    for g,labels in tqdm(gene_dataset):
        # val_genes.append(g)
        
        _domain,_cat,_recon_target = labels
        cog_labels.append(_cat)
        domain_labels.append(_domain)
        reconstruction_targets.append(_recon_target)
    
    # ## Convert them to arrays
    # val_genes = np.concatenate(val_genes, axis=0)
    cog_labels = np.concatenate(cog_labels, axis=0)
    domain_labels = np.concatenate(domain_labels, axis=0)
    reconstruction_targets = np.concatenate(reconstruction_targets, axis=0)

    '''
    Compute predictions
    '''
    embeddings, cog_preds, reconstructions = gene_ae.predict(gene_dataset)

    
    '''
    Evaluate the model performance
    '''
    cog_summary,categ_df = eval_cog_preds(cog_preds, cog_labels)
    reconstruction_df, recon_chart = eval_reconstructions(reconstructions, reconstruction_targets, gene_ae.encoder.vocabulary)
    domain_df, domain_chart = eval_domains(embeddings, domain_labels)


    '''
    Save the outputs
    '''
    ## Define the save path and ensure it exists
    model_name = args.model_path.split("/")[-1].split("_")[1]
    dataset_name = args.dataset_path.split("/")[-1].replace("_dataset", "")
    savepath = f"evaluation/{model_name}/{dataset_name}"

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    print(f"Saving to {savepath}")

    ## Save the COG prediction information
    with open(f"{savepath}/cog_summary.csv","w") as f:
        f.write(cog_summary)
    categ_df.to_csv(f"{savepath}/cog_categories_scores.csv")

    ## Save domain reconstruction performance information
    reconstruction_df.to_csv(f"{savepath}/reconstruction_data.csv")
    recon_chart.save(f"{savepath}/reconstruction_fig.png")

    ## Save the domain clustering performance
    domain_chart.save(f"{savepath}/domain_clustering_fig.png")
    domain_df.to_csv(f"{savepath}/domain_clustering_summary.csv")


    




