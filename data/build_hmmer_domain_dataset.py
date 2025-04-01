import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import numpy as np



def zip_hmmer_and_fasta(fasta_file, hmmer_table):

    '''Load the hmmer dataframe'''
    cols = ['target name','accession1','tlen','query name','accession2','qlen','E-value','score1','1','#','of',
            'c-Evalue','i-Evalue','score2','bias2','from1','to1','from2','to2','from3','to3','acc','description of target']
    hmmer_df = pd.read_csv(hmmer_table, sep=r"\s+", comment="#", names=cols)
    
    # Filter the needed columns only
    hmmer_df = hmmer_df[["target name", "query name", 'E-value', 'c-Evalue', 'i-Evalue']].set_index('target name')

    # Filter only confident 
    hmmer_df = hmmer_df[hmmer_df["E-value"] < 0.05]

    # isolate genes with detected domains
    hmmer_idx_set = set( hmmer_df.index.tolist() )


    '''Load the fasta sequence data'''
    fasta_data = {}
    with open(fasta_file,"r") as f:
        for line in f:
            if ">" in line:
                key = int( line[1:].strip() )
            elif line != "\n":
                fasta_data[key] = line.strip()
    

    '''Build the Tensorflow dataset''' 
    ## Create a template for the one-hot encodings
    _template = pd.Series(0, index=hmmer_df["query name"].unique())

    ## Initialize lists of sequence and domain data
    seq_list = []
    domain_list = []
    
    ## Loop through each gene and record the present domains
    for gene,seq in tqdm(fasta_data.items()):

        # If the current gene has any detected domains
        if gene in hmmer_idx_set:
            vec = _template.copy(deep=True)
            
            domains = hmmer_df.loc[gene, 'query name']
            if isinstance(domains, str):    # Only one detected domain
                vec.loc[domains] = 1
            else:                           # More than one detected domain
                vec.loc[domains.tolist()] = 1

        # If the gene has no detected domains
        else:
            vec = _template.copy(deep=True)

        # Add the protein domain information to the dictionary
        vec = vec.to_numpy()
        seq_list.append(seq)
        domain_list.append(vec)

    ## Create the dataset
    ds = tf.data.Dataset.from_tensor_slices((seq_list, domain_list))
    return ds




if __name__=="__main__":
    test_data = zip_hmmer_and_fasta("gene_sequences/test_genes.fna", "hmmer/dom_test_hmmsearch_Pfam-A.tab")
    train_data = zip_hmmer_and_fasta("gene_sequences/training_genes.fna", "hmmer/dom_training_hmmsearch_Pfam-A.tab")


    ## Save the datasets
    test_data.save("gene_sequences/test_dataset")
    train_data.save("gene_sequences/training_dataset")

