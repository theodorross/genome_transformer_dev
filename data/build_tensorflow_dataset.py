import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import numpy as np


def category_mapper(s:str):
    '''Convert a string of COG categories to an encoded vector'''
    ## Initialize the output vector
    out_vec = np.zeros(23)
    letters = "ABCDEFGHIJKLMNOPQTUVWYZ"
    pos_mapper = {l:ix for ix,l in enumerate(letters)}

    ## Set the corresponding indices of the output vector to 1
    for letter in s.upper():
        if letter in letters:
            out_vec[pos_mapper[letter]] = 1
    return out_vec


def zip_hmmer_and_fasta(fasta_file, hmmer_table, eggnog_table):

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


    '''Load COG category data'''
    eggnog_cols = ['query','seed_ortholog','evalue','score','eggNOG_OGs','max_annot_lvl','COG_category',
                   'Description','Preferred_name','GOs','EC','KEGG_ko','KEGG_Pathway','KEGG_Module','KEGG_Reaction',
                   'KEGG_rclass','BRITE','KEGG_TC','CAZy','BiGG_Reaction','PFAMs']
    eggnog_df = pd.read_csv(eggnog_table, comment="#", names=eggnog_cols, sep="\t", usecols=["query","COG_category"], index_col="query")
    cog_set = set([])
    for _c in eggnog_df["COG_category"].unique():
        cog_set |= set(_c)

    ## define the indices
    eggnog_idx_set = set(eggnog_df.index)
    

    '''Build the Tensorflow dataset''' 
    ## Create a template for the one-hot encodings
    _template = pd.Series(0, index=hmmer_df["query name"].unique())
    # print("saving...")
    # _template.to_csv("hmmer_template.csv")
    # exit()

    ## Initialize lists of sequence and domain data
    seq_list = []
    domain_list = []
    category_list = []
    
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

        # If the current gene has any predicted cog categories
        if gene in eggnog_idx_set:
            category_vec = category_mapper(eggnog_df.loc[gene, "COG_category"])
        else:
            category_vec = category_mapper("S")

        # Add the protein domain information to the dictionary
        vec = vec.to_numpy()
        seq_list.append(seq)
        domain_list.append(vec)
        category_list.append(category_vec)

    ## Create the dataset
    ds = tf.data.Dataset.from_tensor_slices((seq_list, domain_list, category_list))
    return ds




if __name__=="__main__":
    ## Build and save the test data
    # test_data = zip_hmmer_and_fasta("gene_sequences/test_genes.fna", 
    #                                 "hmmer/dom_test_hmmsearch_Pfam-A.tab", 
    #                                 "eggnog/output_test.emapper.annotations")
    # test_data.save("gene_sequences/test_dataset")
    # del test_data

    ## Build and save the training data
    train_data = zip_hmmer_and_fasta("gene_sequences/training_genes.fna", 
                                     "hmmer/dom_training_hmmsearch_Pfam-A.tab",
                                     "eggnog/output_train.emapper.annotations")
    train_data.save("gene_sequences/training_dataset")

