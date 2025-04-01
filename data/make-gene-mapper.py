import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import os
from sklearn.model_selection import train_test_split

'''
Load the data needed
'''
## Load the gene PA matrix
roary_path = "/Users/tro119/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/arete/elac_efm/norwegian_efm_elac_arete_output/pangenomics/panaroo/results/gene_presence_absence.csv"
pa = pd.read_csv(roary_path, index_col="Gene", low_memory=False)
pa = pa.iloc[:,2:]



'''
Create the gene mapper if needed
'''

if not os.path.exists("gene-annotation-mapper_new.csv"):
    ## Load the gene data
    gene_data_path = "/Users/tro119/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/arete/elac_efm/norwegian_efm_elac_arete_output/pangenomics/panaroo/results/gene_data.csv"
    # gene_data_path = "testcsv.csv"
    gene_df = pd.read_csv(gene_data_path, usecols=["annotation_id"])

    '''
    Loop through all gene annotations and find the corresponding gene name
    '''
    ## Create a mapper to convert from annotation_id to gene_name
    mapper_df = pd.DataFrame(index=gene_df['annotation_id'].unique(), columns=["Gene"])
    idx_set = set(gene_df['annotation_id'].tolist())
    print("Making mapper...")

    for row in tqdm(pa.index):
        for col in pa.columns:
            cell = pa.loc[row, col]

            # if isinstance(cell, str):
            if not pd.isna(cell):
                annotation_id = cell.split(";")

                for x in annotation_id:
                    if x not in idx_set:
                        # print(x)
                        new_x = x.replace("_stop","").replace("_len","")
                        mapper_df.loc[new_x,"Gene"] = row
                    else:
                        mapper_df.loc[x,"Gene"] = row

            else:
                print(cell)
    

    '''
    Write the output file
    '''
    print(mapper_df.head())
    print("Number of unique genes listed:", mapper_df["Gene"].nunique())
    print("Number of NaNs:", mapper_df.isna().sum().sum())
    mapper_df.to_csv("gene-annotation-mapper_new.csv", index=True)



'''
Split the data into test and training
'''
df = pd.read_csv("gene-annotation-mapper_new.csv")
df.rename({"Unnamed: 0":"Annotation_ID"}, inplace=True, axis=1)

gene_counts = df["Gene"].value_counts()
train_genes, test_genes = train_test_split(gene_counts.index.to_numpy(), test_size=0.2, random_state=3214)
print("# training genes:", len(train_genes))
print("# training sequences:", gene_counts[train_genes].sum())
print("# test genes:", len(test_genes))
print("# test sequences:", gene_counts[test_genes].sum())

df.set_index("Gene", inplace=True)
df.insert(loc=1, column="Split", value=None)
df.loc[train_genes,'Split'] = "training"
df.loc[test_genes,'Split'] = "test"
df = df.reset_index().set_index("Annotation_ID")
# df.set_index("Annotation_ID", inplace=True)
df.to_csv("gene-annotation-mapper-withsplits.csv")

