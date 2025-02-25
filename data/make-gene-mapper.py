import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

'''
Load the data needed
'''
## Load the roary gene PA matrix
roary_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/panaroo/norwegian/output/gene_presence_absence_roary.csv"
roary_df = pd.read_csv(roary_path, index_col="Gene", low_memory=False)

## Load the gene data
gene_data_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/panaroo/norwegian/output/gene_data.csv"
# gene_data_path = "testcsv.csv"
gene_df = pd.read_csv(gene_data_path, index_col="annotation_id")



'''
Loop through all gene annotations and find the corresponding gene name
'''
## Create a mapper to convert from annotation_id to gene_name
mapper_df = pd.DataFrame(index=gene_df.index, columns=["Gene"])
print("Making mapper...")
for anno_idx in tqdm(mapper_df.index):

    if "refound" not in anno_idx:
        # Isolate only the relevant data
        seq_name = anno_idx.rsplit("_",1)[0]
        sub_df = roary_df[[seq_name]].copy()
        sub_df[seq_name] = sub_df[seq_name].str.split(";").tolist()
        sub_df = sub_df.explode(seq_name)

        # Find the matching annotation_id and its corresponding gene
        row,_ = np.where(sub_df==anno_idx)
        gene_name = sub_df.index[row].tolist()
    else: 
        row,_ = np.where(roary_df==anno_idx)
        gene_name = roary_df.index[row].tolist()

    if len(gene_name) == 1:
        mapper_df.loc[anno_idx,"Gene"] = gene_name[0]
    elif len(gene_name) == 0:
        continue
    else:
        print("Too many genes mapped...")
    

'''
Write the output file
'''
print(mapper_df.head())
print("Number of unique genes listed:", mapper_df["Gene"].nunique())
print("Number of NaNs:", mapper_df.isna().sum().sum())
mapper_df.to_csv("gene-annotation-mapper_new.csv", index=True)
