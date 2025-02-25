import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

'''
- Need gene_data.csv for the "annotation_id" and "scaffold_name"
- Need gene_presence_absence_roary.csv to map the "Annotation" to "Gene""

'''

'''
Load the data needed
'''
## Get the list of unique genes
pa_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/panaroo/norwegian/output/gene_presence_absence.Rtab"
pa_df = pd.read_csv(pa_path, index_col="Gene", sep="\t", low_memory=True)
unique_genes = pa_df.index.tolist()
del pa_df

## Load the gene data
gene_data_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/panaroo/norwegian/output/gene_data.csv"
# gene_data_path = "testcsv.csv"
gene_df = pd.read_csv(gene_data_path, index_col="annotation_id", low_memory=False)

## Read the annotatio mapper
mapper_df = pd.read_csv("data/gene-annotation-mapper_new.csv", index_col="annotation_id")


'''
Loop through all unique contigs
'''
## Initialize the output dataframe
seq_df = pd.DataFrame("", index=gene_df["scaffold_name"].unique(), columns=["sequence"])
out_df = pd.DataFrame(0, index=gene_df["scaffold_name"].unique(), columns=unique_genes)
# out_df["sequence"] = out_df["sequence"].astype("string")
seq_df.index.name = "scaffold"
out_df.index.name = "scaffold"
print(out_df.head())

debug_set = set(mapper_df.index)

# for contig_name in tqdm(gene_df["scaffold_name"].unique()):
for contig_name in tqdm(gene_df["scaffold_name"].unique()):

    ## Isolate only the annotations corresponding to the current contig
    sub_df = gene_df[gene_df["scaffold_name"]==contig_name]

    ## Define the sequence the contig belongs to
    seq = sub_df.iloc[0,0]
    seq_df.loc[contig_name, "sequence"] = seq

    ## Loop through all the annotations in the subsetted dataframe
    for anno_idx in sub_df.index:

        gene_name = mapper_df.loc[anno_idx, "Gene"]

        ## If the gene was found in the presence/absence matrix
        if isinstance(gene_name, str):
            out_df.loc[contig_name, gene_name] = 1
            # out_df.loc[contig_name, "sequence"] = seq

## Save the output
out_df = pd.concat([seq_df, out_df], axis='columns')
out_df.to_csv("contig-gene-presence-absence_new.csv", index=True)
print(out_df.head())
