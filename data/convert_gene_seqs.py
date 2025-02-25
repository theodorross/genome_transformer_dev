import numpy as np
import pandas as pd


## Get the unique sequences and convert them to lines in a text file
df = pd.read_csv("gene_sequences/gene_data.csv")

unique_genes = df["dna_sequence"].unique()

outstr = "\n".join(unique_genes)

with open("gene_sequences/unique_dna_seqs.txt","w") as f:
    f.write(outstr)

