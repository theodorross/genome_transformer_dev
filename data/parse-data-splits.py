import pandas as pd
import numpy as np
import os



def read_seqs(*filepaths):
    outseqs = []
    for fpath in filepaths:
        with open(fpath,"r") as f:
            for l in f.readlines():
                if l != "\n":
                    outseqs.append(l.rstrip("\n"))

    return outseqs



## Load data
# metadata_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/combined/combined-microreact-labels.csv"
# metadata = pd.read_csv(metadata_path, index_col="id")
# metadata.drop("")
assemblypath = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/assemblies/Norwegian_Efm_genomes_assemblies"
assembly_list = [f.rsplit(".",1)[0] for f in os.listdir(assemblypath)]


## Define the train/test splits
test_seqs = read_seqs("holdout-seqs/a1.txt",
                      "holdout-seqs/a2.txt",
                      "holdout-seqs/b1.txt",
                      "holdout-seqs/b2.txt")

# train_seqs = [s for s in metadata.index if s not in test_seqs]
train_seqs = [s for s in assembly_list if s not in test_seqs]

# print("NZ_CP038996" in assembly_list)
# print(len(test_seqs), len(train_seqs), len(test_seqs)+len(train_seqs))


## Define gene sequences in each set
# gene_df = pd.read_csv("gene_sequences/gene_data.csv", nrows=10000)
# print(gene_df.columns)
gene_df = pd.read_csv("gene_sequences/gene_data.csv", index_col="gff_file", low_memory=False)
test_genes_seqs = gene_df.loc[test_seqs,"dna_sequence"].unique().tolist()
train_genes_seqs = gene_df.loc[train_seqs,"dna_sequence"].unique().tolist()


## Write files for each split
# Sequence names
split_df = pd.DataFrame(index = assembly_list, columns=["Split"])
split_df.index.name = "id"
split_df.loc[train_seqs,"Split"] = "training"
split_df.loc[test_seqs,"Split"] = "test"
split_df.to_csv("assembly_splits.csv", index=True)


# Genes
with open("gene_sequences/test_unique_gene_seqs.txt","w") as f:
    f.write("\n".join(test_genes_seqs))

with open("gene_sequences/train_unique_gene_seqs.txt","w") as f:
    f.write("\n".join(train_genes_seqs))

