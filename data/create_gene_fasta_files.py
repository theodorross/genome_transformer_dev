import numpy as np
import pandas as pd
# import dask.dataframe as dd
from tqdm import tqdm
# import tensorflow as tf
import time


def read_seqs(*filepaths):
    outseqs = []
    for fpath in filepaths:
        with open(fpath,"r") as f:
            for l in f.readlines():
                if l != "\n":
                    outseqs.append(l.rstrip("\n"))

    return outseqs

def append_to_file(filepath, text):
    with open(filepath, "a") as f:
        f.write(text)


# ## Define the sequences in the test split
# test_seqs = read_seqs("holdout-seqs/a1.txt",
#                       "holdout-seqs/a2.txt",
#                       "holdout-seqs/b1.txt",
#                       "holdout-seqs/b2.txt")
# test_seqs = set(test_seqs)

## Get the unique sequences and convert them to lines in a text file and fasta files
gene_data_path = "/Users/tro119/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/arete/elac_efm/norwegian_efm_elac_arete_output/pangenomics/panaroo/results/gene_data.csv"
t1 = time.time()
df = pd.read_csv(gene_data_path, index_col="dna_sequence", low_memory=False, 
                 usecols=["dna_sequence","annotation_id","prot_sequence","gff_file","description"])
annotation_mapper = pd.read_csv("gene-annotation-mapper-withsplits.csv", index_col="Annotation_ID")
print("loading:", time.time() - t1)



## Define the output paths and initialize the files
all_genes_fname = "gene_sequences/unique_genes"
training_genes_fname = "gene_sequences/training_genes"
test_genes_fname = "gene_sequences/test_genes"

for _p in [all_genes_fname, training_genes_fname, test_genes_fname]:
    for ext in ["faa", "fna"]:
        with open(f"{_p}.{ext}","w") as f:
            f.write("")


## Initialize the file to store the gene indices
with open("gene_sequences/gene_index_mapper.csv","w") as f:
    f.write("Index,Gene,Split,description,gff_files,annotation_ids\n")


## Loop through all of the 
for ix,dna_seq in enumerate(tqdm(df.index.unique())):
# for ix,dna_seq in enumerate(df.index.unique()):
# for ix,dna_seq in enumerate(df.index.unique()):
# for ix,dna_seq in enumerate(dna_seqs):

    ## Extract information about this gene
    _df = df.loc[dna_seq]

    if isinstance(_df, pd.Series):      # If this specific allele only occurs once
        prot_seq = _df["prot_sequence"]
        annotation_ids = _df["annotation_id"]
        assemblies = _df["gff_file"]
        descriptions = _df["description"]

        split = annotation_mapper.loc[annotation_ids, "Split"]
        gene_name = annotation_mapper.loc[annotation_ids, "Gene"]
    else:                               # If this allele has multiple occurances
        prot_seq = _df["prot_sequence"].tolist()[0]
        annotation_ids = _df["annotation_id"].unique().tolist()
        assemblies = _df["gff_file"].unique().tolist()
        descriptions = ";".join( _df["description"].unique().tolist() )

        split = annotation_mapper.loc[annotation_ids, "Split"].tolist()[0]
        gene_name = annotation_mapper.loc[annotation_ids, "Gene"].tolist()[0]
        annotation_ids = ";".join(annotation_ids)

    ## Add to the index tracker csv
    assemblystr = ";".join(list(assemblies))
    csv_line = f"{ix},{gene_name},{split},{descriptions},{assemblystr},{annotation_ids}\n"
    append_to_file("gene_sequences/gene_index_mapper.csv", csv_line)

    ## Define the fasta header line
    fasta_header=f"\n>{ix}\n"

    ## Add the data to the general data files
    # append_to_file(f"{all_genes_fname}.txt", f"{dna_seq}\n")
    append_to_file(f"{all_genes_fname}.fna", f"{fasta_header}{dna_seq}\n")
    append_to_file(f"{all_genes_fname}.faa", f"{fasta_header}{prot_seq}\n")

    ## Add to the test data files if the gene occurs mainly in test sequences (over 50% of occurances)
    if split == "test":
        # append_to_file(f"{test_genes_fname}.txt", f"{dna_seq}\n")
        append_to_file(f"{test_genes_fname}.fna", f"{fasta_header}{dna_seq}\n")
        append_to_file(f"{test_genes_fname}.faa", f"{fasta_header}{prot_seq}\n")
    elif split == 'training':
        # append_to_file(f"{training_genes_fname}.txt", f"{dna_seq}\n")
        append_to_file(f"{training_genes_fname}.fna", f"{fasta_header}{dna_seq}\n")
        append_to_file(f"{training_genes_fname}.faa", f"{fasta_header}{prot_seq}\n")


    # if ix >= 100:
    #     break
