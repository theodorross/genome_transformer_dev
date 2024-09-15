#!/bin/bash

rsync -aPu utils springfield:genome_transformer_dev/gene_embedding
rsync -Pu train_gene_autoencoder.py springfield:genome_transformer_dev/gene_embedding

frink run --follow springfield_files/springfield_gene_transformer.yaml

