#!/bin/bash


## Log into wandb
sh ~/wandb_login.sh


## Run the python script
train_gene_autoencoder.py < $WORKDIR/lumi_files/args.yml

