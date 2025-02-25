#!/bin/bash


## Log into wandb
sh ~/wandb_login.sh
# WANDB_MODE=offline


## Run the python script
python train_gene_autoencoder.py `cat lumi_files/args.yml`

