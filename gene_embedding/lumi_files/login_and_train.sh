#!/bin/bash


## Log into wandb
sh ~/wandb_login.sh


## Run the python script
python train_gene_autoencoder.py `cat lumi_files/args.yml`

