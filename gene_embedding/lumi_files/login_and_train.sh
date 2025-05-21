#!/bin/bash

## Export SLURM stuff
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID

## Activate the venv
$WITH_CONDA
source wandb-env/bin/activate

## Log into wandb
sh ~/wandb_login.sh
# WANDB_MODE=offline

cat $ARGS_YAML

## Run the python script
# python train_gene_autoencoder.py `cat lumi_files/args_FRC.yml`
# python train_gene_autoencoder.py `cat $ARGS_YAML`

