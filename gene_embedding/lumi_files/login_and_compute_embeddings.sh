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

## Run the python script
python compute_validation_embeddings.py 
