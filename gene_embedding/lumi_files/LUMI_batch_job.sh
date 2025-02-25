#!/bin/bash
#SBATCH --job-name=Genomad
#SBATCH --output=~/genome_transformer_logs/gene_embedding.o%j # Name of stdout output file
#SBATCH --error=~/genome_transformer_logs/gene_embedding.e%j  # Name of stderr error file
#SBATCH --account=project_465001381
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=small-g


## Log into wandb
sh ~/wandb_login.sh


## Define directories of interest
GITDIR=/project/project_465001381/rosstheo/gneome_transformer_dev
WORKDIR=/project/project_465001381/rosstheo/gneome_transformer_dev/gene_embedding

## Run the training scrip
srun singularity exec \
    -B /project/project_465001381/rosstheo \
    $GITDIR/container/tensorflow-experiments_rocm-v0.3.sif python \
    train_gene_autoencoder.py < $WORKDIR/lumi_files/args.txt





## GENOMAD EXAMPLE
# srun singularity exec \
#     -B /project/project_465000263/rosstheo \
#     -B /scratch/project_465000263/rosstheo \
#     genomad_latest.sif /bin/bash \
#     /project/project_465000263/rosstheo/Expanded_dataset_analysis/genomad/lumi-genomad.sh


