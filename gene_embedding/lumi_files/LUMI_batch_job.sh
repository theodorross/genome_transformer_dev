#!/bin/bash
#SBATCH --job-name=GeneAE
#SBATCH --output=lumi_files/logs/gene_embedding.o%j # Name of stdout output file
#SBATCH --error=lumi_files/logs/gene_embedding.e%j  # Name of stderr error file
#SBATCH --account=project_465001381
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=small-g


## Define directories of interest
GITDIR=/project/project_465001381/rosstheo/genome_transformer_dev

## Run the training scrip
srun singularity exec \
    -B /project/project_465001381/rosstheo \
    $GITDIR/container/tensorflow-experiments_rocm-v0.3.sif /bin/bash \
    lumi_files/login_and_train.sh


