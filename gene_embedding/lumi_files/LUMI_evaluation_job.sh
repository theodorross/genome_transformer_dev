#!/bin/bash
#SBATCH --job-name=EvalGeneAE
#SBATCH --output=lumi_files/logs/evaluate_ae.o%j # Name of stdout output file
#SBATCH --error=lumi_files/logs/evaluate_ae.e%j  # Name of stderr error file
#SBATCH --account=project_465002861
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --partition=small-g


## Load the needed LUMI bindings
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

export SIF=/scratch/project_465002309/rosstheo/genome_transformer_dev/container/lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1.sif

## Run the script
srun singularity exec \
    -B /scratch/project_465002861/rosstheo \
    $SIF /bin/bash \
    lumi_files/evaluate_test_data.sh


