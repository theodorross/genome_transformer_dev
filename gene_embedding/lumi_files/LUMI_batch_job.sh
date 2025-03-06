#!/bin/bash
#SBATCH --job-name=GeneAE
#SBATCH --output=lumi_files/logs/gene_embedding.o%j # Name of stdout output file
#SBATCH --error=lumi_files/logs/gene_embedding.e%j  # Name of stderr error file
#SBATCH --account=project_465001381
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --partition=standard-g


## Load the needed LUMI bindings
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
# module use /appl/local/training/modules/AI-20240529
# module load singularity-CPEbits

# To have RCCL use the Slingshot interfaces:
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# To have RCCL use GPU RDMA:
export NCCL_NET_GDR_LEVEL=PHB

## Define CPU binding
# CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

## Define directories of interest
GITDIR=/project/project_465001381/rosstheo/genome_transformer_dev
export SIF=/project/project_465001381/rosstheo/genome_transformer_dev/container/lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1.sif

## Run the training script
srun singularity exec \
    -B /project/project_465001381/rosstheo \
    $GITDIR/container/tensorflow-experiments_rocm-v0.3.sif /bin/bash \
    lumi_files/login_and_train.sh


