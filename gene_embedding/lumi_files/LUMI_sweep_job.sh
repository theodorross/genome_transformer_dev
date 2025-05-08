#!/bin/bash
#SBATCH --job-name=GeneAE_sweep
#SBATCH --output=lumi_files/logs/gene_embedding_sweep.o%j # Name of stdout output file
#SBATCH --error=lumi_files/logs/gene_embedding_sweep.e%j  # Name of stderr error file
#SBATCH --account=project_465001915
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --partition=small-g


## Load the needed LUMI bindings
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
# module use /appl/local/training/modules/AI-20240529
# module load singularity-CPEbits

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID

# To have RCCL use the Slingshot interfaces:
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# To have RCCL use GPU RDMA:
export NCCL_NET_GDR_LEVEL=PHB
export NCLL_DEBUG=WARN

## Define CPU binding
# CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

## Define directories of interest
GITDIR=/scratch/project_465001915/rosstheo/genome_transformer_dev
export SIF=/scratch/project_465001915/rosstheo/genome_transformer_dev/container/lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1.sif


## Initialize the sweep
# singularity exec \
#     -B /scratch/project_465001915/rosstheo \
#     $SIF /bin/bash \
#     -c '$WITH_CONDA && source wandb-env/bin/activate && sh ~/wandb_login.sh && python -m wandb sweep lumi_files/sweep_args.yml &> lumi_files/sweep_info.txt'
# SWEEP_ID=`grep -o -e "ID: [[:alnum:]]*" lumi_files/sweep_info.txt | sed 's/.*ID: //'`
SWEEP_ID='e7o8pguv'


## Run the training script
srun singularity exec \
    -B /scratch/project_465001915/rosstheo \
    --env SWEEP_ID=$SWEEP_ID \
    $SIF /bin/bash \
    lumi_files/login_and_sweep.sh
