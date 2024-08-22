#!/bin/bash
#SBATCH --job-name=GenomeEmbed
#SBATCH --output=Expanded_dataset_analysis/genome_transformer_dev/cog-pa-test.o%j # Name of stdout output file
#SBATCH --error=Expanded_dataset_analysis/genome_transformer_dev/cog-pa-test.e%j  # Name of stderr error file
#SBATCH --account=project_465000263
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --mem=128G
#SBATCH --cpus-per-task=56
#SBATCH --partition=standard-g

### WITH CONTAINER
# cd /scratch/project_465000263/rosstheo/genome_transformer_dev/cog_presence

# cat << EOF > select_gpu
# #!/bin/bash
# export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
# singularity exec \$*
# EOF
# chmod +x ./select_gpu

# CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
# export MPICH_GPU_SUPPORT_ENABLED=1

# srun singularity exec \
#     -B /project/project_465000263/rosstheo \
#     -B /scratch/project_465000263/rosstheo \
#     ../container/tensorflow-experiments_rocm-v0.1.sif \
#     python cog-presence-test.py \
#     --platform "LUMI" \
#     --embedding-dim 1024 \
#     --label-column Vancomycin_res

### WITH MODULES
module use /appl/local/csc/modulefiles/
module load tensorflow
module load rocm

cd /scratch/project_465000263/rosstheo/genome_transformer_dev/cog_presence

srun python3 cog-presence-test.py \
    --platform "LUMI" \
    --embedding-dim 1024 \
    --label-column Vancomycin_res \
    --max-contigs 500
