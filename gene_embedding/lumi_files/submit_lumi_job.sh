#!\bin\bash

## Ensure the working directory is correct
# cd /project/project_465001381/rosstheo/genome_transformer_dev/gene_embedding
cd /scratch/project_465001915/rosstheo/genome_transformer_dev/gene_embedding

## Update the codebase
git pull origin working

## Submit the batch job(s)
# export ARGS_YAML=lumi_files/args_FRC
# sbatch lumi_files/LUMI_batch_job.sh

export ARGS_YAML=lumi_files/args_FR
sbatch lumi_files/LUMI_batch_job.sh

# export ARGS_YAML=lumi_files/args_FC
# sbatch lumi_files/LUMI_batch_job.sh

# export ARGS_YAML=lumi_files/args_F
# sbatch lumi_files/LUMI_batch_job.sh

