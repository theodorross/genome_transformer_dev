#!\bin\bash

## Ensure the working directory is correct
# cd /project/project_465001381/rosstheo/genome_transformer_dev/gene_embedding
cd /scratch/project_465001915/rosstheo/genome_transformer_dev/gene_embedding

## Update the codebase
git pull origin working

## Submit the batch job(s)
export ARGS_YAML=lumi_files/args_FRC.yml
sbatch lumi_files/LUMI_batch_job.sh

export ARGS_YAML=lumi_files/args_FR.yml
sbatch lumi_files/LUMI_batch_job.sh

export ARGS_YAML=lumi_files/args_FC.yml
sbatch lumi_files/LUMI_batch_job.sh

export ARGS_YAML=lumi_files/args_RC.yml
sbatch lumi_files/LUMI_batch_job.sh

# export ARGS_YAML=lumi_files/args_F.yml
# sbatch lumi_files/LUMI_batch_job.sh

export ARGS_YAML=lumi_files/args_R.yml
sbatch lumi_files/LUMI_batch_job.sh

export ARGS_YAML=lumi_files/args_C.yml
sbatch lumi_files/LUMI_batch_job.sh

