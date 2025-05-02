#!\bin\bash

## Ensure the working directory is correct
# cd /project/project_465001381/rosstheo/genome_transformer_dev/gene_embedding
cd /scratch/project_465001915/rosstheo/genome_transformer_dev/gene_embedding

## Update the codebase
git pull origin working

## Submit the batch job(s)
sbatch lumi_files/LUMI_sweep_job.sh
sbatch lumi_files/LUMI_sweep_job.sh
sbatch lumi_files/LUMI_sweep_job.sh
sbatch lumi_files/LUMI_sweep_job.sh
sbatch lumi_files/LUMI_sweep_job.sh
sbatch lumi_files/LUMI_sweep_job.sh
sbatch lumi_files/LUMI_sweep_job.sh
sbatch lumi_files/LUMI_sweep_job.sh
