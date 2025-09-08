#!\bin\bash

## Ensure the working directory is correct
cd /scratch/project_465001915/rosstheo/genome_transformer_dev/gene_embedding

## Update the codebase
git pull origin working

## Sumbit the batch job
sbatch lumi_files/LUMI_embedding_job.sh
