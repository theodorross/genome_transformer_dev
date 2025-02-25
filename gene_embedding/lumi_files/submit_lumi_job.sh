#!\bin\bash

## Update the codebase
git pull origin working

## Ensure the working directory is correct
cd /project/project_465001381/rosstheo/genome_transformer_dev/gene_embedding

## Submit the batch job
sbatch lumi_files/LUMI_batch_job.sh

