#!\bin\bash

## Update the codebase
git pull origin working

cd /project/project_465001381/rosstheo/gneome_transformer_dev/gene_embedding
sbatch lumi_files/LUMI_batch_job.sh

