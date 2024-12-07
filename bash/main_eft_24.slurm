#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name eft_24 --alpha None --normal_alpha 0.1 --curiosity hidden_state --try_batchnorm_1 True --try_batchnorm_2 True --try_batchnorm_3 True --try_batchnorm_4 False --try_batchnorm_5 True  --agents $agents_per_job --previous_agents $previous_agents
