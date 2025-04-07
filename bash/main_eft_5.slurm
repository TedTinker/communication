#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name eft_5 --alpha None --normal_alpha 0.1 --curiosity hidden_state --be_near_distance 3.5  --agents $agents_per_job --previous_agents $previous_agents
