#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name eft_8 --alpha None --normal_alpha 0.1 --curiosity hidden_state --global_push_amount 0.2 --global_left_right_amount 0.3  --agents $agents_per_job --previous_agents $previous_agents
