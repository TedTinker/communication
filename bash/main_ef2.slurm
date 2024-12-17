#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name ef2 --alpha None --normal_alpha 0.1 --curiosity hidden_state --max_steps 20 --time_step 0.1 --steps_per_step 10 --push_amount 0.375 --pull_amount 0.125 --left_right_amount 0.125  --agents $agents_per_job --previous_agents $previous_agents
