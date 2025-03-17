#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name et_slow_2 --alpha None --normal_alpha 0.1 --step_cost 1 --processor wtpulr --time_step 0.1 --global_push_amount 0.1 --global_pull_amount 0.1 --global_left_right_amount 0.1  --agents $agents_per_job --previous_agents $previous_agents
