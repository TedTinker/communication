#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name eft_one_head_arm_b_3 --alpha None --normal_alpha 0.1 --curiosity hidden_state --max_steps 10 --steps_per_epoch 30 --robot_name one_head_arm_b  --agents $agents_per_job --previous_agents $previous_agents
