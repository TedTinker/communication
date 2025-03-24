#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name eft_1 --alpha None --normal_alpha 0.1 --curiosity hidden_state --max_wheel_speed_for_left 3 --wide_view False --robot_name two_head_arm_a  --agents $agents_per_job --previous_agents $previous_agents
