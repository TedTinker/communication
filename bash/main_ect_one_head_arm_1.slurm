#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name ect_one_head_arm_1 --alpha None --normal_alpha 0.1 --curiosity hidden_state --hidden_state_eta_rgbd 0 --hidden_state_eta_sensors 0 --smooth_steps False --consideration False --robot_name one_head_arm --max_joint_1_angle 0.7853981633974483  --agents $agents_per_job --previous_agents $previous_agents
