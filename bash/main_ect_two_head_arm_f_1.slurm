#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name ect_two_head_arm_f_1 --alpha None --normal_alpha 0.1 --curiosity hidden_state --hidden_state_eta_rgbd 0 --hidden_state_eta_sensors 0 --consideration 0 --cnn_upscale False --robot_name two_head_arm_f --max_joint_1_angle 0.7853981633974483 --min_joint_2_angle -1.5707963267948966 --max_joint_2_angle 0  --agents $agents_per_job --previous_agents $previous_agents
