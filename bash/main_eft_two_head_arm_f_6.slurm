#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name eft_two_head_arm_f_6 --alpha None --normal_alpha 0.1 --curiosity hidden_state --consideration 0 --cnn_upscale True --robot_name two_head_arm_f --max_joint_1_angle 0.7853981633974483 --min_joint_2_angle -1.5707963267948966 --max_joint_2_angle 0  --agents $agents_per_job --previous_agents $previous_agents
