#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name ect_two_head_arm_g_4 --alpha None --normal_alpha 0.1 --curiosity hidden_state --hidden_state_eta_vision 0 --hidden_state_eta_touch 0 --max_steps 30 --steps_per_epoch 30 --robot_name two_head_arm_g  --agents $agents_per_job --previous_agents $previous_agents
