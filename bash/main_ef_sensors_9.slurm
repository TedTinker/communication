#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 24:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=30
module load singularity
singularity exec  maze.sif python communication/main.py --comp deigo --arg_name ef_sensors_9 --alpha None --normal_alpha 0.1 --curiosity hidden_state --beta_sensors 1 --hidden_state_eta_rgbd 0 --hidden_state_eta_comm 0 --hidden_state_eta_sensors 0.3  --agents $agents_per_job --previous_agents $previous_agents
