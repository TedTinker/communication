#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 5:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=30
module load singularity
singularity exec  maze.sif python communication/main.py --comp deigo --arg_name ef_4 --alpha None --normal_alpha 0.1 --curiosity hidden_state --beta_rgbd 1 --beta_comm 1 --beta_sensors 0 --hidden_state_eta_sensors 0  --agents $agents_per_job --previous_agents $previous_agents
