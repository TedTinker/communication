#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 24:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=30
module load singularity
singularity exec  maze.sif python communication/main.py --comp deigo --arg_name ef_6 --alpha None --normal_alpha 0.1 --curiosity hidden_state --actor_lr 0.001 --critic_lr 0.003  --agents $agents_per_job --previous_agents $previous_agents
