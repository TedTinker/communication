#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name ef_watch_and_this_4 --alpha None --normal_alpha 0.1 --curiosity hidden_state --epochs_per_processor "[(70000, 'wl')]"  --agents $agents_per_job --previous_agents $previous_agents
