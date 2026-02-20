#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec communication.sif python communication/main.py --comp deigo --arg_name e_q4 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --agents_per_agent_save 1 --epochs_per_agent_save 30000 --save_behaviors False --epochs_per_composition_data 10000  --agents $agents_per_job --previous_agents $previous_agents
