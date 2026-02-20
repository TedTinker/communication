#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec communication.sif python communication/main.py --comp deigo --arg_name eq_seeds_1 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --save_agents False --save_behaviors False --save_compositions False --init_seed 555.0  --agents $agents_per_job --previous_agents $previous_agents
