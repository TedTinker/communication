#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 24:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=30
module load singularity
singularity exec  maze.sif python communication/main.py --comp deigo --arg_name e_no_distance_no_angle --alpha None --normal_alpha 0.1 --dist_reward 0 --angle_reward 0  --agents $agents_per_job --previous_agents $previous_agents
