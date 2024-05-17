#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=490G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=30
module load singularity cuda
singularity exec --nv maze.sif python communication/main.py --comp saion --arg_name en --alpha None --normal_alpha 0.1 --curiosity prediction_error  --agents $agents_per_job --previous_agents $previous_agents
