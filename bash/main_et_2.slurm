#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name et_2 --alpha None --normal_alpha 0.1 --try_thing_7 False --try_thing_8 True --try_thing_9 True  --agents $agents_per_job --previous_agents $previous_agents
