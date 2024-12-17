#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name e2t_2 --alpha None --normal_alpha 0.1 --max_steps 20 --time_step 0.1 --steps_per_step 10 --max_speed 5.0 --max_shoulder_speed 4.0 --step_cost 1 --steps_ahead 2  --agents $agents_per_job --previous_agents $previous_agents
