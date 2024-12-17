#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name e3t_1 --alpha None --normal_alpha 0.1 --max_steps 30 --time_step 0.06666666666666667 --steps_per_step 6.666666666666667 --max_speed 3.3333333333333335 --max_shoulder_speed 2.6666666666666665 --step_cost 1 --steps_ahead 1  --agents $agents_per_job --previous_agents $previous_agents
