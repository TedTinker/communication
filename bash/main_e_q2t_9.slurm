#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name e_q2t_9 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --epochs_per_agent_save 60000 --save_behaviors False --save_compositions False --object_size 2.5 --robot_name robot --watch_distance 99 --min_object_distance 5 --max_object_distance 8 --be_near_distance 7  --agents $agents_per_job --previous_agents $previous_agents
