#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name eqt_6 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --save_agents False --save_behaviors False --save_compositions False --tanh_touch True --pointing_at_object_for_watch 0.5235987755982988 --watch_duration 5  --agents $agents_per_job --previous_agents $previous_agents
