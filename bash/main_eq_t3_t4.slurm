#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=0
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name eq_t3_t4 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --save_agents False --save_behaviors False --save_compositions False --vision_encode_size 256 --vision_state_size 256 --voice_encode_size 256 --voice_state_size 256  --agents $agents_per_job --previous_agents $previous_agents
