#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 24:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name e_w_weights --alpha None --normal_alpha 0.1 --processor_list "[     'fp',   'w',    'wpulr_9',  'wpulr_8',  'wpulr_7',  'wpulr_6',  'wpulr_5',  'wpulr_4',  'wpulr_3',  'wpulr_2',  'wpulr_1',  'wpulr']" --epochs "[             100,    50,     30,         30,         30,         30,         30,         30,         30,         30,         30,         30]"  --agents $agents_per_job --previous_agents $previous_agents
