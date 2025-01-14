#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_components.py --comp deigo --arg_title ___e_watch_and_this_1+e_watch_and_this_2+e_watch_and_this_3+e_watch_and_this_4+ec_watch_and_this_1+ec_watch_and_this_2+ec_watch_and_this_3+ec_watch_and_this_4+ef_watch_and_this_1+ef_watch_and_this_2+ef_watch_and_this_3+ef_watch_and_this_4___ --arg_name plotting_components
