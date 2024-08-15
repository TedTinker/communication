#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 24:00:00
#SBATCH --mem=50G
module load singularity
singularity exec  maze.sif python communication/plotting_episodes.py --comp deigo --arg_title ___e+ef_comm+ef+e5+ef_comm5+ef5+e_hard+ef_comm_hard+ef_hard+e_hard5+ef_comm_hard5+ef_hard5___ --arg_name plotting_episodes
