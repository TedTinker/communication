#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 24:00:00
#SBATCH --mem=30G
module load singularity
singularity exec  maze.sif python communication/finish_dicts.py --comp deigo --arg_title ___e+e_no_distance_no_angle+ef_comm+ef_comm_no_distance_no_angle+ef+ef_no_distance_no_angle___ --arg_name finishing_dictionaries
