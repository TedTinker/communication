#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 5:00:00
#SBATCH --mem=25G
module load singularity
singularity exec  maze.sif python communication/finish_dicts.py --comp deigo --arg_title ___d+e+n+en+f+ef___ --arg_name finishing_dictionaries
