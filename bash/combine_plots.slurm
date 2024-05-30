#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 5:00:00
#SBATCH --mem=25G
module load singularity
singularity exec  maze.sif python communication/combine_plots.py --comp deigo --arg_title ___d+e+f+ef+n+en___ --arg_name combining_plots
