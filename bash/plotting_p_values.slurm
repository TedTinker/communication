#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_p_val.py --comp deigo --arg_title ___efm_1+efm_2___ --arg_name plotting_p_values
