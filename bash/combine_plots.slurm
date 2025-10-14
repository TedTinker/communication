#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec communication.sif python communication/combine_plots.py --comp deigo --arg_title ___ef_q2t_3+e_q2t_11+ef_q2t_2___ --arg_name combining_plots
