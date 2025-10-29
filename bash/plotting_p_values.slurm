#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec communication.sif python communication/plotting_p_val.py --comp deigo --arg_title ___ef_q2t_12+ef_q2+ec_q2___ --arg_name plotting_p_values
