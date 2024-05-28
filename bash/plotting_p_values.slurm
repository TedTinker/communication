#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 5:00:00
#SBATCH --mem=25G
module load singularity
singularity exec  maze.sif python communication/plotting_p_val.py --comp deigo --arg_title ___e+ef_1+ef_2+ef_3+ef_4+ef_5+ef_6+ef_7+ef_8+ef_9+ef_10+ef_11+ef_12+ef_13+ef_14+ef_15+ef_16___ --arg_name plotting_p_values
