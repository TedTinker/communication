#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec communication.sif python communication/plotting_p_val.py --comp deigo --arg_title ___e_seeds_3+eq_seeds_3+eq_seeds_4+ep_seeds_2+ep_seeds_3+ep_seeds_4___ --arg_name plotting_p_values
