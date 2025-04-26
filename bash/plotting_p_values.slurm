#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_p_val.py --comp deigo --arg_title ___e+ec+ef+entropy_1+entropy_2+entropy_3+entropy_4+entropy_5+entropy_6+entropy_7+entropy_8+entropy_9+entropy_10+entropy_11+entropy_12+entropy_13+entropy_14+entropy_15+entropy_16+entropy_17+entropy_18+entropy_19+entropy_20+e_touch_1+e_touch_2+e_touch_3+e_touch_4+e_touch_5___ --arg_name plotting_p_values
