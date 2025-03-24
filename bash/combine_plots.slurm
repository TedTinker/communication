#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/combine_plots.py --comp deigo --arg_title ___et_1+et_2+et_3+et_4+et_5+et_6+et_7+et_8+et_9+et_10+et_11+et_12+ect_1+ect_2+ect_3+ect_4+ect_5+ect_6+ect_7+ect_8+ect_9+ect_10+ect_11+ect_12+eft_1+eft_2+eft_3+eft_4+eft_5+eft_6+eft_7+eft_8+eft_9+eft_10+eft_11+eft_12___ --arg_name combining_plots
