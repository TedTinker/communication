#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/combine_plots.py --comp deigo --arg_title ___ect_one_arm_1+ect_one_arm_2+ect_one_arm_3+ect_one_arm_4+eft_one_arm_1+eft_one_arm_2+eft_one_arm_3+eft_one_arm_4___ --arg_name combining_plots
