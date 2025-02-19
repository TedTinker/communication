#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_episodes.py --comp deigo --arg_title ___et_1+et_2+ect_1+ect_2+eft_1+eft_2+et_one_head_arm_1+et_one_head_arm_2+ect_one_head_arm_1+ect_one_head_arm_2+eft_one_head_arm_1+eft_one_head_arm_2+et_two_head_arm_1+et_two_head_arm_2+ect_two_head_arm_1+ect_two_head_arm_2+eft_two_head_arm_1+eft_two_head_arm_2___ --arg_name plotting_episodes
