#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_behavior.py --comp deigo --arg_title ___e+ec+ef+eft_one_head_arm_1+eft_one_head_arm_2+eft_one_head_arm_3+eft_one_head_arm_4+eft_one_head_arm_5+eft_one_head_arm_6+eft_one_head_arm_7+eft_one_head_arm_8+eft_one_head_arm_9+eft_one_head_arm_10+eft_one_head_arm_11+eft_one_head_arm_12+eft_two_head_arm_1+eft_two_head_arm_2+eft_two_head_arm_3+eft_two_head_arm_4+eft_two_head_arm_5+eft_two_head_arm_6+eft_two_head_arm_7+eft_two_head_arm_8+eft_two_head_arm_9+eft_two_head_arm_10+eft_two_head_arm_11+eft_two_head_arm_12+eft_two_head_arm_d_1+eft_two_head_arm_d_2+eft_two_head_arm_d_3+eft_two_head_arm_d_4+eft_two_head_arm_d_5+eft_two_head_arm_d_6+eft_two_head_arm_d_7+eft_two_head_arm_d_8+eft_two_head_arm_d_9+eft_two_head_arm_d_10+eft_two_head_arm_d_11+eft_two_head_arm_d_12___ --arg_name plotting_behavior
