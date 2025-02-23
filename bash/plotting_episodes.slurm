#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_episodes.py --comp deigo --arg_title ___eft_1+eft_2+eft_3+eft_4+eft_5+eft_6+eft_7+eft_8+eft_one_head_arm_1+eft_one_head_arm_2+eft_one_head_arm_3+eft_one_head_arm_4+eft_one_head_arm_5+eft_one_head_arm_6+eft_one_head_arm_7+eft_one_head_arm_8+eft_two_head_arm_1+eft_two_head_arm_2+eft_two_head_arm_3+eft_two_head_arm_4+eft_two_head_arm_5+eft_two_head_arm_6+eft_two_head_arm_7+eft_two_head_arm_8+eft_two_head_arm_b_1+eft_two_head_arm_b_2+eft_two_head_arm_b_3+eft_two_head_arm_b_4+eft_two_head_arm_b_5+eft_two_head_arm_b_6+eft_two_head_arm_b_7+eft_two_head_arm_b_8___ --arg_name plotting_episodes
