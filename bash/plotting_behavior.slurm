#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_behavior.py --comp deigo --arg_title ___e+ec+ef+et_one_head_arm_1+et_one_head_arm_2+et_one_head_arm_3+et_one_head_arm_4+ect_one_head_arm_1+ect_one_head_arm_2+ect_one_head_arm_3+ect_one_head_arm_4+eft_one_head_arm_1+eft_one_head_arm_2+eft_one_head_arm_3+eft_one_head_arm_4+et_two_head_arm_1+et_two_head_arm_2+et_two_head_arm_3+et_two_head_arm_4+ect_two_head_arm_1+ect_two_head_arm_2+ect_two_head_arm_3+ect_two_head_arm_4+eft_two_head_arm_1+eft_two_head_arm_2+eft_two_head_arm_3+eft_two_head_arm_4+et_two_head_arm_b_1+et_two_head_arm_b_2+et_two_head_arm_b_3+et_two_head_arm_b_4+ect_two_head_arm_b_1+ect_two_head_arm_b_2+ect_two_head_arm_b_3+ect_two_head_arm_b_4+eft_two_head_arm_b_1+eft_two_head_arm_b_2+eft_two_head_arm_b_3+eft_two_head_arm_b_4___ --arg_name plotting_behavior
