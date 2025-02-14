#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting.py --comp deigo --arg_title ___e+ec+ef+e_one_head_arm+ec_one_head_arm+ef_one_head_arm+e_two_head_arm+ec_two_head_arm+ef_two_head_arm___ --arg_name plotting
