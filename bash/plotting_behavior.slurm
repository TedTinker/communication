#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_behavior.py --comp deigo --arg_title ___e+ec+ef+efq_t1+efq_t2+eq_t3+eq_t4+eq_t3_t4+ecq_t3+ecq_t4+ecq_t3_t4+efq_t3+efq_t4+efq_t3_t4___ --arg_name plotting_behavior
