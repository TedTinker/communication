#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting.py --comp deigo --arg_title ___e+eq_t1+eq_t2+ec+ecq_t1+ecq_t2+ef+efq_t1+efq_t2+ecq_t3_1+ecq_t3_2+ecq_t3_3+efq_t3_t4_1+efq_t3_t4_2+efq_t3_t4_3+efq_t3_t4_4+efq_t3_t4_5+efq_t3_t4_6+efq_t3_t4_7+efq_t3_t4_8+efq_t3_t4_9+efq_t3_t4_10+efq_t3_t4_11+efq_t3_t4_12___ --arg_name plotting
