#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_composition.py --comp deigo --arg_title ___e+eq_t1+eq_t2+ec+ecq_t1+ecq_t2+ef+efq_t1+efq_t2+efq_t3_t5_1+efq_t3_t5_2+efq_t3_t5_3+efq_t3_t5_4+efq_t3_t5_5+efq_t3_t5_6+efq_t3_t5_7+efq_t3_t5_8+efq_t3_t5_9+efq_t4_t5_1+efq_t4_t5_2+efq_t4_t5_3+efq_t4_t5_4+efq_t4_t5_5+efq_t4_t5_6+efq_t4_t5_7+efq_t4_t5_8+efq_t4_t5_9___ --arg_name plotting_composition
