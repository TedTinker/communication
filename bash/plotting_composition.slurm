#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_composition.py --comp deigo --arg_title ___ec+ef+eq_t1_1+eq_t1_2+eq_t1_3+eq_t1_4+eq_t1_5+eq_t1_6+ecq_t1_1+ecq_t1_2+ecq_t1_3+ecq_t1_4+ecq_t1_5+ecq_t1_6+efq_t1_t2_1+efq_t1_t2_2+efq_t1_t2_3+efq_t1_t2_4+efq_t1_t2_5+efq_t1_t2_6+efq_t1_t2_7+efq_t1_t2_8+efq_t1_t2_9+efq_t1_t2_10+efq_t1_t2_11+efq_t1_t2_12+efq_t1_t2_13+efq_t1_t2_14+efq_t1_t2_15+efq_t1_t2_16+efq_t1_t2_17+efq_t1_t2_18___ --arg_name plotting_composition
