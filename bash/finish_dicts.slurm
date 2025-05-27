#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/finish_dicts.py --comp deigo --arg_title ___e+ec+ef+eq_t7_1+eq_t7_2+eq_t7_3+eq_t7_4+eq_t7_5+eq_t7_6+eq_t7_7+eq_t7_8+ecq_t7_1+ecq_t7_2+ecq_t7_3+ecq_t7_4+ecq_t7_5+ecq_t7_6+ecq_t7_7+ecq_t7_8+efq_t7_1+efq_t7_2+efq_t7_3+efq_t7_4+efq_t7_5+efq_t7_6+efq_t7_7+efq_t7_8+efq_tf_1+efq_tf_2+efq_tf_3+efq_tf_4___ --arg_name finishing_dictionaries
