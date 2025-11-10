#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec communication.sif python communication/finish_dicts.py --comp deigo --arg_title ___e+ep+ec+ef+e_q2+ep_q2+ec_q2+ef_q2+ef_q2_t1+ef_q2_t2+ef_q4t_1+ef_q4t_2+ef_q4t_3+ef_q4t_4+ef_q4t_5+ef_q4t_6+ef_q4t_7+ef_q4t_8+ef_q4t_9+ef_q4t_10+ef_q4t_11+ef_q4t_12+ef_q4t_13+ef_q4t_14+ef_q4t_15+ef_q4t_16___ --arg_name finishing_dictionaries
