#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_episodes.py --comp deigo --arg_title ___e+ep+ec+ef+efq_t1+efq_t2+epq_tt_1+epq_tt_2+epq_tt_3+ecq_tr_1+ecq_tr_2+ecq_tr_3+efq_tv_tt_tr_1+efq_tv_tt_tr_2+efq_tv_tt_tr_3+efq_tv_tt_tr_4+efq_tv_tt_tr_5+efq_tv_tt_tr_6+efq_tv_tt_tr_7+efq_tv_tt_tr_8+efq_tv_tt_tr_9+efq_tv_tt_tr_10+efq_tv_tt_tr_11+efq_tv_tt_tr_12+efq_tv_tt_tr_13+efq_tv_tt_tr_14+efq_tv_tt_tr_15+efq_tv_tt_tr_16+efq_tv_tt_tr_17+efq_tv_tt_tr_18+efq_tv_tt_tr_19+efq_tv_tt_tr_20+efq_tv_tt_tr_21+efq_tv_tt_tr_22+efq_tv_tt_tr_23+efq_tv_tt_tr_24+efq_tv_tt_tr_25+efq_tv_tt_tr_26+efq_tv_tt_tr_27___ --arg_name plotting_episodes
