#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_p_val.py --comp deigo --arg_title ___e+ep+ef+e_q2t_1+e_q2t_2+e_q2t_3+e_q2t_4+e_q2t_5+e_q2t_6+e_q2t_7+e_q2t_8+e_q2t_9+e_q2t_10+e_q2t_11+e_q2t_12+e_q2t_13+e_q2t_14+e_q2t_15+e_q2t_16+ep_q2t_1+ep_q2t_2+ep_q2t_3+ep_q2t_4+ep_q2t_5+ep_q2t_6+ep_q2t_7+ep_q2t_8+ep_q2t_9+ep_q2t_10+ep_q2t_11+ep_q2t_12+ep_q2t_13+ep_q2t_14+ep_q2t_15+ep_q2t_16+ef_q2t_1+ef_q2t_2+ef_q2t_3+ef_q2t_4+ef_q2t_5+ef_q2t_6+ef_q2t_7+ef_q2t_8+ef_q2t_9+ef_q2t_10+ef_q2t_11+ef_q2t_12+ef_q2t_13+ef_q2t_14+ef_q2t_15+ef_q2t_16___ --arg_name plotting_p_values
