#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_behavior.py --comp deigo --arg_title ___e_q3+e_q3+e_q3+ep_q3+ep_q3+ep_q3+ef+ef+ef+efq_t1+efq_t1+efq_t1+efq_t2+efq_t2+efq_t2+e_q2t_1+e_q2t_2+e_q2t_3+e_q2t_4+e_q2t_1+e_q2t_2+e_q2t_3+e_q2t_4+e_q2t_1+e_q2t_2+e_q2t_3+e_q2t_4+ep_q2t_1+ep_q2t_2+ep_q2t_3+ep_q2t_4+ep_q2t_1+ep_q2t_2+ep_q2t_3+ep_q2t_4+ep_q2t_1+ep_q2t_2+ep_q2t_3+ep_q2t_4+ef_q2t_1+ef_q2t_2+ef_q2t_3+ef_q2t_4+ef_q2t_1+ef_q2t_2+ef_q2t_3+ef_q2t_4+ef_q2t_1+ef_q2t_2+ef_q2t_3+ef_q2t_4___ --arg_name plotting_behavior
