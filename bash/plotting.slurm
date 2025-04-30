#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting.py --comp deigo --arg_title ___e+ec+ef+eq_vision_1+eq_vision_2+eq_vision_3+eq_vision_4+eq_vision_5+eq_touch_1+eq_touch_2+eq_touch_3+eq_touch_4+eq_touch_5+eq_t1+ecq_t1+efq_t1+eq_t2+ecq_t2+efq_t2+eq_t3+ecq_t3+efq_t3___ --arg_name plotting
