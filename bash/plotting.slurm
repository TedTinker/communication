#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting.py --comp deigo --arg_title ___e+ec+ef+eqt_t1_1+eqt_t1_2+eqt_t2_1+eqt_t2_2+eqt_t3_1+eqt_t3_2+ecqt_t1_1+ecqt_t1_2+ecqt_t2_1+ecqt_t2_2+ecqt_t3_1+ecqt_t3_2+efqt_t1_1+efqt_t1_2+efqt_t2_1+efqt_t2_2+efqt_t3_1+efqt_t3_2___ --arg_name plotting
