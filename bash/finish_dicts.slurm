#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/finish_dicts.py --comp deigo --arg_title ___e+ec+ef+eqt_1+eqt_2+eqt_3+eqt_4+eqt_5+eqt_6+eqt_7+eqt_8+eqt_9+eqt_10+eqt_11+eqt_12+ecqt_1+ecqt_2+ecqt_3+ecqt_4+ecqt_5+ecqt_6+ecqt_7+ecqt_8+ecqt_9+ecqt_10+ecqt_11+ecqt_12+efqt_1+efqt_2+efqt_3+efqt_4+efqt_5+efqt_6+efqt_7+efqt_8+efqt_9+efqt_10+efqt_11+efqt_12___ --arg_name finishing_dictionaries
