#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/combine_plots.py --comp deigo --arg_title ___e+ec+ef+eqt_1+eqt_2+eqt_3+eqt_4+eqt_5+eqt_6+eqt_7+eqt_8+ecqt_1+ecqt_2+ecqt_3+ecqt_4+ecqt_5+ecqt_6+ecqt_7+ecqt_8+efqt_1+efqt_2+efqt_3+efqt_4+efqt_5+efqt_6+efqt_7+efqt_8___ --arg_name combining_plots
