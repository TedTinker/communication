#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=490G
#SBATCH --gres=gpu:4
module load singularity cuda
singularity exec --nv maze.sif python communication/combine_plots.py --comp saion --arg_title ___e+en+ef___ --arg_name combining_plots
