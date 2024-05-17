#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=490G
#SBATCH --gres=gpu:4
module load singularity cuda
singularity exec --nv maze.sif python communication/finish_dicts.py --comp saion --arg_title ___e+en+ef___ --arg_name finishing_dictionaries
