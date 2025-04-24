#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/combine_plots.py --comp deigo --arg_title ___e+ec+ef+e_vision_1+e_vision_2+e_vision_3+e_vision_4+e_vision_5+e_touch_1+e_touch_2+e_touch_3+e_touch_4+e_touch_5+e_report_voice_1+e_report_voice_2+e_report_voice_3+e_report_voice_4+e_report_voice_5___ --arg_name combining_plots
