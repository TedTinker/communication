#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_components.py --comp deigo --arg_title ___et_1+et_2+et_vision_1+et_vision_2+et_vision_3+et_vision_4+et_vision_5+et_vision_6+et_touch_1+et_touch_2+et_touch_3+et_touch_4+et_touch_5+et_touch_6+et_report_voice_1+et_report_voice_2+et_report_voice_3+et_report_voice_4+et_report_voice_5+et_report_voice_6___ --arg_name plotting_components
