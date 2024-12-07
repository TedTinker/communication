#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_components.py --comp deigo --arg_title ___efm_1+efm_2+eft_1+eft_2+eft_3+eft_4+eft_5+eft_6+eft_7+eft_8+eft_9+eft_10+eft_11+eft_12+eft_13+eft_14+eft_15+eft_16+eft_17+eft_18+eft_19+eft_20+eft_21+eft_22+eft_23+eft_24+eft_25+eft_26+eft_27+eft_28+eft_29+eft_30+eft_31+eft_32___ --arg_name plotting_components
