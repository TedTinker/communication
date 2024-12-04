#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting_behavior.py --comp deigo --arg_title ___eft_1+eft_2+eft_3+eft_4+eft_5+eft_6+eft_7+eft_8+eft_9+eft_10+eft_11+eft_12+eft_13+eft_14+eft_15+eft_16+eft_17+eft_18+eft_19+eft_20+eft_21+eft_22+eft_23+eft_24+eft_25+eft_26+eft_27+eft_28+eft_29+eft_30+eft_31+eft_32+eft_33+eft_34+eft_35+eft_36+eft_37+eft_38+eft_39+eft_40+eft_41+eft_42+eft_43+eft_44+eft_45+eft_46+eft_47+eft_48+eft_49+eft_50+eft_51+eft_52+eft_53+eft_54+eft_55+eft_56+eft_57+eft_58+eft_59+eft_60+eft_61+eft_62+eft_63+eft_64___ --arg_name plotting_behavior
