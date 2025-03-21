#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec maze.sif python communication/plotting.py --comp deigo --arg_title ___et_1+et_2+et_3+et_4+et_5+et_6+ect_1+ect_2+ect_3+ect_4+ect_5+ect_6+eft_1+eft_2+eft_3+eft_4+eft_5+eft_6+ebt_1+ebt_2+ebt_3+ebt_4+ebt_5+ebt_6+ecbt_1+ecbt_2+ecbt_3+ecbt_4+ecbt_5+ecbt_6+efbt_1+efbt_2+efbt_3+efbt_4+efbt_5+efbt_6___ --arg_name plotting
