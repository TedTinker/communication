#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec communication.sif python communication/plotting.py --comp deigo --arg_title ___eft_5+eft_5+eft_5+eft_5+eft_5+eft_5+eft_5___ --arg_name plotting
