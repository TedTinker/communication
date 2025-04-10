#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name et_vision_1 --alpha None --normal_alpha 0.1 --task_duration 3 --curiosity hidden_state --hidden_state_eta_vision 0.1 --hidden_state_eta_touch 0 --hidden_state_eta_report_voice 0  --agents $agents_per_job --previous_agents $previous_agents
