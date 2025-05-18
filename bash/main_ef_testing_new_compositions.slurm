#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=2
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name ef_testing_new_compositions --alpha None --normal_alpha 0.05 --target_entropy -1.5 --curiosity hidden_state --hidden_state_eta_vision 0.05 --hidden_state_eta_touch 2 --hidden_state_eta_report_voice 0.75 --save_agents False --save_behaviors False --epochs_per_composition_data 10 --agents_per_composition_data 1 --epochs 30 --keep_data 100  --agents $agents_per_job --previous_agents $previous_agents
