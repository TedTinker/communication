#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec communication.sif python communication/main.py --comp deigo --arg_name efk --alpha None --normal_alpha 0.05 --target_entropy -1.5 --curiosity hidden_state --hidden_state_eta_vision 0.05 --hidden_state_eta_touch 1 --hidden_state_eta_prop 0.1 --hidden_state_eta_report_voice 0.2 --exceptions 1 --save_behaviors False --epochs_per_agent_save 2500 --agents_per_agent_save 99 --epochs_per_composition_data 10000  --agents $agents_per_job --previous_agents $previous_agents
