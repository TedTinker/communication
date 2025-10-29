#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec communication.sif python communication/main.py --comp deigo --arg_name efqt_7 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --curiosity hidden_state --hidden_state_eta_vision 0.05 --hidden_state_eta_touch 1 --hidden_state_eta_prop 0.1 --hidden_state_eta_report_voice 0.2 --save_agents False --save_behaviors False --save_compositions False --exceptions 1 --max_wheel_speed_for_exception 2 --max_arm_speed_for_exception 0.1  --agents $agents_per_job --previous_agents $previous_agents
