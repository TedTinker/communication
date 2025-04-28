#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=30
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name ef_t2 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --curiosity hidden_state --hidden_state_eta_vision 0.3 --hidden_state_eta_touch 2 --hidden_state_eta_report_voice 0.75 --save_agents False --save_behaviors False --save_compositions False --watch_duration 4 --be_near_duration 4 --top_duration 4 --push_duration 4  --agents $agents_per_job --previous_agents $previous_agents
