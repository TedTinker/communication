#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name efqt_3 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --curiosity hidden_state --hidden_state_eta_vision 0.05 --hidden_state_eta_touch 2 --hidden_state_eta_report_voice 0.75 --save_agents False --save_behaviors False --save_compositions False --tanh_touch False --pointing_at_object_for_watch 0.39269908169872414 --watch_duration 3  --agents $agents_per_job --previous_agents $previous_agents
