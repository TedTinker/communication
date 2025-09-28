#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec maze.sif python communication/main.py --comp deigo --arg_name ef_q2t_14 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --curiosity hidden_state --hidden_state_eta_vision 0.05 --hidden_state_eta_touch 1 --hidden_state_eta_prop 0.1 --hidden_state_eta_report_voice 0.2 --epochs_per_agent_save 30000 --save_behaviors False --save_compositions False --object_size 2.5 --robot_name robot_2 --min_object_distance 5 --max_object_distance 8 --be_near_distance 4 --pointing_at_object_for_being_near 1.0471975511965976 --pointing_at_object_for_touch_top 0.5235987755982988  --agents $agents_per_job --previous_agents $previous_agents
