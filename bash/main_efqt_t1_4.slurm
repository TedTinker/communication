#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec communication.sif python communication/main.py --comp deigo --arg_name efqt_t1_4 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --curiosity hidden_state --hidden_state_eta_vision 0.05 --hidden_state_eta_touch 1 --hidden_state_eta_prop 0.1 --hidden_state_eta_report_voice 0.2 --save_agents False --save_behaviors False --save_compositions False --object_size 2.5 --min_object_distance 4 --max_object_distance 8 --be_near_distance 6 --pointing_at_object_for_being_near 0.5235987755982988 --pointing_at_object_for_touch_top 1.0471975511965976 --touch_top False --yellow False --cone False --hourglass False --test_train_num 2  --agents $agents_per_job --previous_agents $previous_agents
