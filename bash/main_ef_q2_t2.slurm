#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 7:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec communication.sif python communication/main.py --comp deigo --arg_name ef_q2_t2 --alpha None --normal_alpha 0.05 --target_entropy -1.5 --curiosity hidden_state --hidden_state_eta_vision 0.05 --hidden_state_eta_touch 1 --hidden_state_eta_prop 0.1 --hidden_state_eta_feedback_voice 0.2 --agents_per_agent_save 11 --epochs_per_agent_save 10000 --save_behaviors False --save_compositions False --touch_top False --be_near False --magenta False --yellow False --cone False --hourglass False --test_train_num 1  --agents $agents_per_job --previous_agents $previous_agents
