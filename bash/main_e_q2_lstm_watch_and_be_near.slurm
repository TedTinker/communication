#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 96:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec communication.sif python communication/main.py --comp deigo --arg_name e_q2_lstm_watch_and_be_near --alpha None --normal_alpha 0.05 --target_entropy -1.5 --agents_per_agent_save 11 --epochs_per_agent_save 10000 --save_behaviors False --save_compositions False --lstm True --touch_top False --push_forward False --push_left False --push_right False --test_train_num 6  --agents $agents_per_job --previous_agents $previous_agents
