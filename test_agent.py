#%%
import os
import pickle
import gzip

from processor import Processor
from agent import Agent 



hyper_parameters = "ef"
agent_num = "0001"
epochs = "070000"
saved_file = "saved_deigo"

print("\n\nLoading...", end = " ")

load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
with gzip.open(load_path, "rb") as f:
    agent = pickle.load(f)  # Load the compressed agent file
agent.start_physics(GUI = True)

episodes = 0
wins = 0
print("Ready to go!")

agent.args.global_push_amount = agent.args.push_amount
agent.args.global_pull_amount = agent.args.pull_amount
agent.args.global_left_right_amount = agent.args.left_right_amount

agent.args.local_push_pull_limit = agent.args.push_amount
agent.args.local_left_right_amount = agent.args.left_right_amount



#%%

    #1,  # Watch
    #2,  # Push
    #3,  # Pull
    #4,  # Left
    #5   # Right   
    
agent.processors = {0 : Processor(
    agent.args, agent.arena_1, agent.arena_2,
    tasks_and_weights = [(2, 1)], 
    objects = 1, 
    colors = [0], 
    shapes = [0], 
    parenting = True)}

agent.processor_name = 0



episodes += 1
win = agent.save_episodes(
    test = None, 
    verbose = True,
    sleep_time = 1, 
    waiting = False, 
    user_action = True, 
    dreaming = False)
if(win): 
    wins += 1
print(f"\tWIN RATE: {round(100 * (wins / episodes), 2)}% \t ({wins} wins out of {episodes} episodes)")




# %%