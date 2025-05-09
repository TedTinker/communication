#%%
import os
import pickle
import gzip
from math import pi
import tkinter as tk

from processor import Processor
from agent import Agent 



hyper_parameters = "e"
agent_num = "0001"
epochs = "000000"
saved_file = "saved_deigo"

print("\n\nLoading default agent...", end = " ")

load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
with gzip.open(load_path, "rb") as f:
    agent = pickle.load(f) 

agent.start_physics(GUI = True)



episodes = 0
wins = 0
print("Ready to go!")



#%%



hyper_parameters = "e"
agent_num = "0001"
epochs = "020000"
saved_file = "saved_deigo"

"""hyper_parameters = "ef"
agent_num = "0001"
epochs = "020000"
saved_file = "saved_deigo"""



def change_agent(hyper_parameters, agent_num, epochs, saved_file = "saved_deigo"):
    print("\n\nLoading new agent...", end = " ")
    load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
    with gzip.open(load_path, "rb") as f:
        new_agent = pickle.load(f) 
    agent.load_state_dict(new_agent.state_dict())
    episodes = 0
    wins = 0
    print("Ready to go!")
    
change_agent(hyper_parameters, agent_num, epochs)
     


#%%

    #0,  # Free Play (can we do this?)
    #1,  # Watch
    #2,  # Be Near
    #3,  # Top
    #4,  # Push
    #5,  # Left
    #6   # Right   
    
agent.processors = {0 : Processor(
    agent.args, agent.arena_1, agent.arena_2,
    tasks_and_weights = [(6, 1)], 
    objects = 2, 
    colors = [0, 1, 2, 3, 4, 5], 
    shapes = [0, 1, 2, 3, 4], 
    parenting = True)}

agent.processor_name = 0

episodes += 1
win = agent.save_episodes(
    test = False, 
    verbose = False,
    display = False, 
    video_display = True,
    sleep_time = 1, 
    waiting = False, 
    user_action = True, 
    dreaming = False)
if(win): 
    wins += 1
print(f"\tWIN RATE: {round(100 * (wins / episodes), 2)}% \t ({wins} wins out of {episodes} episodes)")




# %%