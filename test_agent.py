#%%
import os
import pickle
import gzip
from math import pi
import tkinter as tk

from processor import Processor
from agent import Agent 



hyper_parameters = "ect_slow_5"
agent_num = "0002"
epochs = "070000"
saved_file = "saved_deigo"

print("\n\nLoading...", end = " ")

load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
with gzip.open(load_path, "rb") as f:
    agent = pickle.load(f) 
agent.start_physics(GUI = True)
        
episodes = 0
wins = 0
print("Ready to go!")

#%%

    #1,  # Watch
    #2,  # Top
    #3,  # Push
    #4,  # Pull
    #5,  # Left
    #6   # Right   
    
agent.processors = {0 : Processor(
    agent.args, agent.arena_1, agent.arena_2,
    tasks_and_weights = [(2, 1)], 
    objects = 2, 
    colors = [0, 1, 2, 3, 4, 5], 
    shapes = [0, 1, 2, 3, 4], 
    parenting = True)}

agent.processor_name = 0

episodes += 1
win = agent.save_episodes(
    test = False, 
    verbose = True,
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