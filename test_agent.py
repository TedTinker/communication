#%%
import os
import pickle
import gzip
from math import pi
import tkinter as tk
import pybullet as p

from processor import Processor
from agent import Agent 



hyper_parameters = "ef_t3"
agent_num = "0001"
epochs = "005000"
saved_file = "saved_deigo"

print("\n\nLoading default agent...", end = " ")

load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
with gzip.open(load_path, "rb") as f:
    agent = pickle.load(f) 

agent.start_physics(GUI = True)



episodes = 0
wins = 0
print("Ready to go!")


agent.args.language = "task_color_shape"

agent.args.max_steps = 5



#%%



hyper_parameters = "ef_t3"
agent_num = "0001"
epochs = "005000"
saved_file = "saved_deigo"



def change_agent(hyper_parameters, agent_num, epochs, saved_file = "saved_deigo"):
    print("\n\nLoading new agent...", end = " ")
    load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
    with gzip.open(load_path, "rb") as f:
        new_agent = pickle.load(f) 
    agent.load_state_dict(new_agent.state_dict())
    new_agent.args.language = "color_shape_task"
    change_args(new_agent)

    """multiply_steps_per_step = 5
    agent.args.steps_per_step *= multiply_steps_per_step 
    agent.args.time_step *= multiply_steps_per_step
    agent.arena_1.args.steps_per_step = multiply_steps_per_step
    agent.arena_1.args.time_step *= multiply_steps_per_step
    p.setTimeStep(agent.args.time_step, physicsClientId=agent.arena_1.physicsClient)"""  # More accurate time step
    episodes = 0
    wins = 0
    print("Ready to go!")
    
def change_args(new_agent):
    args = new_agent.args
    agent.args = args
    agent.arena_1.args = args
    agent.arena_2.args = args
    for processor_name, processor in new_agent.processors.items():
        processor.args = args 
        processor.arena_1.args = args
        processor.arena_2.args = args
        processor.goal.make_texts(args.language)
    for processor_name, processor in new_agent.all_processors.items():
        processor.args = args 
        processor.arena_1.args = args
        processor.arena_2.args = args
        processor.goal.make_texts(args.language)
        
        
    # STILL NEED TO DO GOALS AND MAYBE ARENA
    
change_agent(hyper_parameters, agent_num, epochs)
     


#%%



    #0,  # Free Play (can we do this?)
    #1,  # Watch
    #2,  # Be Near
    #3,  # Top
    #4,  # Push
    #5,  # Left
    #6   # Right   
        
print("MAKING AGENT LANGUAGE:", agent.args.language)
agent.processors = {0 : Processor(
    agent.args, agent.arena_1, agent.arena_2,
    tasks_and_weights = [(3, 1)], 
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
    sleep_time = .5, 
    waiting = False, 
    user_action = False, 
    dreaming = False)
if(win): 
    wins += 1
print(f"\tWIN RATE: {round(100 * (wins / episodes), 2)}% \t ({wins} wins out of {episodes} episodes)")




# %%