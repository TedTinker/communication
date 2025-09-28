#%%
import os
import pickle
import gzip
from math import pi

import tkinter as tk
import pybullet as p

from utils import make_objects_and_task
from processor import Processor
from models import Actor
from agent import Agent 

set_goal = None



hyper_parameters = "ef_q2t_1"
agent_num = "0001"
epochs = "000000"
saved_file = "saved_deigo"

print("\n\nLoading default agent...", end = " ")

load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
with gzip.open(load_path, "rb") as f:
    agent = pickle.load(f) 
    
agent.args.object_size = 2.5
agent.start_physics(GUI = True)



episodes = 0
wins = 0
print("Ready to go!")




#%%



hyper_parameters = "ef_q2t_1"
agent_num = "0001"
epochs = "000000"
saved_file = "saved_deigo"



def change_agent(hyper_parameters, agent_num, epochs, saved_file = "saved_deigo"):
    print("\n\nLoading new agent...", end = " ")
    load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
    with gzip.open(load_path, "rb") as f:
        new_agent = pickle.load(f) 
    agent.load_state_dict(new_agent.state_dict())
    
    new_agent.args.min_object_distance = 10
    new_agent.args.max_object_distance = 10
    new_agent.args.be_near_distance = 7
    new_agent.args.watch_distance = 10.75
    #new_agent.args.be_near_distance = 7
    #new_agent.args.watch_distance = 99
    
    
    
    """parser.add_argument('--steps_per_step',                 type=int,           default = 20,
                    help='numSubSteps in pybullet environment.')
parser.add_argument('--numSolverIterations',            type=int,           default = 1,
                    help='numSubSteps in pybullet environment.')
parser.add_argument('--numSubSteps',                    type=int,           default = 1,
                    help='numSubSteps in pybullet environment.')"""
    
    
    
    change_args(new_agent)

    episodes = 0
    wins = 0
    print("Ready to go!")
    
def change_args(new_agent):
    args = new_agent.args
    agent.args = args
    agent.arena_1.args = args
    agent.arena_1.change_physicsClient()
    agent.arena_2.args = args
    agent.arena_2.change_physicsClient()
    for processor_name, processor in new_agent.processors.items():
        processor.args = args 
        processor.arena_1.args = args
        processor.arena_2.args = args
    for processor_name, processor in new_agent.all_processors.items():
        processor.args = args 
        processor.arena_1.args = args
        processor.arena_2.args = args
        
        
    
change_agent(hyper_parameters, agent_num, epochs)
     


#%% 



set_goal = make_objects_and_task(
                num_objects = agent.processors["all"].objects, 
                allowed_tasks_and_weights = agent.processors["all"].tasks_and_weights, 
                allowed_colors = agent.processors["all"].colors, 
                allowed_shapes = agent.processors["all"].shapes, 
                test_train_num = agent.processors["all"].args.test_train_num, test = None)



#%%



    #0,  # Free Play
    #1,  # Watch
    #2,  # Be Near
    #3,  # Top
    #4,  # Push
    #5,  # Left
    #6   # Right   
            
agent.processors = {0 : Processor(
    agent.args, agent.arena_1, agent.arena_2,
    tasks_and_weights = [(5, 1)], 
    objects = 2, 
    colors = [0, 1, 2, 3, 4, 5], 
    shapes = [0], 
    parenting = True)}

agent.processor_name = 0

episodes += 1
win = agent.save_episodes(
    test = None, 
    verbose = False,
    display = False, 
    video_display = True,
    sleep_time = .25, 
    waiting = False, 
    user_action = True, 
    dreaming = False,
    set_positions = [(0, 6), (6, 0)],
    set_goal = set_goal)
if(win): 
    wins += 1
#print(f"\tWIN RATE: {round(100 * (wins / episodes), 2)}% \t ({wins} wins out of {episodes} episodes)")




# %%