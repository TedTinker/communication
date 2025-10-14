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

# If the user has no specifical goal in mind, leave this as None. 
# Otherwise, another step will change the set_goal.
set_goal = None



# Change these to the agent you would like to test.
hyper_parameters = "eft_4"
agent_num = "0001"
epochs = "060000"
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


# This step allows customising the agent's arguments.

hyper_parameters = "ef_old"
agent_num = "0002"
epochs = "020000"
saved_file = "saved_deigo"



def change_agent(hyper_parameters, agent_num, epochs, saved_file = "saved_deigo"):
    print("\n\nLoading new agent...", end = " ")
    load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
    with gzip.open(load_path, "rb") as f:
        new_agent = pickle.load(f) 
    agent.load_state_dict(new_agent.state_dict())
    
    new_agent.args.pointing_at_object_for_touch_top = 9999 # For example
    change_args(new_agent)

    episodes = 0
    wins = 0
    print("Ready to go!")
    
# Some arguments are only relevent in the arena or processor.
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



# Make a random goal.
set_goal = make_objects_and_task(
                num_objects = agent.processors["all"].objects, 
                allowed_tasks_and_weights = agent.processors["all"].tasks_and_weights, 
                allowed_colors = agent.processors["all"].colors, 
                allowed_shapes = agent.processors["all"].shapes, 
                test_train_num = agent.processors["all"].args.test_train_num, test = None)

# Details about that goal.
print(set_goal)
print(set_goal[0].name)
print(set_goal[1][0][0].name)
print(set_goal[1][0][1].name)
print(set_goal[1][1][0].name)
print(set_goal[1][1][1].name)




#%%



# Creating a processor and running an episode.

    #   Tasks:
    #0,  # Free Play
    #1,  # Watch
    #2,  # Be Near
    #3,  # Top
    #4,  # Push
    #5,  # Left
    #6   # Right   
    
    #   Colors:
    #0,  # Red
    #1,  # Green
    #2,  # Blue
    #3,  # Cyan
    #4,  # Magenta
    #5,  # Yellow
    
    #   Shapes:
    #0,  # Pillar
    #1,  # Pole
    #2,  # Dumbbell
    #3,  # Cone
    #4,  # Hourglass
            
agent.processors = {0 : Processor(
    agent.args, agent.arena_1, agent.arena_2,
    tasks_and_weights = [(5, 1)], 
    objects = 2, 
    colors = [0, 1, 2, 3, 4, 5], 
    shapes = [0, 1, 2, 3, 4], 
    parenting = True)}

agent.processor_name = 0

episodes += 1
win = agent.save_episodes(
    test = True,                               # Should randomly chosen objects be for training, test, or either?
    verbose = False,                            # Should extra information be printed?
    display = True,                             # Should a complete set of observations, prior predictions, and posterior predictions be plotted?
    video_display = False,                      # Should a simple view of the robot's observations be plotted?
    sleep_time = .25,                           # How slow should the episode be?
    waiting = False,                            # Should the episode wait every step?
    user_action = True,                        # Should the user be able to choose the robot's actions?
    dreaming = True,                            # Should the robot be "dreaming" or "hallucinating," only seeing its own predictions?
    set_positions = None, #([4, -2], [4, 2]),                       # Should the objects be in specific positions?
    set_goal = set_goal)                        # Should a specific goal be used?
if(win): 
    wins += 1
#print(f"\tWIN RATE: {round(100 * (wins / episodes), 2)}% \t ({wins} wins out of {episodes} episodes)")




# %%
