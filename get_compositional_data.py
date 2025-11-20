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
hyper_parameters = "efk"
agent_num = "0007"
epochs = "060000"
saved_file = "saved_deigo"

print("\n\nLoading default agent...", end = " ")

load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
with gzip.open(load_path, "rb") as f:
    agent = pickle.load(f) 
    
agent.start_physics(GUI = False)




#%% 



from utils import task_map, color_map, shape_map, testing_combos_1, testing_combos_2, testing_combos_3
    
from itertools import product



saved_plot_dict = {
    "args" : agent.args,
    "arg_title" : agent.args.arg_title,
    "arg_name" : agent.args.arg_name,
    "all_processor_names" : agent.all_processor_names,
    "composition_data" : {}}

for epoch in [i for i in range(0, 60001, 2500)]:
    str_epoch = str(epoch)
    while(len(str_epoch) < 6):
        str_epoch = "0" + str_epoch
    print("\nEpoch:", str_epoch)
    load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{str_epoch}.pkl.gz'
    with gzip.open(load_path, "rb") as f:
        agent = pickle.load(f) 
        
    agent.args.agents_per_composition_data = -1
    agent.epochs = epoch
    agent.all_processors = {f"{task_map[task].name}_{color_map[color].name}_{shape_map[shape].name}" : 
        Processor(agent.args, agent.arena_1, agent.arena_2, tasks_and_weights = [(task, 1)], objects = 2, colors = [color], shapes = [shape], parenting = True) for task, color, shape in \
            product([t for t in agent.args.allowed_tasks], [c for c in agent.args.allowed_colors], [s for s in agent.args.allowed_shapes])}
    all_processor_names = list(agent.all_processors.keys())
    agent.all_processor_names = all_processor_names
    
    agent.plot_dict = {
        "args" : agent.args,
        "arg_title" : agent.args.arg_title,
        "arg_name" : agent.args.arg_name,
        "all_processor_names" : agent.all_processor_names,
        "composition_data" : {}}
    
    agent.get_composition_data()
    saved_plot_dict["composition_data"][agent.epochs] = agent.plot_dict["composition_data"][agent.epochs]
    print(saved_plot_dict["composition_data"].keys())
    print(saved_plot_dict["composition_data"][epoch].keys())
    
    

# %%

from utils import folder

            
file_end = str(agent.agent_num).zfill(3)
        
# Save.
with open(f"{folder}/plot_dict_{file_end}.pickle", "wb") as handle:
    pickle.dump(saved_plot_dict, handle)


# %%
