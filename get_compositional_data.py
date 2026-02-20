#%% 
import os
import pickle
import gzip
from math import pi
import numpy as np

import tkinter as tk
import pybullet as p

from utils import make_objects_and_task
from processor import Processor
from models import Actor
from agent import Agent

# If the user has no specific goal in mind, leave this as None.
# Otherwise, another step will change the set_goal.
set_goal = None

# Change these to the agent you would like to test.
hyper_parameters = 'ef'
agent_num = '0001'
epochs = '060000'
saved_file = 'saved_deigo'

load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
with gzip.open(load_path, 'rb') as f:
    agent = pickle.load(f)

agent.start_physics(GUI=False)




# %%


from utils import (
    task_map, color_map, shape_map,
    testing_combos_1, testing_combos_2, testing_combos_3
)
from itertools import product

saved_plot_dict = {
    'args': agent.args,
    'arg_title': agent.args.arg_title,
    'arg_name': agent.args.arg_name,
    'all_processor_names': agent.all_processor_names,
    'composition_data': {}
}

agent.args.agents_per_composition_data = -1
agent.epochs = epochs

agent.all_processors = {
    f'{task_map[task].name}_{color_map[color].name}_{shape_map[shape].name}':
    Processor(
        agent.args, agent.arena_1, agent.arena_2,
        tasks_and_weights=[(task, 1)],
        objects=2, colors=[color], shapes=[shape], parenting=True
    )
    for task, color, shape in product(
        agent.args.allowed_tasks,
        agent.args.allowed_colors,
        agent.args.allowed_shapes
    )
}

agent.all_processor_names = list(agent.all_processors.keys())

agent.plot_dict = {
    'args': agent.args,
    'arg_title': agent.args.arg_title,
    'arg_name': agent.args.arg_name,
    'all_processor_names': agent.all_processor_names,
    'composition_data': {}
}

agent.get_composition_data()
saved_plot_dict['composition_data'][agent.epochs] = agent.plot_dict['composition_data'][agent.epochs]

print(saved_plot_dict['composition_data'].keys())
print(saved_plot_dict['composition_data'][epoch].keys())

    
    

# %%

from utils import folder

file_end = str(agent.agent_num).zfill(3)
        
# Save.
with open(f"{folder}/plot_dict_{file_end}_composition.pickle", "wb") as handle:
    pickle.dump(saved_plot_dict, handle)


# %%
