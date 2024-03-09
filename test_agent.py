#%%
import os
import pickle

from utils import default_args 
from task import Task
from agent import Agent 

args = default_args

tasks = {
    "1" : Task(actions = 1, objects = 2, shapes = 5, colors = 6, parent = True,  args = args),
    "2" : Task(actions = 2, objects = 2, shapes = 5, colors = 6, parent = True,  args = args),
    "3" : Task(actions = 5, objects = 2, shapes = 5, colors = 6, parent = False, args = args)}

agent = Agent(GUI = True, args = args)



def load_nested_dictionaries(parent_directory):
    master_dict = {}
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        if os.path.isdir(item_path):
            pickle_path = os.path.join(item_path, 'plot_dict.pickle')
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as file:
                    plot_dict = pickle.load(file)
                    master_dict[item] = plot_dict["agent_lists"]
    return master_dict

master_dict = load_nested_dictionaries('saved')

example_agent_state_dict = master_dict["ef"]["1_1000"]

agent.load_state_dict(example_agent_state_dict)



while(True):
    agent.gen_test()