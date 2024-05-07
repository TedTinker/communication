#%%
import os
import pickle

from utils import default_args 
from arena import Arena, get_physics
from task import Task, Task_Runner
from agent import Agent 

args = default_args

agent = Agent(GUI = True, args = args)

agent.tasks = {
            "0" : Task(actions = [-1], objects = 4, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parent = True, args = agent.args),
            "1" : Task(actions = [1], objects = 1, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parent = True, args = agent.args),
            "2" : Task(actions = [1], objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0], parent = True, args = agent.args),
            "3" : Task(actions = [0, 1, 2, 3, 4], objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parent = True, args = agent.args)}
agent.task_runners = {task_name : Task_Runner(task, agent.arena_1, agent.arena_2) for i, (task_name, task) in enumerate(agent.tasks.items())}
agent.task_name = "1"
agent.task = agent.task_runners[agent.task_name]

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

example_agent_state_dict = master_dict["ef"]["1_10000"]

print("\n\nLoading...", end = " ")
agent.load_state_dict(example_agent_state_dict)
print("Loaded!")

i = 1
while(True):
    print("Test {}".format(i))
    agent.gen_test(sleep_time = .1)
    i += 1
# %%
