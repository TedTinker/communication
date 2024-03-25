#%%
import os
import pickle

from utils import default_args 
from arena import Arena, get_physics
from task import Task, Task_Runner
from agent import Agent 

args = default_args

agent = Agent(GUI = False, args = args)

agent.tasks = {
    "1" : Task(actions = 2, objects = 2, shapes = 5, colors = 6, parent = True,  args = args),
    "2" : Task(actions = 2, objects = 2, shapes = 5, colors = 6, parent = True,  args = args),
    "3" : Task(actions = 5, objects = 2, shapes = 5, colors = 6, parent = False, args = args)}
physicsClient_1 = get_physics(GUI = True)
arena_1 = Arena(physicsClient_1)
physicsClient_2 = get_physics(GUI = False)
arena_2 = Arena(physicsClient_2)
agent.task_runners = {task_name : Task_Runner(task, arena_1, arena_2) for i, (task_name, task) in enumerate(agent.tasks.items())}
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

example_agent_state_dict = master_dict["ef"]["1_3000"]

print("Loading...")
agent.load_state_dict(example_agent_state_dict)
print("Loaded!")

i = 1
while(True):
    print("Test {}".format(i))
    agent.gen_test(sleep_time = 1)
    i += 1