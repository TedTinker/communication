#%%
from memory_profiler import profile
from math import pi

from utils import args, get_num_sensors, update_args
from agent import Agent

args.show_duration = True

args.local = True

def run():
    agent = Agent(
        args = args,
        i = 1, 
        GUI = True)
    agent.training(sleep_time = 0)
    
run()
# %%
