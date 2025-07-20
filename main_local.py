#%%
from memory_profiler import profile
from math import pi

from utils import args, get_num_sensors, update_args
from agent import Agent

args.local = True
args.show_duration = True
args.save_compositions = False

args.touch_top = False,
args.yellow = False,
args.hourglass = False,
args.test_train_num = 2

def run():
    agent = Agent(
        args = args,
        i = 1, 
        GUI = False)
    agent.training(sleep_time = 0)
    
run()
# %%
