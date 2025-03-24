#%%
from memory_profiler import profile
from math import pi

from utils import args, get_num_sensors, update_args
from agent import Agent

args.show_duration = True

args.local = True
args.processor_list = ["wtplr"]
args.epochs = [10]
args.time_step = .1
args.wide_view = True


"""x = 4
args.max_steps = 10 * x
args.max_speed = 10 / x
args.max_shoulder_speed = 8 / x
args.push_amount = .75 / x
args.left_right_amount = .25 / x"""


def run():
    agent = Agent(
        args = args,
        i = 1, 
        GUI = True)
    agent.training(sleep_time = 1)
    
run()
# %%
