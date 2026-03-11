#%%
from memory_profiler import profile
from math import pi

from utils import args, get_num_sensors, update_args
from agent import Agent
from agent_lstm import Agent as Agent_lstm

# Set local run flags for debugging or argument testing.
args.local = True
args.show_duration = True
args.save_compositions = False
args.exceptions = 1



def run():
    """
    This is helpful for testing new arguments.
    """
    agent = Agent(
        args=args,
        i=1,
        GUI=True)
    agent.training(sleep_time=0)


run()
# %%
