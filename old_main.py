#%%
from memory_profiler import profile

from utils import args
from agent import Agent

args.alpha = None
args.delta = 10
args.show_duration = True

def run():
    agent = Agent(1, GUI = True, args = args)
    agent.training()
    
run()
# %%
