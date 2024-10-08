#%%
from memory_profiler import profile

from utils import args
from agent import Agent

args.alpha = None
args.normal_alpha = .1
args.curiosity = "hidden_state"
args.processor_list = ["fp", "w"]
args.epochs = [100, 100]
args.agents_per_component_data = 0
#args.show_duration = True

def run():
    agent = Agent(
        1, 
        GUI = False, 
        args = args)
    agent.training()
    
run()
# %%
