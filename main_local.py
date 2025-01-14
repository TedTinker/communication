#%%
from memory_profiler import profile

from utils import args
from agent import Agent

args.alpha = None
args.normal_alpha = .1
args.curiosity = "hidden_state"
args.processor_list = ["w"]
args.epochs = [100]
args.agents_per_component_data = 0
args.eta_reduction = .99
args.reward = 10
args.steps_per_epoch = args.max_steps

args.show_duration = True

def run():
    agent = Agent(
        1, 
        GUI = True, 
        args = args)
    agent.training()
    
run()
# %%
