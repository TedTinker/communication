#%%
from memory_profiler import profile

from utils import args
from agent import Agent

args.task_list = [1]
args.alpha = None
args.normal_alpha = .1
args.curiosity = "hidden_state"
args.task_list = ["fp5", "w5"]
args.epochs = [100, 100]
#args.show_duration = True

def run():
    agent = Agent(
        1, 
        GUI = False, 
        args = args)
    agent.training()
    
run()
# %%
