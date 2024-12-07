#%%
from memory_profiler import profile

from utils import args
from agent import Agent

args.alpha = None
args.normal_alpha = .1
args.curiosity = "hidden_state"
args.processor_list = ["f", "w"]
args.epochs = [10, 10]
args.agents_per_component_data = 0
args.eta_reduction = .99
args.reward = 10
args.max_steps = 50
args.steps_per_epoch = args.max_steps
args.time_step = .2/5
args.steps_per_step = int(20/5)
args.reward_inflation_type = "exp_10"
args.steps_ahead = 5

args.try_batchnorm_1 = True
args.try_batchnorm_2 = True
args.try_batchnorm_3 = True
args.try_batchnorm_4 = True
args.try_batchnorm_5 = True
args.try_batchnorm_6 = True
args.try_batchnorm_7 = True
args.try_multi_step = True

args.show_duration = True

def run():
    agent = Agent(
        1, 
        GUI = False, 
        args = args)
    agent.training()
    
run()
# %%
