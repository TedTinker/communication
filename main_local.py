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
args.steps_per_epoch = args.max_steps
args.time_step = .2/5
args.steps_per_step = int(20/5)
args.reward_inflation_type = "exp_10"
args.steps_ahead = 5

x = 4
args.max_steps = 10 * x
args.max_speed = 10 / x
args.max_shoulder_speed = 8 / x
args.push_amount = .75 / x
args.pull_amount = .25 / x
args.left_right_amount = .25 / x

args.show_duration = True

def run():
    agent = Agent(
        1, 
        GUI = True, 
        args = args)
    agent.training()
    
run()
# %%
