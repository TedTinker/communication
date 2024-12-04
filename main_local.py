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
args.joint_dialogue = True 
args.ignore_silence = True
args.eta_reduction = .99
args.reward = 10
args.max_steps = 50
args.steps_per_epoch = args.max_steps
args.time_step = .2/5
args.steps_per_step = int(20/5)
args.reward_inflation_type = "exp_10"

args.try_thing_1 = True
args.try_thing_2 = True
args.try_thing_3 = True
args.try_thing_4 = True
args.try_thing_5 = True
args.try_thing_6 = True
args.try_thing_7 = True
args.try_thing_8 = True
args.try_thing_9 = True

args.show_duration = True

def run():
    agent = Agent(
        1, 
        GUI = False, 
        args = args)
    agent.training()
    
run()
# %%
