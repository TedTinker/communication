#%%
from memory_profiler import profile
from math import pi

from utils import args
from agent import Agent

args.local = True
args.alpha = None
args.normal_alpha = .1
args.curiosity = "hidden_state"
args.processor_list = ["f", "w"]
args.epochs = [10, 10]
args.agents_per_component_data = 0
args.eta_reduction = .99
args.reward = 10
args.steps_per_epoch = args.max_steps
args.smooth_steps = True

if(args.robot_name != "two_side_arm"):
    args.min_shoulder_angle = -pi/2

"""x = 4
args.max_steps = 10 * x
args.max_speed = 10 / x
args.max_shoulder_speed = 8 / x
args.push_amount = .75 / x
args.pull_amount = .25 / x
args.left_right_amount = .25 / x"""

args.show_duration = True

def run():
    agent = Agent(
        1, 
        GUI = True, 
        args = args)
    agent.training(sleep_time = 1)
    
run()
# %%
