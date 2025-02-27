#%%
from memory_profiler import profile
from math import pi

from utils import args, get_num_sensors
from agent import Agent

args.show_duration = True

args.local = True
args.alpha = None
args.normal_alpha = .1
args.curiosity = "hidden_state"
args.processor_list = ["wpulr"]
args.epochs = [10]
args.agents_per_component_data = 0
args.eta_reduction = .99
args.reward = 10
args.steps_per_epoch = args.max_steps
args.smooth_steps = True

args.robot_name = "two_head_arm_c"
args.min_joint_1_angle = -pi/4
args.max_joint_1_angle = pi/4
args.min_joint_2_angle = -pi/2
args.max_joint_2_angle = 0

num_sensors, sensors = get_num_sensors(args.robot_name)
args.sensors_state_size = num_sensors
args.sensors_encode_size = num_sensors 
args.sensors_shape = num_sensors
args.sensor_names = sensors

"""x = 4
args.max_steps = 10 * x
args.max_speed = 10 / x
args.max_shoulder_speed = 8 / x
args.push_amount = .75 / x
args.pull_amount = .25 / x
args.left_right_amount = .25 / x"""


def run():
    agent = Agent(
        args = args,
        i = 1, 
        GUI = True)
    agent.training(sleep_time = 1)
    
run()
# %%
