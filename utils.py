#%% 

# To do: 
#   Make it work.
#   Make it work FASTER.
#   Make comm prediction testing more successful.
#   Try making sensor-observation deeper, with speed or something. 
#   Maybe using separate PVRNNs for rgbd, comm, sensors?
#   Plot episodes with names of just activated sensors.
#   Try predicting multiple steps in the future.
#   Can we use smaller data-type?

import os
import pickle
import pybullet as p
from math import pi
from time import sleep
import builtins
import datetime 
import matplotlib
import argparse, ast
from math import exp, sqrt, log
from random import choice, randint
import torch
from torch import nn 
import platform
from torch.distributions import Normal
import torch.nn.functional as F
from torchvision.transforms import Resize
from kornia.color import rgb_to_hsv 
import psutil
from itertools import product

if(os.getcwd().split("/")[-1] != "communication"): os.chdir("communication")

torch.set_printoptions(precision=3, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
device = "cpu"

action_map = {
    -1: ["Z", "NONE"],
    0: ["A", "WATCH"],
    1: ["B", "PUSH"],     
    2: ["C", "PULL"],   
    3: ["D", "LEFT"],   
    4: ["E", "RIGHT"]}    
max_len_action_name = max([len(a[1]) for a in action_map.values()])
action_name_list = [action[1] for action in action_map.values()]

color_map = {
    0: ["F", "RED",     (1,0,0,1)], 
    1: ["G", "GREEN",   (0,1,0,1)],
    2: ["H", "BLUE",    (0,0,1,1)],
    3: ["I", "CYAN",    (0,1,1,1)], 
    4: ["J", "PINK",    (1,0,1,1)], 
    5: ["K", "YELLOW",  (1,1,0,1)]} 
max_len_color_name = max([len(c[2]) for c in color_map.values()])
color_name_list = [c[2] for c in color_map.values()]

data_path = "pybullet_data"
robot_file = data_path + "/robot"
shape_files = [f.name for f in os.scandir(data_path + "/shapes") if f.name.endswith("urdf")] ; shape_files.sort()
shape_num_letter_name_file = [[f.split("_")[0], f.split("_")[1], f.split("_")[2][:-5], f] for f in shape_files]
shape_map = {int(num) : [l, n, f] for num, l, n, f in shape_num_letter_name_file} 
max_len_shape_name = max([len(s[1]) for s in shape_map.values()])
shape_name_list = [s[1] for s in shape_map.values()]

def agent_to_english(agent_string):
    if(agent_string == "NONE"):
        return("NONE")
    english_string = ""
    for char in agent_string:
        translated = False
        for d in [action_map, color_map, shape_map]:
            for val in d.values():
                c = val[0]
                n = val[1]
                if(char == c and char != " "):
                    english_string += n + " "
                    translated = True
        if(not translated):
            if(char == " "):
                english_string += "___ "
            else:
                english_string += f"_{char}_ "
    english_string = english_string.strip()
    return(english_string)



comm_map = {
    0: ' ', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G',
    8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N',
    15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U',
    22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}

char_to_index = {v: k for k, v in comm_map.items()}

all_combos = list(product(action_map.keys(), color_map.keys(), shape_map.keys()))
training_combos = [(a, c, s) for (a, c, s) in all_combos if 
                   a == -1 or
                   (s in [0,1] and c in [0,1]) or
                   (s == 0 or c == 0) or  
                   ((s + c) % 2 == 0 and a % 2 == 0) or 
                   ((s + c) % 2 == 1 and a % 2 == 1)]

testing_combos = [combo for combo in all_combos if not combo in training_combos]

def valid_color_shape(action_num, other_shape_colors, allowed_colors, allowed_shapes, test = False):
    if(test):
        these_combos = testing_combos
    else:
        these_combos = training_combos
    these_combos = [combo for combo in these_combos if combo[0] == action_num]
    these_combos = [(combo[1], combo[2]) for combo in these_combos if combo[1] in allowed_colors and combo[2] in allowed_shapes]
    these_combos = [combo for combo in these_combos if not combo in other_shape_colors]
    color_num, shape_num = choice(these_combos)
    return(color_num, shape_num)

def make_objects_and_action(num_objects, allowed_actions, allowed_colors, allowed_shapes, test = False):
    action_num = choice(allowed_actions)
    goal_object = valid_color_shape(action_num, [], allowed_colors, allowed_shapes, test = test)
    colors_shapes_1 = [goal_object]
    colors_shapes_2 = [goal_object]
    for n in range(num_objects-1):
        colors_shapes_1.append(valid_color_shape(action_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test = test))
    for n in range(num_objects-1):
        colors_shapes_2.append(valid_color_shape(action_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test = test))
    return(action_num, colors_shapes_1, colors_shapes_2)



if(__name__ == "__main__"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    for a, action in action_map.items():
        fig = plt.figure(figsize=(15, 15))
        fig.suptitle(action[1])
        gs = gridspec.GridSpec(len(shape_map), len(color_map), width_ratios=[1, 1, 1, 1, 1, 1])
        axs = []
        for s in range(len(shape_map)):
            row = []
            for c in range(len(color_map)):
                ax = fig.add_subplot(gs[s, c])
                ax.axis('off')
                if((a,c,s) in training_combos):
                    rect = patches.Rectangle((0, 0), 2, 2, color='gray', alpha=0.5)
                    ax.add_patch(rect)
                color = list(color_map.values())[c][1]
                shape = list(shape_map.values())[s][1]
                ax.text(.5, .5, f"{color}\n{shape}", va='center', ha='center', fontsize=20)
                row.append(ax)
            axs.append(row)
        for ax_2 in [axs[1][1], axs[2][2], axs[3][3], axs[4][5]]:
            p0 = axs[0][0].get_position()
            p1 = ax_2.get_position()
            width, height = p1.x1 - p0.x0, p1.y0 - p0.y1
            rect = patches.Rectangle((p0.x0, p0.y1), width, height, linewidth=5, edgecolor='black', facecolor='none')
            fig.add_artist(rect)
        plt.show()
        plt.close()

# Adjusting printing for computer-cluster.
def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)

# Adjusting PLT.
font = {'family' : 'sans-serif',
        #'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

# Duration functions.
start_time = datetime.datetime.now()

def duration(start_time = start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time# - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)

def estimate_total_duration(proportion_completed, start_time=start_time):
    if(proportion_completed != 0): 
        so_far = datetime.datetime.now() - start_time
        estimated_total = so_far / proportion_completed
        estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    else: estimated_total = "?:??:??"
    return(estimated_total)

# Memory functions. 
def cpu_memory_usage():
    process = psutil.Process(os.getpid())
    mem_usage_bytes = process.memory_info().rss  # rss is the Resident Set Size
    mem_usage_gb = mem_usage_bytes / (1024 ** 3)  # Convert bytes to gigabytes
    print('memory use:', mem_usage_gb, "gigabytes")
    
# Checking robot parts.

physicsClient = p.connect(p.DIRECT)
default_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId = physicsClient)
robot_index = p.loadURDF("pybullet_data/robot.urdf", (0, 0, 0), default_orn, useFixedBase=False, globalScaling = 1, physicsClientId = physicsClient)
sensors = []
for link_index in range(p.getNumJoints(robot_index, physicsClientId = physicsClient)):
    joint_info = p.getJointInfo(robot_index, link_index, physicsClientId = physicsClient)
    link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
    if("sensor" in link_name):
        sensors.append(link_name)
p.disconnect(physicsClientId = physicsClient)

# Arguments to parse. 
def literal(arg_string): return(ast.literal_eval(arg_string))
parser = argparse.ArgumentParser()

    # Meta 
parser.add_argument("--arg_title",          type=str,        default = "default",
                    help='Title of argument-set containign all non-default arguments.') 
parser.add_argument("--arg_name",           type=str,        default = "default",
                    help='Title of argument-set for human-understanding.') 
parser.add_argument("--agents",             type=int,        default = 36,
                    help='How many agents are trained in this job?')
parser.add_argument("--previous_agents",    type=int,        default = 0,
                    help='How many agents with this argument-set are trained in previous jobs?')
parser.add_argument("--init_seed",          type=float,      default = 777,
                    help='Random seed.')
parser.add_argument('--comp',               type=str,        default = "deigo",
                    help='Cluster name (deigo or saion).')
parser.add_argument('--device',             type=str,        default = device,
                    help='Which device to use for Torch.')
parser.add_argument('--show_duration',      type=bool,       default = False,
                    help='Should durations be printed?')

    # Things which have list-values.
parser.add_argument('--task_list',          type=literal,    default = ["1"],
                    help='List of tasks. Agent trains on each task based on epochs in epochs parameter.')
parser.add_argument('--epochs',             type=literal,    default = [10000],
                    help='List of how many epochs to train in each task.')
parser.add_argument('--time_scales',        type=literal,    default = [1],
                    help='Time-scales for MTRNN.')
parser.add_argument("--beta",               type=literal,    default = [2],
                    help='Relative importance of complexity in each layer.')
parser.add_argument("--hidden_state_eta",   type=literal,    default = [1],
                    help='Nonnegative valued, how much to consider hidden_state curiosity in each layer.') 

    # Simulation details
parser.add_argument('--max_object_distance',type=float,      default = 6,
                    help='How far objects can start from the agent.')
parser.add_argument('--min_object_separation',type=float,    default = 3,
                    help='How far objects must start from each other.')
parser.add_argument('--object_size',        type=float,      default = 2,
                    help='How large is the agent\'s body?')    
parser.add_argument('--body_size',          type=float,      default = 2,
                    help='How large is the agent\'s body?')    
parser.add_argument('--time_step',          type=float,      default = .2,
                    help='numSubSteps in pybullet environment.')
parser.add_argument('--steps_per_step',     type=int,        default = 30,
                    help='numSubSteps in pybullet environment.')

    # Agent details
parser.add_argument('--image_size',         type=int,        default = 16,
                    help='Dimensions of the images observed.')
parser.add_argument('--max_speed',          type=float,      default = 10,
                    help='Max wheel speed.')
parser.add_argument('--angular_scaler',     type=float,      default = .4,
                    help='How to scale angular velocity vs linear velocity.')
parser.add_argument('--min_shoulder_angle', type=float,      default = 0,
                    help='Agent\'s maximum shoulder velocity.')
parser.add_argument('--max_shoulder_angle', type=float,      default = pi/2,
                    help='Agent\'s maximum shoulder velocity.')
parser.add_argument('--max_shoulder_speed', type=float,      default = 8,
                    help='Max shoulder speed.')

    # Task details
parser.add_argument('--reward',             type=float,      default = 10,
                    help='Extrinsic reward for choosing correct action, shape, and color.') 
parser.add_argument('--max_steps',          type=int,        default = 10,
                    help='How many steps the agent can make in one episode.')
parser.add_argument('--step_lim_punishment',type=float,      default = -10,
                    help='Extrinsic punishment for taking max_steps steps.')
parser.add_argument('--step_cost',          type=float,      default = .975,
                    help='How much extrinsic rewards for exiting are reduced per step.')
parser.add_argument('--actions',            type=int,        default = 5,
                    help='Maximum count of actions in one episode.')
parser.add_argument('--objects',            type=int,        default = 4,
                    help='Maximum count of objects in one episode.')
parser.add_argument('--shapes',             type=int,        default = 5,
                    help='Maximum count of shapes in one episode.')
parser.add_argument('--colors',             type=int,        default = 6,
                    help='Maximum count of colors in one episode.')
parser.add_argument('--max_comm_len',       type=int,        default = 6,
                    help='Maximum length of communication.')
parser.add_argument('--watch_distance',     type=float,      default = 8,
                    help='How close must the agent watch the object to achieve watching.')
parser.add_argument('--watch_duration',     type=int,        default = 3,
                    help='How long must the agent watch the object to achieve watching.')
parser.add_argument('--push_amount',        type=float,      default = .75,
                    help='Needed distance of an object for push/pull/left/right.')

    # Rewards for distances
parser.add_argument('--dist_reward',        type=float,      default = 3,
                    help='Give agents a reward just for getting close to the correct object.')
parser.add_argument('--dist_reward_min',    type=float,      default = 3.5,
                    help='If agent closer to correct object that this, rewarded.')
parser.add_argument('--dist_reward_max',    type=float,      default = 5,
                    help='If agent farther to correct object that this, punished. If agent between min and max, agent relatively rewarded.')

    # Rewards for angles
parser.add_argument('--angle_reward',       type=float,      default = 3,
                    help='Give agents a reward just for pointing at the right object.')
parser.add_argument('--angle_reward_min',   type=float,      default = 30,
                    help='If agent pointing at correct object that this, rewarded.')
parser.add_argument('--angle_reward_max',   type=float,      default = 90,
                    help='If agent pointing farther from correct object that this, punished. If angle between min and max, agent relatively rewarded.')

    # Training
parser.add_argument('--capacity',           type=int,        default = 256,
                    help='How many episodes can the memory buffer contain.')
parser.add_argument('--batch_size',         type=int,        default = 32, 
                    help='How many episodes are sampled for each epoch.')      
parser.add_argument('--rgbd_scaler',        type=float,      default = 5, 
                    help='How much to consider rgbd prediction in accuracy compared to comm and sensors.')  
parser.add_argument('--comm_scaler',        type=float,      default = 1, 
                    help='How much to consider comm prediction in accuracy compared to rgbd and sensors.')       
parser.add_argument('--sensors_scaler',     type=float,      default = 1, 
                    help='How much to consider sensors prediction in accuracy compared to rgbd and comm.')       
parser.add_argument('--forward_lr',         type=float,      default = .0003,
                    help='Learning rate for forward model.')
parser.add_argument('--discriminator_lr',   type=float,      default = .0003,
                    help='Learning rate for discriminator model.')
parser.add_argument('--actor_lr',           type=float,      default = .0003,
                    help='Learning rate for actor model.')
parser.add_argument('--critic_lr',          type=float,      default = .0003,
                    help='Learning rate for critic model.')
parser.add_argument('--critics',            type=int,        default = 2,
                    help='How many critics?')  
parser.add_argument('--alpha_lr',           type=float,      default = .0003,
                    help='Learning rate for alpha value.') 
parser.add_argument('--alpha_text_lr',      type=float,      default = .0003,
                    help='Learning rate for alpha value.') 
parser.add_argument("--tau",                type=float,      default = .1,
                    help='Rate at which target-critics approach critics.')      
parser.add_argument('--GAMMA',              type=float,      default = .9,
                    help='How heavily critics consider the future.')
parser.add_argument("--d",                  type=int,        default = 2,
                    help='Delay for training actors.') 

    # Module  
parser.add_argument('--hidden_size',        type=int,        default = 64,
                    help='Parameters in hidden layers.')   
parser.add_argument('--pvrnn_mtrnn_size',   type=int,        default = 256,
                    help='Parameters in hidden layers pf PVRNN\'s mtrnn.')   
parser.add_argument('--state_size',         type=int,        default = 256,
                    help='Parameters in prior and posterior inner-states.')
parser.add_argument('--encode_char_size',   type=int,        default = 8,
                    help='Parameters in encoding.')   
parser.add_argument('--encode_rgbd_size',   type=int,        default = 128,
                    help='Parameters in encoding image.')   
parser.add_argument('--encode_comm_size',   type=int,        default = 128,
                    help='Parameters in encoding communicaiton.')   
parser.add_argument('--encode_sensors_size',  type=int,      default = 8,
                    help='Parameters in encoding sensors, angles, speed.')   
parser.add_argument('--encode_action_size', type=int,        default = 8,
                    help='Parameters in encoding action.')   
parser.add_argument('--dropout',            type=float,      default = .001,
                    help='Dropout percentage.')
parser.add_argument('--use_hsv',            type=literal,    default = True,
                    help='Should RGBD_In use hsv?')   
parser.add_argument('--pos_channels',       type=int,        default = 2,
                    help='How many channels for positions in rgbd?')   
parser.add_argument('--divisions',          type=int,        default = 1,
                    help='How many times should RBGD_Out double size to image-size?')      

    # Complexity 
parser.add_argument('--std_min',            type=int,        default = exp(-20),
                    help='Minimum value for standard deviation.')
parser.add_argument('--std_max',            type=int,        default = exp(2),
                    help='Maximum value for standard deviation.')

    # Entropy
parser.add_argument("--alpha",              type=literal,    default = 0,
                    help='Nonnegative value, how much to consider entropy. Set to None to use target_entropy.')        
parser.add_argument("--target_entropy",     type=float,      default = 0,
                    help='Target for choosing alpha if alpha set to None. Recommended: negative size of action-space.')      
parser.add_argument("--alpha_text",         type=literal,    default = 0,
                    help='Nonnegative value, how much to consider entropy regarding communication. Set to None to use target_entropy_text.')        
parser.add_argument("--target_entropy_text",type=float,      default = -2,
                    help='Target for choosing alpha_text if alpha_text set to None. Recommended: negative size of action-space.')      
parser.add_argument('--action_prior',       type=str,        default = "normal",
                    help='The actor can be trained based on normal or uniform distributions.')
parser.add_argument("--normal_alpha",       type=float,      default = 0,
                    help='Nonnegative value, how much to consider policy prior.')      

    # Curiosity
parser.add_argument("--curiosity",          type=str,        default = "none",
                    help='Which kind of curiosity: none, prediction_error, or hidden_state.')  
parser.add_argument("--dkl_max",            type=float,      default = 1,
                    help='Maximum value for clamping Kullback-Liebler divergence for hidden_state curiosity.')        
parser.add_argument("--prediction_error_eta", type=float,    default = 1,
                    help='Nonnegative value, how much to consider prediction_error curiosity.')          

    # Imitation
parser.add_argument("--delta",              type=float,      default = 0,
                    help='How much to consider action\'s similarity to recommended action.')  

    # Saving data
parser.add_argument('--keep_data',           type=int,       default = 10,
                    help='How many epochs should pass before saving data.')

parser.add_argument('--epochs_per_gen_test', type=int,       default = 10,
                    help='How many epochs should pass before trying generalization test.')

parser.add_argument('--epochs_per_episode_dict',type=int,    default = 2000,
                    help='How many epochs should pass before saving an episode.')
parser.add_argument('--agents_per_episode_dict',type=int,    default = 3,
                    help='How many agents to save episodes.')
parser.add_argument('--episodes_in_episode_dict',type=int,   default = 1,
                    help='How many episodes to save per agent.')

parser.add_argument('--epochs_per_agent_list',type=int,       default = 2000,
                    help='How many epochs should pass before saving agent model.')
parser.add_argument('--agents_per_agent_list',type=int,       default = 3,
                    help='How many agents to save.') 

try:
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
except:
    import sys ; sys.argv=[''] ; del sys           
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
    
def extend_list_to_match_length(target_list, length, value):
    while len(target_list) < length:
        target_list.append(value)
    return target_list

for arg_set in [default_args, args]:
    arg_set.steps_per_epoch = arg_set.max_steps
    arg_set.object_shape = arg_set.shapes + arg_set.colors
    arg_set.comm_shape = len(comm_map)
    arg_set.sensors_shape = len(sensors)
    arg_set.sensor_names = sensors
    arg_set.action_shape = 3
    arg_set.encode_obs_size = arg_set.encode_rgbd_size + arg_set.encode_comm_size + arg_set.encode_sensors_size
    arg_set.h_w_action_size = arg_set.pvrnn_mtrnn_size + arg_set.encode_action_size
    max_length = max(len(arg_set.time_scales), len(arg_set.beta), len(arg_set.hidden_state_eta))
    arg_set.time_scales = extend_list_to_match_length(arg_set.time_scales, max_length, 1)
    arg_set.beta = extend_list_to_match_length(arg_set.beta, max_length, 0)
    arg_set.hidden_state_eta = extend_list_to_match_length(arg_set.hidden_state_eta, max_length, 0)
    arg_set.layers = len(arg_set.time_scales)
        
args_not_in_title = ["arg_title", "id", "agents", "previous_agents", "init_seed", "keep_data", "epochs_per_pred_list", "episodes_in_pred_list", "agents_per_pred_list", "epochs_per_pos_list", "episodes_in_pos_list", "agents_per_pos_list"]
def get_args_title(default_args, args):
    if(args.arg_title[:3] == "___"): return(args.arg_title)
    name = "" ; first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index("arg_name")))
    for arg in arg_list:
        if(arg in args_not_in_title): pass 
        else: 
            default, this_time = getattr(default_args, arg), getattr(args, arg)
            if(this_time == default): pass
            elif(arg == "arg_name"):
                name += "{} (".format(this_time)
            else: 
                if first: first = False
                else: name += ", "
                name += "{}: {}".format(arg, this_time)
    if(name == ""): name = "default" 
    else:           name += ")"
    if(name.endswith(" ()")): name = name[:-3]
    parts = name.split(',')
    name = "" ; line = ""
    for i, part in enumerate(parts):
        if(len(line) > 50 and len(part) > 2): name += line + "\n" ; line = ""
        line += part
        if(i+1 != len(parts)): line += ","
    name += line
    return(name)

args.arg_title = get_args_title(default_args, args)

try: os.mkdir("saved")
except: pass
folder = "saved/" + args.arg_name
if(args.arg_title[:3] != "___" and not args.arg_name in ["default", "finishing_dictionaries", "plotting", "plotting_predictions", "plotting_positions"]):
    try: os.mkdir(folder)
    except: pass
    try: os.mkdir("saved/thesis_pics")
    except: pass
    try: os.mkdir("saved/thesis_pics/final")
    except: pass
if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

if(args == default_args): print("Using default arguments.")
else:
    for arg in vars(default_args):
        default, this_time = getattr(default_args, arg), getattr(args, arg)
        if(this_time == default): pass
        else: print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))



# For printing tensors as what they represent.
def string_to_onehots(s):
    s = ''.join([char.upper() if char.upper() in char_to_index else ' ' for char in s])
    onehots = []
    for char in s:
        tensor = torch.zeros(len(comm_map))
        tensor[char_to_index[char]] = 1
        onehots.append(tensor.unsqueeze(0))
    onehots = torch.cat(onehots, dim = 0)
    return onehots

def onehots_to_string(onehots):
    if(onehots == None):
        return("NONE")
    string = ''
    for tensor in onehots:
        index = torch.argmax(tensor).item()
        string += comm_map[index]
    return string

def many_onehots_to_strings(onehots):
    if(onehots == None):
        return("NONE")
    if onehots.dim() == 2: 
        return onehots_to_string(onehots)
    else:  
        return [many_onehots_to_strings(sub_tensor) for sub_tensor in onehots]

def action_to_string(action):
    while(len(action.shape) > 1):
        action = action.squeeze(0)
    string = "Left Wheel: {} ".format(round(action[0].item(),2))
    string += "Right Wheel: {} ".format(round(action[1].item(),2))
    string += "Shoulder: {} ".format(round(action[2].item(),2))
    return(string)



# Functions for relative value of actor-output to actions.
def relative_to(this, min, max):
    this = min + ((this + 1)/2) * (max - min)
    this = [min, max, this]
    this.sort()
    return(this[1])

def opposite_relative_to(this, min, max):
    return ((this - min) / (max - min)) * 2 - 1



# PyTorch functions.
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

def episodes_steps(this):
    return(this.shape[0], this.shape[1])

def pad_zeros(value, length):
    rows_to_add = length - value.size(-2)
    padding_shape = list(value.shape)
    padding_shape[-2] = rows_to_add
    padding = torch.zeros(padding_shape)
    padding
    padding[..., 0] = 1
    value = torch.cat([value, padding], dim=-2)
    return value

def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std, device):
    e = Normal(0, 1).sample(std.shape).to(device)
    return(mu + e * std)

def rnn_cnn(do_this, to_this):
    episodes, steps = episodes_steps(to_this)
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)

def extract_and_concatenate(tensor, expected_len):
    episodes, steps, _ = tensor.shape
    all_indices = []
    for episode in range(episodes):
        episode_indices = []
        for step in range(steps):
            ones_indices = tensor[episode, step].nonzero(as_tuple=True)[0]
            if(ones_indices.shape[0] < expected_len):
                ones_indices = torch.zeros((expected_len,)).int()
            episode_indices.append(ones_indices)
        episode_tensor = torch.stack(episode_indices)
        all_indices.append(episode_tensor)
    final_tensor = torch.stack(all_indices, dim=0)
    return final_tensor
    
def attach_list(tensor_list, device):
    updated_list = []
    for tensor in tensor_list:
        if isinstance(tensor, list):
            updated_sublist = [t.to(device) if t.device != device else t for t in tensor]
            updated_list.append(updated_sublist)
        else:
            updated_tensor = tensor.to(device) if tensor.device != device else tensor
            updated_list.append(updated_tensor)
    return updated_list

def detach_list(l): 
    return([element.detach() for element in l])
    
def memory_usage(device):
    print(device, ":", platform.node(), torch.cuda.memory_allocated(device), "out of", torch.cuda.max_memory_allocated(device))
    
    

def dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)



class ConstrainedConv1d(nn.Conv1d):
    def forward(self, input):
        return nn.functional.conv1d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
class Ted_Conv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_sizes = [1,2,3], stride = 1):
        super(Ted_Conv1d, self).__init__()
        
        self.Conv1ds = nn.ModuleList()
        for kernel, out_channel in zip(kernel_sizes, out_channels):
            padding = (kernel-1)//2
            layer = nn.Sequential(
                ConstrainedConv1d(
                    in_channels = in_channels,
                    out_channels = out_channel,
                    kernel_size = kernel,
                    padding = padding,
                    padding_mode = "reflect",
                    stride = stride))
            self.Conv1ds.append(layer)
                
    def forward(self, x):
        y = []
        for Conv1d in self.Conv1ds: y.append(Conv1d(x)) 
        return(torch.cat(y, dim = -2))
    
    
    
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
class Ted_Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_sizes = [(1,1),(3,3),(5,5)], stride = 1):
        super(Ted_Conv2d, self).__init__()
        
        self.Conv2ds = nn.ModuleList()
        for kernel, out_channel in zip(kernel_sizes, out_channels):
            if(type(kernel) == int): 
                kernel = (kernel, kernel)
            padding = ((kernel[0]-1)//2, (kernel[1]-1)//2)
            layer = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = in_channels,
                    out_channels = out_channel,
                    kernel_size = kernel,
                    padding = padding,
                    padding_mode = "reflect",
                    stride = stride))
            self.Conv2ds.append(layer)
                
    def forward(self, x):
        y = []
        for Conv2d in self.Conv2ds: y.append(Conv2d(x)) 
        return(torch.cat(y, dim = -3))
    
    

class ResidualBlock2d(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, paddings, padding_modes, activations, dropouts):
        super(ResidualBlock2d, self).__init__()
        
        # Ensure all parameter lists have compatible lengths
        assert len(channels) - 1 == len(kernel_sizes) == len(strides) == len(paddings) == \
               len(padding_modes) == len(activations) == len(dropouts) - 1, "All parameter lists must have compatible lengths."
        
        self.layers = nn.ModuleList()
        self.adjust_channels = channels[0] != channels[-1] or strides[0] != 1
        self.final_activation = activations[-1] is not None
        
        # Add the convolutional layers, batch normalization, activation functions, and dropout based on the parameters
        for i in range(len(channels) - 1):
            self.layers.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    padding_mode=padding_modes[i]))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))
            
            if activations[i] is not None:  # Check if the activation is specified
                self.layers.append(activations[i]())
            
            if dropouts[i] > 0:  # Add dropout layer if dropout rate is greater than 0
                self.layers.append(nn.Dropout2d(dropouts[i]))
                
        self.final_layer = nn.ModuleList()
        if self.final_activation:
            self.final_layer.append(activations[-1]())
        if dropouts[-1] > 0:
            self.final_layer.append(nn.Dropout2d(dropouts[-1]))
        self.final_layer.append(nn.BatchNorm2d(channels[-1]))
        
        # Add an adjustment layer to match the shortcut connection, if necessary
        if self.adjust_channels:
            self.adjustment_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=channels[0],
                    out_channels=channels[-1],
                    kernel_size=1,
                    stride=strides[0],  # Adjust based on the first stride value if needed
                    padding=0),
                nn.BatchNorm2d(channels[-1]))
        
    def forward(self, x):
        identity = x
        for layer in self.layers:
            x = layer(x)
        if self.adjust_channels:
            identity = self.adjustment_layer(identity)
        x += identity
        for layer in self.final_layer:
            x = layer(x)
        return x
    


class ResidualBlock1d(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, paddings, padding_modes, activations, dropouts):
        super(ResidualBlock1d, self).__init__()
        
        # Ensure all parameter lists have compatible lengths
        assert len(channels) - 1 == len(kernel_sizes) == len(strides) == len(paddings) == \
               len(padding_modes) == len(activations) == len(dropouts) - 1, "All parameter lists must have compatible lengths."
        
        self.layers = nn.ModuleList()
        self.adjust_channels = channels[0] != channels[-1] or strides[0] != 1
        self.final_activation = activations[-1] is not None
        
        # Add the convolutional layers, batch normalization, activation functions, and dropout based on the parameters
        for i in range(len(channels) - 1):
            self.layers.append(
                nn.Conv1d(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    padding_mode=padding_modes[i]))
            self.layers.append(nn.BatchNorm1d(channels[i+1]))
            
            if activations[i] is not None:  # Check if the activation is specified
                self.layers.append(activations[i]())
            
            if dropouts[i] > 0:  # Add dropout layer if dropout rate is greater than 0
                self.layers.append(nn.Dropout1d(dropouts[i]))
                
        self.final_layer = nn.ModuleList()
        if self.final_activation:
            self.final_layer.append(activations[-1]())
        if dropouts[-1] > 0:
            self.final_layer.append(nn.Dropout1d(dropouts[-1]))
        self.final_layer.append(nn.BatchNorm1d(channels[-1]))
        
        # Add an adjustment layer to match the shortcut connection, if necessary
        if self.adjust_channels:
            self.adjustment_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels[0],
                    out_channels=channels[-1],
                    kernel_size=1,
                    stride=strides[0],  # Adjust based on the first stride value if needed
                    padding=0),
                nn.BatchNorm1d(channels[-1]))
        
    def forward(self, x):
        identity = x
        for layer in self.layers:
            x = layer(x)
        if self.adjust_channels:
            identity = self.adjustment_layer(identity)
        x += identity
        for layer in self.final_layer:
            x = layer(x)
        return x
    
    
    
class DenseBlock(nn.Module):
    def __init__(self, input_channels, growth_rates, kernel_sizes, paddings, padding_modes, activations, dropouts):
        super(DenseBlock, self).__init__()
        
        assert len(growth_rates) == len(kernel_sizes) == len(activations) == len(dropouts)
        
        self.layers = nn.ModuleList()
        for i in range(len(growth_rates)):
            layer_channels = input_channels + sum(growth_rates[:i])
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(layer_channels),
                activations[i](),
                nn.Conv2d(
                    in_channels = layer_channels, 
                    out_channels = growth_rates[i], 
                    kernel_size = kernel_sizes[i], 
                    padding = paddings[i],
                    padding_mode = padding_modes[i]),
                nn.Dropout2d(dropouts[i])))
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            these_features = torch.cat(features, 1)
            new_features = layer(these_features)
            features.append(new_features)
        return torch.cat(features, 1)
    
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'position encoding' matrix in advance
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register 'pe' as a constant buffer that does not require gradients
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Add position encoding to input tensor
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
    
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, src, src_mask = None):
        src = self.encoder(src) * sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
    
    
    
class ImageTransformer(nn.Module):
    def __init__(self, in_channels, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(ImageTransformer, self).__init__()
        self.transformer = TransformerModel(ntoken, ninp, nhead, nhid, nlayers, dropout)
        
        # Resize and flatten image to fit transformer input
        self.resize = Resize((ninp, ninp))  # Assuming square images for simplicity
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels * ninp * ninp, ninp)  # Adapt to transformer input dimension
        
    def forward(self, img, src_mask = None):
        # img shape: [batch_size, in_channels, H, W]
        img = self.resize(img)
        img = self.flatten(img)
        img = self.linear(img)
        img = img.unsqueeze(0)  # Add sequence dimension
        output = self.transformer(img, src_mask)
        return output
    
    

def custom_loss(guess, target, max_shift=1):
    losses = []
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            shifted_guess = guess
        elif shift < 0:
            shifted_guess = guess[:, :, -shift:] 
            zeros = torch.zeros_like(guess[:, :, :-shift])
            shifted_guess = torch.cat((shifted_guess, zeros), dim=-1)
        else:
            shifted_guess = guess[:, :, :-shift] 
            zeros = torch.zeros_like(guess[:, :, -shift:])
            shifted_guess = torch.cat((zeros, shifted_guess), dim=-1)
        loss = F.cross_entropy(shifted_guess, target, reduction="none") + (1.1 * abs(shift))
        losses.append(loss)  
    min_loss = torch.stack(losses, dim=-1)
    min_loss = min_loss.min(dim=-1)[0]
    return min_loss
    
    
    
def calculate_similarity(recommended_actions, actor_actions):
    similarities = -F.mse_loss(recommended_actions, actor_actions, reduction='none').mean(-1)
    return similarities



def hsv_to_circular_hue(hsv_image):
    hue = hsv_image[:, 0, :, :] * 2 * torch.pi  # Convert hue from [0, 1] to [0, 2*pi]
    hue_sin = (torch.sin(hue) + 1) / 2
    hue_cos = (torch.cos(hue) + 1) / 2
    hsv_circular = torch.stack([hue_sin, hue_cos, hsv_image[:, 1, :, :], hsv_image[:, 2, :, :]], dim=1)
    return hsv_circular



def add_hsv_to_rgbd(rgbd):
    pass 




# For loading and plotting in final files.
real_names = {
    "d"  : "No Entropy, No Curiosity",
    "e"  : "Entropy",
    "n"  : "Prediction Error Curiosity",
    "f"  : "Hidden State Curiosity",
    "i"  : "Imitation",
    "en" : "Entropy and Prediction Error Curiosity",
    "ef" : "Entropy and Hidden State Curiosity",
    "ei" : "Entropy and Imitation",
    "ni" : "Prediction Error Curiosity and Imitation",
    "fi" : "Hidden State Curiosity and Imitation",
    "eni" : "Entropy, Prediction Error Curiosity, and Imitation",
    "efi" : "Entropy, Hidden State Curiosity, and Imitation"}

def add_this(name):
    keys, values = [], []
    for key, value in real_names.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        new_key = key + "_" + name 
        real_names[new_key] = value
add_this("hard")
add_this("many")

short_real_names = {
    "d"  : "N",
    "e"  : "E",
    "n"  : "P",
    "f"  : "H",
    "i"  : "I",
    "en" : "EP",
    "ef" : "EH",
    "ei" : "EI",
    "ni" : "PI",
    "fi" : "HI",
    "eni" : "EPI",
    "efi" : "EHI"}



def load_dicts(args):
    if(os.getcwd().split("/")[-1] != "saved"): os.chdir("saved")
    plot_dicts = [] ; min_max_dicts = []
        
    complete_order = args.arg_title[3:-3].split("+")
    order = [o for o in complete_order if not o in ["empty_space", "break"]]
    
    for name in order:
        got_plot_dicts = False ; got_min_max_dicts = False
        while(not got_plot_dicts):
            try:
                with open(name + "/" + "plot_dict.pickle", "rb") as handle: 
                    plot_dicts.append(pickle.load(handle)) ; got_plot_dicts = True
            except: print("Stuck trying to get {}'s plot_dicts...".format(name)) ; sleep(1)
        while(not got_min_max_dicts):
            try:
                with open(name + "/" + "min_max_dict.pickle", "rb") as handle: 
                    min_max_dicts.append(pickle.load(handle)) ; got_min_max_dicts = True 
            except: print("Stuck trying to get {}'s min_max_dicts...".format(name)) ; sleep(1)
    
    min_max_dict = {}
    for key in plot_dicts[0].keys():
        if(not key in ["args", "arg_title", "arg_name", "episode_dicts", "agent_lists", "spot_names", "steps", "goal_action"]):
            if(key == "hidden_state"):
                min_maxes = []
                for layer in range(len(min_max_dicts[0][key])):
                    minimum = None ; maximum = None
                    for mm_dict in min_max_dicts:
                        if(  minimum == None):                  minimum = mm_dict[key][layer][0]
                        elif(minimum > mm_dict[key][layer][0]): minimum = mm_dict[key][layer][0]
                        if(  maximum == None):                  maximum = mm_dict[key][layer][1]
                        elif(maximum < mm_dict[key][layer][1]): maximum = mm_dict[key][layer][1]
                    min_maxes.append((minimum, maximum))
                min_max_dict[key] = min_maxes
            else:
                minimum = None ; maximum = None
                for mm_dict in min_max_dicts:
                    if(mm_dict[key] != (None, None)):
                        if(  minimum == None):           minimum = mm_dict[key][0]
                        elif(minimum > mm_dict[key][0]): minimum = mm_dict[key][0]
                        if(  maximum == None):           maximum = mm_dict[key][1]
                        elif(maximum < mm_dict[key][1]): maximum = mm_dict[key][1]
                min_max_dict[key] = (minimum, maximum)
            
    final_complete_order = [] ; final_plot_dicts = []

    for arg_name in complete_order: 
        if(arg_name in ["break", "empty_space"]): 
            final_complete_order.append(arg_name)
        else:
            for plot_dict in plot_dicts:
                if(plot_dict["args"].arg_name == arg_name):    
                    final_complete_order.append(arg_name) 
                    final_plot_dicts.append(plot_dict)
                    
    while(len(final_complete_order) > 0 and final_complete_order[0] in ["break", "empty_space"]): 
        final_complete_order.pop(0)              
    
    return(final_plot_dicts, min_max_dict, complete_order)
# %%
