#%% 

# To do: most important 
#   You notice, not all objects allowed in freeplay
#   Try old arena. 
#   Give actions percentage chances when making a task.
#   Make it work FASTER. Trying float16 on cuda, getting NaN.
#   "push" task detected at odd times. Should "left" and "right" only win when object is in gaze?
#   Plotting sometimes shows big changes immediately after changing epoch-list values. 

# To do: less important 
#   Why the heck to win-rate plotting take SOOO LOOONG?
#   Allow multiple layers in PVRNN.
#   Try predicting multiple steps into the future.

import os
import pickle
import pybullet as p
from math import pi
from time import sleep
import builtins
import datetime 
import matplotlib
import argparse, ast
from math import exp
from random import choice
import torch
import psutil
from itertools import product
from collections import namedtuple

if(os.getcwd().split("/")[-1] != "communication"): os.chdir("communication")
print(os.getcwd())

torch.set_printoptions(precision=3, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    
    
#%%



Task = namedtuple('Task', ['char', 'name'])
task_map = {
    0:  Task("A", "FREEPLAY"),
    1:  Task("B", "WATCH"),
    2:  Task("C", "PUSH"),
    3:  Task("D", "PULL"),
    4:  Task("E", "LEFT"),
    5:  Task("F", "RIGHT")} 
task_name_list = [a.name for a in task_map.values()]

Color = namedtuple('Color', ['char', 'name', 'rgba'])
color_map = {
    0: Color("G", "RED",     (1,0,0,1)), 
    1: Color("H", "GREEN",   (0,1,0,1)),
    2: Color("I", "BLUE",    (0,0,1,1)),
    3: Color("J", "CYAN",    (0,1,1,1)), 
    4: Color("K", "PINK",    (1,0,1,1)), 
    5: Color("L", "YELLOW",  (1,1,0,1))} 
color_name_list = [c.name for c in color_map.values()]

Shape = namedtuple('Shape', ['char', 'name', 'file'])
data_path = "pybullet_data"
shape_files = [f.name for f in os.scandir(data_path + "/shapes") if f.name.endswith("urdf")]
shape_files.sort()
shape_letter_name_file = [[f.split("_")[0], f.split("_")[1][:-5], f] for f in shape_files]
shape_map = {int(num) : Shape(l, n, f) for num, (l, n, f) in enumerate(shape_letter_name_file)} 
shape_name_list = [s.name for s in shape_map.values()]

Object = namedtuple('Object', ['index', 'default_pos', 'color', 'shape'])
Goal = namedtuple('Goal', ['task', 'color', 'shape', 'parenting'])
Whole_Obs = namedtuple('Whole_Obs', ['rgbd', 'sensors', 'father_comm', 'mother_comm'])
Action = namedtuple('Action', ['wheels_shoulders', 'comm_out'])
ZP_ZQ_DKL = namedtuple("ZP_ZQ_DKL", ["zp", "zq", "dkl"])
To_Push = namedtuple('To_Push', ['rgbd', 'sensors', 'father_comm', 'mother_comm', 'wheels_shoulders', 'comm_out', 'reward', 'next_rgbd', 'next_sensors', 'next_father_comm', 'next_mother_comm', 'done'])

used_chars = list(
                 [a.char for a in task_map.values()] +
                 [c.char for c in color_map.values()] +
                 [s.char for s in shape_map.values()])
used_chars.sort()

comm_map = {k: v for k, v in {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}.items() if v in used_chars}
char_to_index = {v: k for k, v in comm_map.items()}



if(__name__ == "__main__"):
    print("Tasks:")
    for key, value in task_map.items():
        print(f"\t{key} : \t {value}")
    print("Colors:")
    for key, value in color_map.items():
        print(f"\t{key} : \t {value}")
    print("Shapes:")
    for key, value in shape_map.items():
        print(f"\t{key} : \t {value}")
        


#%%



def agent_to_english(agent_string):
    if(agent_string == "NONE"):
        return("NONE")
    english_string = ""
    for char in agent_string:
        translated = False
        for d in [task_map, color_map, shape_map]:
            for val in d.values():
                c = val.char
                n = val.name
                if(char == c and char != " "):
                    english_string += n + " "
                    translated = True
        if(not translated):
            english_string += f"_{char}_ "
    english_string = english_string.strip()
    return(english_string)

if(__name__ == "__main__"):
    print(agent_to_english("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    
    
    
#%%



all_combos = list(product(task_map.keys(), color_map.keys(), shape_map.keys()))

def train_or_test(a, c, s):
    train = True
    if(a == 0):
        return(train)
    if((s + c) % 2 == 0 and a % 2 == 0):
        train = False 
    if((s + c) % 2 == 1 and a % 2 == 1):
        train = False
    return(train)

training_combos = []
testing_combos = []
for (a, c, s) in all_combos:
    if(train_or_test(a, c, s)):
        training_combos.append((a, c, s))
    else:
        testing_combos.append((a, c, s))



if(__name__ == "__main__"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    for t, task in task_map.items():
        fig = plt.figure(figsize=(15, 15))
        fig.suptitle(task.name)
        gs = gridspec.GridSpec(len(shape_map), len(color_map), width_ratios=[1, 1, 1, 1, 1, 1])
        axs = []
        for s in range(len(shape_map)):
            row = []
            for c in range(len(color_map)):
                ax = fig.add_subplot(gs[s, c])
                ax.axis('off')
                if((a, c, s) in training_combos):
                    rect = patches.Rectangle((0, 0), 2, 2, color='gray', alpha=0.5)
                    ax.add_patch(rect)
                color = list(color_map.values())[c].name
                shape = list(shape_map.values())[s].name
                ax.text(.5, .5, f"{color}\n{shape}", va='center', ha='center', fontsize=20)
                row.append(ax)
            axs.append(row)
        plt.show()
        plt.close()
        
        
#%%

def valid_color_shape(task_num, used_combos, allowed_colors, allowed_shapes, test = False):
    if(test == None):
        these_combos = testing_combos + training_combos
    elif(test):
        these_combos = testing_combos
    else:
        these_combos = training_combos
    these_combos = [combo for combo in these_combos if combo[0] == task_num]
    these_combos = [(combo[1], combo[2]) for combo in these_combos if combo[1] in allowed_colors and combo[2] in allowed_shapes]
    these_combos = [combo for combo in these_combos if not combo in used_combos]
    color_num, shape_num = choice(these_combos)
    return(color_num, shape_num)

def make_objects_and_task(num_objects, allowed_tasks, allowed_colors, allowed_shapes, test = False):
    task_num = choice(allowed_tasks)
    if(task_num == 0):
        test = False
    goal_object = valid_color_shape(task_num, [], allowed_colors, allowed_shapes, test = test)
    colors_shapes_1 = [goal_object]
    colors_shapes_2 = [goal_object]
    for n in range(num_objects-1):
        color_shape = valid_color_shape(task_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test = test)
        colors_shapes_1.append(color_shape)
    for n in range(num_objects-1):
        color_shape = valid_color_shape(task_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test = test)
        colors_shapes_2.append(color_shape)
    task = task_map[task_num]
    colors_shapes_1 = [(color_map[c], shape_map[s]) for (c, s) in colors_shapes_1]
    colors_shapes_2 = [(color_map[c], shape_map[s]) for (c, s) in colors_shapes_2]
    return(task, colors_shapes_1, colors_shapes_2)



if(__name__ == "__main__"):
    print("Train")
    for i in range(1):
        print(make_objects_and_task(2, [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]))
    print("\nTest")
    for i in range(1):
        print(make_objects_and_task(2, [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4], test = True))
        
    print(make_objects_and_task(1, [0], [0], [0]))



#%%


    
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
num_sensors = len(sensors)

if(__name__ == "__main__"):
    print("Sensors:", num_sensors)



#%%



# Arguments to parse. 
def literal(arg_string): return(ast.literal_eval(arg_string))



parser = argparse.ArgumentParser()

    # Meta 
parser.add_argument("--arg_title",                      type=str,           default = "default",
                    help='Title of argument-set containing all non-default arguments.') 
parser.add_argument("--arg_name",                       type=str,           default = "default",
                    help='Title of argument-set for human-understanding.') 
parser.add_argument("--agents",                         type=int,           default = 36,
                    help='How many agents are trained in this job?')
parser.add_argument("--previous_agents",                type=int,           default = 0,
                    help='How many agents with this argument-set are trained in previous jobs?')
parser.add_argument("--init_seed",                      type=float,         default = 777,
                    help='Random seed.')
parser.add_argument('--comp',                           type=str,           default = "deigo",
                    help='Cluster name (deigo or saion).')
parser.add_argument('--device',                         type=str,           default = device,
                    help='Which device to use for Torch.')
parser.add_argument('--cpu',                            type=int,           default = 0,
                    help='Which cpu for affinity.')
parser.add_argument('--show_duration',                  type=bool,          default = False,
                    help='Should durations be printed?')
parser.add_argument('--save_agents',                    type=literal,       default = False,
                    help='Are we saving agents?')   
parser.add_argument('--load_agents',                    type=literal,       default = False,
                    help='Are we loading agents?')    

    # Simulation details   
parser.add_argument('--body_size',                      type=float,         default = 2,
                    help='How large is the agent\'s body?')    
parser.add_argument('--object_size',                    type=float,         default = 2,
                    help='How large is the are objects?') 
parser.add_argument('--time_step',                      type=float,         default = .2,
                    help='numSubSteps in pybullet environment.')
parser.add_argument('--steps_per_step',                 type=int,           default = 20,
                    help='numSubSteps in pybullet environment.')

    # Agent details
parser.add_argument('--image_size',                     type=int,           default = 16, #20,
                    help='Dimensions of the images observed.')
parser.add_argument('--max_comm_len',                   type=int,           default = 3,
                    help='Maximum length of communication.')
parser.add_argument('--max_speed',                      type=float,         default = 10,
                    help='Max wheel speed.')
parser.add_argument('--angular_scaler',                 type=float,         default = .4,
                    help='How to scale angular velocity vs linear velocity.')
parser.add_argument('--min_shoulder_angle',             type=float,         default = 0,
                    help='Agent\'s maximum shoulder velocity.')
parser.add_argument('--max_shoulder_angle',             type=float,         default = pi/2,
                    help='Agent\'s maximum shoulder velocity.')
parser.add_argument('--max_shoulder_speed',             type=float,         default = 8,
                    help='Max shoulder speed.')
parser.add_argument('--silent_mother',                  type=literal,       default = False,
                    help='Is the mother always silent? (No language-based curiosity.)')
parser.add_argument('--silent_mother_outside_free_play',                  type=literal,       default = False,
                    help='Is the mother silent outside free-play? (No language-based curiosity.)')

    # Task details
parser.add_argument('--reward',                         type=float,         default = 10,
                    help='Extrinsic reward for choosing correct task, shape, and color.') 
parser.add_argument('--max_steps',                      type=int,           default = 10,
                    help='How many steps the agent can make in one episode.')
parser.add_argument('--step_lim_punishment',            type=float,         default = 0,
                    help='Extrinsic punishment for taking max_steps steps.')
parser.add_argument('--step_cost',                      type=float,         default = .975,
                    help='How positive extrinsic rewards are reduced per step.')
parser.add_argument('--watch_distance',                 type=float,         default = 8,
                    help='How close must the agent watch the object to achieve watching.')
parser.add_argument('--watch_duration',                 type=int,           default = 3,
                    help='How long must the agent watch the object to achieve watching.')
parser.add_argument('--push_amount',                    type=float,         default = .75,
                    help='Needed distance of an object for push/pull/left/right.')
parser.add_argument('--pull_amount',                    type=float,         default = .25,
                    help='Needed distance of an object for push/pull/left/right.')
parser.add_argument('--left_right_amount',              type=float,         default = .25,
                    help='Needed distance of an object for push/pull/left/right.')

    # Training
parser.add_argument('--processors_epochs',              type=literal,       default = [("f", 100), ("w", 50), ("wpulr", 300)],
                    help='Agents perform these tasks for these numbers of episodes.')
parser.add_argument('--capacity',                       type=int,           default = 256,
                    help='How many episodes the recurrent memory buffer can contain.')
parser.add_argument('--batch_size',                     type=int,           default = 32, 
                    help='How many episodes are sampled for each epoch.')      
parser.add_argument('--rgbd_scaler',                    type=float,         default = 5, 
                    help='How much to consider rgbd prediction in accuracy.')  
parser.add_argument('--sensors_scaler',                 type=float,         default = .3, 
                    help='How much to consider sensors prediction in accuracy.')   
parser.add_argument('--father_comm_scaler',             type=float,         default = 3, 
                    help='How much to consider comm prediction in accuracy.')       
parser.add_argument('--mother_comm_scaler',             type=float,         default = 3, 
                    help='How much to consider comm prediction in accuracy.')       
parser.add_argument('--lr',                             type=float,         default = .0003,
                    help='Learning rate.')
parser.add_argument('--weight_decay',                   type=float,         default = .00001,
                    help='Weight decay for modules.')       
parser.add_argument('--critics',                        type=int,           default = 2,
                    help='How many critics?')  
parser.add_argument("--tau",                            type=float,         default = .1,
                    help='Rate at which target-critics approach critics.')      
parser.add_argument('--GAMMA',                          type=float,         default = .9,
                    help='How heavily bellman equations consider the future.')
parser.add_argument("--d",                              type=int,           default = 2,
                    help='Delay for training actors.') 

    # Module  
parser.add_argument('--hidden_size',                    type=int,           default = 64,
                    help='Parameters in hidden layers.')   
parser.add_argument('--pvrnn_mtrnn_size',               type=int,           default = 256,
                    help='Parameters in hidden layers 0f PVRNN\'s mtrnn.')   
parser.add_argument('--wheels_shoulders_encode_size',             type=int,           default = 8,
                    help='Parameters in encoding agent\'s wheels/shoulders action.')   
parser.add_argument('--mtrnn_sigmoid',                  type=literal,       default = True,
                    help='Should mtrnn use sigmoid or tanh?')   

parser.add_argument('--rgbd_state_size',                type=int,           default = 128,
                    help='Parameters in prior and posterior inner-states.')
parser.add_argument('--rgbd_encode_size',               type=int,           default = 128,
                    help='Parameters in encoding image.')   

parser.add_argument('--sensors_state_size',             type=int,           default = num_sensors,
                    help='Parameters in prior and posterior inner-states.')
parser.add_argument('--sensors_encode_size',            type=int,           default = num_sensors,
                    help='Parameters in encoding sensors, angles, speed.')   

parser.add_argument('--comm_state_size',                type=int,           default = 128,
                    help='Parameters in prior and posterior inner-states.')
parser.add_argument('--comm_encode_size',               type=int,           default = 128,
                    help='Parameters in encoding communicaiton.')   
parser.add_argument('--char_encode_size',               type=int,           default = 8,
                    help='Parameters in encoding individual characters.')   

parser.add_argument('--dropout',                        type=float,         default = .001,
                    help='Dropout percentage.')
parser.add_argument('--divisions',                      type=int,           default = 2,
                    help='How many times should RBGD_Out double size to image-size?')
parser.add_argument('--half',                           type=literal,       default = True,
                    help='Should the models use float16 instead of float32?')      

    # Entropy
parser.add_argument("--normal_alpha",                   type=float,         default = .1,
                    help='Nonnegative value, how much to consider policy prior.') 
parser.add_argument("--alpha",                          type=literal,       default = 0,
                    help='Nonnegative value, how much to consider entropy. Set to None to use target_entropy.')        
parser.add_argument("--target_entropy",                 type=float,         default = -1,
                    help='Target for choosing alpha if alpha set to None. Recommended: negative size of action-space.')      

parser.add_argument("--normal_alpha_text",                   type=float,         default = 0,
                    help='Nonnegative value, how much to consider policy prior.') 
parser.add_argument("--alpha_text",                     type=literal,       default = 0,
                    help='Nonnegative value, how much to consider entropy regarding communication. Set to None to use target_entropy_text.')        
parser.add_argument("--target_entropy_text",            type=float,         default = -2,
                    help='Target for choosing alpha_text if alpha_text set to None. Recommended: negative size of action-space.')      

    # Complexity 
parser.add_argument('--std_min',                        type=int,           default = exp(-20),
                    help='Minimum value for standard deviation.')
parser.add_argument('--std_max',                        type=int,           default = exp(2),
                    help='Maximum value for standard deviation.')
parser.add_argument("--beta_rgbd",                      type=float,         default = .03,
                    help='Relative importance of complexity for rgbd.')
parser.add_argument("--beta_sensors",                   type=float,         default = .3,
                    help='Relative importance of complexity for sensors.')     
parser.add_argument("--beta_father_comm",                      type=float,         default = .06,
                    help='Relative importance of complexity for father\'s communication.')
parser.add_argument("--beta_mother_comm",                      type=float,         default = .06,
                    help='Relative importance of complexity for mother\'s communication.')

    # Curiosity
parser.add_argument("--curiosity",                      type=str,           default = "none",
                    help='Which kind of curiosity: none, prediction_error, or hidden_state.')  
parser.add_argument("--dkl_max",                        type=float,         default = 1,
                    help='Maximum value for clamping Kullback-Liebler divergence for hidden_state curiosity.')         

parser.add_argument("--prediction_error_eta_rgbd",      type=float,         default = .3,
                    help='Nonnegative value, how much to consider prediction_error curiosity for rgbd.')    
parser.add_argument("--hidden_state_eta_rgbd",          type=float,         default = .3,
                    help='Nonnegative values, how much to consider hidden_state curiosity for rgbd.') 

parser.add_argument("--prediction_error_eta_sensors",   type=float,         default = .03,
                    help='Nonnegative value, how much to consider prediction_error curiosity for sensors.')    
parser.add_argument("--hidden_state_eta_sensors",       type=float,         default = .03,
                    help='Nonnegative values, how much to consider hidden_state curiosity for sensors.')   

parser.add_argument("--prediction_error_eta_father_comm",      type=float,         default = 1,
                    help='Nonnegative value, how much to consider prediction_error curiosity for father\'s communication.')   
parser.add_argument("--hidden_state_eta_father_comm",          type=float,         default = 1,
                    help='Nonnegative values, how much to consider hidden_state curiosity for father\'s communication.') 

parser.add_argument("--prediction_error_eta_mother_comm",      type=float,         default = 1,
                    help='Nonnegative value, how much to consider prediction_error curiosity for mother\'s communication.')   
parser.add_argument("--hidden_state_eta_mother_comm",          type=float,         default = 1,
                    help='Nonnegative values, how much to consider hidden_state curiosity for mother\'s communication.') 

    # Saving data
parser.add_argument('--keep_data',                      type=int,           default = 250,
                    help='How many epochs should pass before saving data.')

parser.add_argument('--epochs_per_gen_test',            type=int,           default = 25,
                    help='How many epochs should pass before trying generalization test.')

parser.add_argument('--epochs_per_episode_dict',        type=int,           default = 999999,
                    help='How many epochs should pass before saving an episode.')
parser.add_argument('--agents_per_episode_dict',        type=int,           default = 1,
                    help='How many agents to save episodes.')
parser.add_argument('--episodes_in_episode_dict',       type=int,           default = 1,
                    help='How many episodes to save per agent.')

parser.add_argument('--epochs_per_agent_list',          type=int,           default = 999999,
                    help='How many epochs should pass before saving agent model.')
parser.add_argument('--agents_per_agent_list',          type=int,           default = 3,
                    help='How many agents to save.') 

parser.add_argument('--epochs_per_values_for_composition',        type=int,           default = 5000,
                    help='How many epochs should pass before saving an episode.')
parser.add_argument('--agents_per_values_for_composition',       type=int,           default = 1,
                    help='How many agents to save episodes.')

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
    if(arg_set.comp == "deigo"):
        arg_set.half = False
    arg_set.steps_per_epoch = arg_set.max_steps
    arg_set.comm_shape = len(comm_map)
    arg_set.sensors_shape = num_sensors
    arg_set.sensor_names = sensors
    arg_set.wheels_shoulders_shape = 4
    arg_set.complete_action_encode_size = arg_set.wheels_shoulders_encode_size + arg_set.comm_encode_size
    arg_set.encode_obs_size = arg_set.rgbd_encode_size + arg_set.sensors_encode_size + 2 * arg_set.comm_encode_size
    arg_set.h_w_action_size = arg_set.pvrnn_mtrnn_size + arg_set.wheels_shoulders_encode_size # + arg_set.complete_action_encode_size

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

save_file = f"saved_{args.comp}"
try: os.mkdir(f"{save_file}")
except: pass
folder = f"{save_file}/{args.arg_name}"
if(args.arg_title[:3] != "___" and not args.arg_name in ["default", "finishing_dictionaries", "plotting", "plotting_predictions", "plotting_positions"]):
    try: os.mkdir(folder)
    except: pass
    try: os.mkdir(folder + "/agents")
    except: pass
    try: os.mkdir(f"{save_file}/thesis_pics")
    except: pass
    try: os.mkdir(f"{save_file}/thesis_pics/final")
    except: pass
if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

if(args == default_args): print("Using default arguments.")
else:
    for arg in vars(default_args):
        default, this_time = getattr(default_args, arg), getattr(args, arg)
        if(this_time != default):
            print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))
        elif(arg == "device"):
            print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))
            
            
            
#%%

def wheels_shoulders_to_string(task):
    while(len(task.shape) > 1):
        task = task.squeeze(0)
    string = "Left Wheel: {} ".format(round(task[0].item(),2))
    string += "Right Wheel: {} ".format(round(task[1].item(),2))
    string += "Left Shoulder: {} ".format(round(task[2].item(),2))
    string += "Right Shoulder: {} ".format(round(task[3].item(),2))
    return(string)

def onehots_to_string(onehots):
    if(onehots == None):
        return("NONE")
    string = ''
    for tensor in onehots:
        if(torch.all(tensor == 0)):
            string += "A"
        else:
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
    
def string_to_onehots(s):
    s = ''.join([char.upper() if char.upper() in char_to_index else 'A' for char in s])
    onehots = []
    for char in s:
        tensor = torch.zeros(len(comm_map))
        index = char_to_index[char]
        tensor[index] = 1
        onehots.append(tensor.unsqueeze(0))
    onehots = torch.cat(onehots, dim = 0)
    return onehots

def strings_to_human(strings):
    human_strings = []
    for s in strings:
        human_s = ""
        for c in s:
            known = False
            for d in [task_map, color_map, shape_map]:
                for val in d.values():
                    if val.char == c:
                        known = True
                        human_s += val.name + " "
            if(not known):
                human_s += f"_{c}_ "
        human_strings.append(human_s[:-1])
    return(human_strings)

def goal_to_onehots(goal):
    if(goal.task.name == "FREEPLAY"):
        string = "AAA"
    else:    
        string = goal.task.char + goal.color.char + goal.shape.char
    return(string_to_onehots(string))

def goal_to_human(goal):
    string = goal.task.char + goal.color.char + goal.shape.char
    if(string[0] == "AAA"):
        for_human = "silent"
    else:
        for_human = strings_to_human([string][0])
    return(for_human)



if(__name__ == "__main__"):
    #task = torch.tensor((0, 0, 0, 0))
    #print("Task to string:", task_to_string(task))
    
    onehots = string_to_onehots("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    print("String to onehots:", onehots)
    
    many_onehots = torch.stack([onehots, onehots, onehots], dim = 0)
    strings = many_onehots_to_strings(many_onehots)
    print("Many onehots to strings:", strings)
    
    human_strings = strings_to_human(strings)
    print("Strings to human:", human_strings)



#%%
        


# Functions for relative value of actor-output to actions.
def relative_to(this, min, max):
    this = min + ((this + 1)/2) * (max - min)
    this = [min, max, this]
    this.sort()
    return(this[1])

def opposite_relative_to(this, min, max):
    return ((this - min) / (max - min)) * 2 - 1



# PyTorch functions.

def how_many_nans(tensor, place = "tensor"):
    if(tensor == None):
        return
    nan_count = torch.isnan(tensor).sum().item()
    if(nan_count > 0):
        print(f'nans in {place}: \t{nan_count}.')



def dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)
    
    

def load_dicts(args):
    if(os.getcwd().split("/")[-1] != save_file): os.chdir(save_file)
    plot_dicts = [] ; min_max_dicts = []
        
    complete_order = args.arg_title[3:-3].split("+")
    order = [o for o in complete_order if not o in ["empty_space", "break"]]
        
    for name in order:
        print(f"Loading dictionaries for {name}...")
        got_plot_dicts = False ; got_min_max_dicts = False
        while(not got_plot_dicts):
            try:
                with open(name + "/" + "plot_dict.pickle", "rb") as handle: 
                    plot_dicts.append(pickle.load(handle)) ; got_plot_dicts = True
            except Exception as e:
                print(e) 
                print("Stuck trying to get {}'s plot_dicts...".format(name)) ; sleep(1)
        while(not got_min_max_dicts):
            try:
                with open(name + "/" + "min_max_dict.pickle", "rb") as handle: 
                    min_max_dicts.append(pickle.load(handle)) ; got_min_max_dicts = True 
            except: 
                print("Stuck trying to get {}'s min_max_dicts...".format(name)) ; sleep(1)
    print("Loaded all dicts! Making min/max dict...")
    
    min_max_dict = {}
    for key in plot_dicts[0].keys():
        if(not key in ["args", "arg_title", "arg_name", "all_processor_names", "values_for_composition", "episode_dicts", "agent_lists", "spot_names", "steps", "goal_task"]):
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
    print("Made min/max dict!")
            
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
        
    print("Done with Load Dicts!")          
    
    return(final_plot_dicts, min_max_dict, complete_order)
# %%