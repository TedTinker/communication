#%% 

# To do:
#   Are are generalized win-rates so weird? Everything but "all" look like they're only half full.
#   Try adding longer durations instead of just 3; episodes are ending so fast!

import os
import pickle
import pybullet as p
from time import sleep
import builtins
import datetime 
import matplotlib
import argparse, ast
from math import exp, log, pi
from random import choice, choices
import torch
import psutil
from itertools import product
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np

if(os.getcwd().split("/")[-1] != "communication"): os.chdir("communication")
print(f"\n\nWorking in: {os.getcwd()}\n\n")

torch.set_printoptions(precision=3, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

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

def print_duration(start_time, end_time, text = None, end_text = ""):
    if(text == None):
        print(f"{end_time - start_time}{end_text}")
    else:
        print(f"{text}: {end_time - start_time}{end_text}")

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



class Task:
    def __init__(self, char, name):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
    def __str__(self):
        return(f"{self.char}, {self.name}")
    
task_map = {
    0:  Task("A", "SILENCE"),
    1:  Task("B", "WATCH"),
    2:  Task("C", "BE NEAR"),
    3:  Task("D", "TOUCH THE TOP"),
    4:  Task("E", "PUSH FORWARD"),     
    5:  Task("F", "PUSH LEFT"),   
    6:  Task("G", "PUSH RIGHT")}    
max_len_taskname = max([len(t.name) for t in task_map.values()])
task_name_list = [task.name for task in task_map.values()]


        
class Color:
    def __init__(self, char, name, rgba):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
    def __str__(self):
        return(f"{self.char}, {self.name}")
    
color_map = {
    0: Color("H", "RED",        (1,0,0,1)), 
    1: Color("I", "GREEN",      (0,1,0,1)),
    2: Color("J", "BLUE",       (0,0,1,1)),
    3: Color("K", "CYAN",       (0,1,1,1)), 
    4: Color("L", "MAGENTA",    (1,0,1,1)), 
    5: Color("M", "YELLOW",     (1,1,0,1))} 
max_len_color_name = max([len(c.name) for c in color_map.values()])
color_name_list = [c.name for c in color_map.values()]


        
class Shape:
    def __init__(self, char, file_name):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        self.name = file_name.split("_")[-1][:-5]
        
    def __str__(self):
        return(f"{self.char}, {self.name}")
    
shape_files = [f.name for f in os.scandir("pybullet_data/shapes") if f.name.endswith("urdf")] 
shape_files.sort()
shape_letter_file = [[f.split("_")[0], f] for f in shape_files]
shape_map = {i : Shape(l, f) for i, (l, f) in enumerate(shape_letter_file)} 
max_len_shape_name = max([len(s.name) for s in shape_map.values()])
shape_name_list = [s.name for s in shape_map.values()]



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


        
class Goal:
    def __init__(self, task, color, shape, parenting):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
        one_hots = torch.zeros((3, len(task_map) + len(color_map) + len(shape_map)))
        for i, char in enumerate([self.task.char, self.color.char, self.shape.char]):
            index = ord(char) - ord('A')
            one_hots[i, index] = 1
        self.one_hots = one_hots
        self.char_text = f"{self.task.char}{self.color.char}{self.shape.char}"
        self.human_text = f"{self.task.name} {self.color.name} {self.shape.name}"
        
    def human_friendly_text(self, command = True):
        return(f"{'Command' if command else 'Report'}: {self.human_text}")
        
empty_goal = Goal(task_map[0], task_map[0], task_map[0], parenting = False)



def get_goal_from_one_hots(one_hots):
    while(len(one_hots.shape) > 2):
        one_hots = one_hots.squeeze(0)
    task_one_hot = one_hots[0, : len(task_map)]
    color_one_hot = one_hots[1, len(task_map) : len(task_map) + len(color_map)]
    shape_one_hot = one_hots[2, len(task_map) + len(color_map) : len(task_map) + len(color_map) + len(shape_map)]
    
    task_index = torch.argmax(task_one_hot).item()
    color_index = torch.argmax(color_one_hot).item()
    shape_index = torch.argmax(shape_one_hot).item()
            
    task = task_map[task_index]
    color = color_map[color_index]
    shape = shape_map[shape_index]
    
    goal = Goal(task, color, shape, parenting=False)
    if(task.name == "SILENCE"):
        goal = empty_goal
    return goal


        
class Obs:
    def __init__(self, vision, touch, command_voice, report_voice):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
class Action:
    def __init__(self, wheels_joints, voice_out):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
class To_Push:
    def __init__(self, obs, action, reward, next_obs, done):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
    def push(self, memory):
        memory.push(
            self.obs.vision.to("cpu"),
            self.obs.touch.to("cpu"),
            self.obs.command_voice.to("cpu"),
            self.obs.report_voice.to("cpu"),
            self.action.wheels_joints.to("cpu"), 
            self.action.voice_out.to("cpu"),
            self.reward, 
            self.next_obs.vision.to("cpu"),
            self.next_obs.touch.to("cpu"),
            self.next_obs.command_voice.to("cpu"), 
            self.next_obs.report_voice.to("cpu"), 
            self.done)

class Inner_States:
    def __init__(self, zp, zq, dkl):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})



used_chars = list(
                 [t.char for t in task_map.values()] +
                 [c.char for c in color_map.values()] +
                 [s.char for s in shape_map.values()])
used_chars.sort()



voice_map = {k: v for k, v in {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}.items() if v in used_chars}



char_to_index = {v: k for k, v in voice_map.items()}



if(__name__ == "__main__"):
    print("\n\nEmpty Goal:")
    example = empty_goal
    print(example.one_hots)
    print(example.char_text)
    print(example.human_text)
    print(get_goal_from_one_hots(example.one_hots).human_text)


    print("\n\nExample Goal:")
    example = Goal(task_map[1], color_map[2], shape_map[2], parenting = False)
    print(example.one_hots)
    print(example.char_text)
    print(example.human_text)
    print(get_goal_from_one_hots(example.one_hots).human_text)

    print("\n\nExample Goal:")
    example = Goal(task_map[4], color_map[3], shape_map[3], parenting = False)
    print(example.one_hots)
    print(example.char_text)
    print(example.human_text)
    print(get_goal_from_one_hots(example.one_hots).human_text)
    print("\n\n")



#%%



all_combos = list(product(task_map.keys(), color_map.keys(), shape_map.keys()))
#training_combos = [(a, c, s) for (a, c, s) in all_combos if 
#                   a == 0 or a == 1 or
#                   ((s + c) % 2 == 1 and a % 2 == 0) or
#                   ((s + c) % 2 == 0 and a % 2 == 1)]
training_combos = [(a, c, s) for (a, c, s) in all_combos if 
                    a == 0 or a == 1 or
                    
                   (a == 2 and (s, c) in 
                    [(0, 0), (0, 1), (0, 2), (0, 3),
                            (1, 1), (1, 2), (1, 3), (1, 4),
                                    (2, 2), (2, 3), (2, 4), (2, 5),
                    (3, 0),                 (3, 3), (3, 4), (3, 5),
                    (4, 0), (4, 1),                 (4, 4), (4, 5)]) or
                   
                    (a == 3 and (s, c) in 
                    [       (0, 1), (0, 2), (0, 3), (0, 4),
                                    (1, 2), (1, 3), (1, 4), (1, 5),
                                            (2, 3), (2, 4), (2, 5),
                    (3, 0), (3, 1),                 (3, 4), (3, 5),
                    (4, 0), (4, 1), (4, 2),                 (4, 5)]) or
                    
                    (a == 4 and (s, c) in 
                    [               (0, 2), (0, 3), (0, 4),
                                    (1, 2), (1, 3), (1, 4), (1, 5),
                                            (2, 3), (2, 4), (2, 5),
                    (3, 0), (3, 1),                 (3, 4), (3, 5),
                    (4, 0), (4, 1), (4, 2),                 (4, 5)]) or
                    
                    (a == 5 and (s, c) in 
                    [       (0, 1), (0, 2), (0, 3), (0, 4),
                                    (1, 2), (1, 3), (1, 4), (1, 5),
                                            (2, 3), (2, 4), (2, 5),
                    (3, 0), (3, 1),                 (3, 4), (3, 5),
                    (4, 0), (4, 1), (4, 2),                 (4, 5)]) or
                    
                    (a == 6 and (s, c) in 
                    [       (0, 1), (0, 2), (0, 3), (0, 4),
                                    (1, 2), (1, 3), (1, 4), (1, 5),
                                            (2, 3), (2, 4), (2, 5),
                    (3, 0), (3, 1),                 (3, 4), (3, 5),
                    (4, 0), (4, 1), (4, 2),                 (4, 5)])
                   
                   ]

testing_combos = [combo for combo in all_combos if not combo in training_combos]



if(__name__ == "__main__"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    for a, task in task_map.items():
        fig = plt.figure(figsize=(15, 15))
        fig.suptitle(task.name)
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
                color = list(color_map.values())[c].name
                shape = list(shape_map.values())[s].name
                ax.text(.5, .5, f"{color}\n{shape}", va='center', ha='center', fontsize=20)
                row.append(ax)
            axs.append(row)
        plt.show()
        plt.close()
        
        
        
#%%



def valid_color_shape(task_num, other_shape_colors, allowed_colors, allowed_shapes, test = False):
    if(test == None):
        these_combos = testing_combos + training_combos
    elif(test):
        these_combos = testing_combos
    else:
        these_combos = training_combos
    these_combos = [combo for combo in these_combos if combo[0] == task_num]
    these_combos = [(combo[1], combo[2]) for combo in these_combos if combo[1] in allowed_colors and combo[2] in allowed_shapes]
    if(test != None):
        these_combos = [combo for combo in these_combos if not combo in other_shape_colors]
    color_num, shape_num = choice(these_combos)
    return(color_num, shape_num)

def make_objects_and_task(num_objects, allowed_tasks_and_weights, allowed_colors, allowed_shapes, test = False):
    #print(f"num_objects {num_objects}, allowed_tasks_and_weights {allowed_tasks_and_weights}, allowed_colors {allowed_colors}, allowed_shapes {allowed_shapes}, test {test}")
    tasks   = [v for v, w in allowed_tasks_and_weights]
    weights = [w for v, w in allowed_tasks_and_weights]
    task_num = choices(tasks, weights=weights, k=1)[0]
    
    goal_object = valid_color_shape(task_num, [], allowed_colors, allowed_shapes, test = test)
    colors_shapes_1 = [goal_object]
    colors_shapes_2 = [goal_object]
    for n in range(num_objects-1):
        colors_shapes_1.append(valid_color_shape(task_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test = test))
    for n in range(num_objects-1):
        colors_shapes_2.append(valid_color_shape(task_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test = test))
    
    task = task_map[task_num]
    colors_shapes_1 = [(color_map[color_index], shape_map[shape_index]) for color_index, shape_index in colors_shapes_1]
    colors_shapes_2 = [(color_map[color_index], shape_map[shape_index]) for color_index, shape_index in colors_shapes_2]
    return(task, colors_shapes_1, colors_shapes_2)


        
if(__name__ == "__main__"):
    print("Train")
    for i in range(1):
        task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(2, [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
        print(task.name, [(color.name, shape.name) for color, shape in colors_shapes_1], [(color.name, shape.name) for color, shape in colors_shapes_2])
    #print("\nTest")
    #for i in range(1):
    #    task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(2, [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4], test = True)
    #    print(task.name, [(color.name, shape.name) for color, shape in colors_shapes_1], [(color.name, shape.name) for color, shape in colors_shapes_2])
        
        
        
        
#%% 



# Arguments to parse. 
def literal(arg_string): return(ast.literal_eval(arg_string))



parser = argparse.ArgumentParser()

    # Stuff I'm testing right now   
parser.add_argument('--robot_name',                     type=str,           default = "robot",
                    help='Options: two_side_arm, one_head_arm.') 
parser.add_argument('--prefer_top',                     type=literal,       default = False,
                    help='Should topping overwrite pushing?')    
parser.add_argument('--harder_left_right',              type=literal,       default = False,
                    help='Should pushing left/right be the more demanding version?')    
parser.add_argument("--harder_left_right_amount",       type=float,         default = pi/24,
                    help='If using the harder_left_right, how far must the object be pushed from one side to the other?')
parser.add_argument('--max_wheel_speed_for_left',       type=float,         default = 5,
                    help='How close must the agent watch the object to achieve pushing left or right.')
parser.add_argument('--be_near_distance',               type=float,         default = 3,
                    help='How close must the agent watch the object to achieve be_near.')

parser.add_argument('--global_push_amount',             type=float,         default = .1,
                    help='Needed distance of an object for push/left/right.')
parser.add_argument('--local_push_limit',          type=float,         default = .3,
                    help='Prevent bogus pushing by requiring local stillness.')
parser.add_argument('--global_left_right_amount',       type=float,         default = .2,
                    help='Needed distance of an object for push/left/right.')
parser.add_argument('--local_left_right_amount',        type=float,         default = .25,
                    help='Needed distance of an object for push/left/right.')
    

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
parser.add_argument('--local',                          type=bool,          default = False,
                    help='Is this running on a local machine for testing?')
parser.add_argument('--show_duration',                  type=bool,          default = False,
                    help='Should durations be printed?')
parser.add_argument('--save_agents',                    type=literal,       default = False,
                    help='Are we saving agents?')   
parser.add_argument('--load_agents',                    type=literal,       default = False,
                    help='Are we loading agents?')    



    # Things which have list-values.
parser.add_argument('--epochs',                         type=int,       default = 50000,
                    help='List of processors. Agent trains on each processor based on epochs in epochs parameter.')
parser.add_argument('--processor',                      type=str,       default = "all",
                    help='List of processors. Agent trains on each processor based on epochs in epochs parameter.')
    

    # Simulation details
parser.add_argument('--time_step',                      type=float,         default = .1,
                    help='numSubSteps in pybullet environment.')
parser.add_argument('--steps_per_step',                 type=int,           default = 20,
                    help='numSubSteps in pybullet environment.')
parser.add_argument('--min_object_separation',          type=float,         default = 3,
                    help='How far objects must start from each other.')
parser.add_argument('--max_object_distance',            type=float,         default = 4,
                    help='How far objects can start from the agent.')
parser.add_argument('--object_size',                    type=float,         default = 2,
                    help='How large is the agent\'s body?')    
parser.add_argument('--body_size',                      type=float,         default = 2,
                    help='How large is the agent\'s body?')        
parser.add_argument('--force',                          type=float,         default = 15000,
                    help='Force for moving joints.') 
parser.add_argument('--gravity',                        type=float,         default = -9.8,
                    help='Force of gravity.') 



    # Agent details
parser.add_argument('--image_size',                     type=int,           default = 16, #20,
                    help='Dimensions of the images observed.')
parser.add_argument('--max_wheel_acceleration',         type=float,         default = 100000,
                    help='Max wheel speed.')
parser.add_argument('--max_wheel_speed',                type=float,         default = 10,
                    help='Max wheel speed.')
parser.add_argument('--angular_scaler',                 type=float,         default = .4,
                    help='How to scale angular velocity vs linear velocity.')

parser.add_argument('--max_joint_speed',                type=float,         default = 8,
                    help='Max joint speed.')
parser.add_argument('--max_joint_1_angle',              type=float,         default = pi/6,
                    help='Max yaw angle.')
parser.add_argument('--min_joint_2_angle',              type=float,         default = -pi/2,
                    help='Max yaw angle.')
parser.add_argument('--max_joint_2_angle',              type=float,         default = 0,
                    help='Max yaw angle.')



    # Processor details
parser.add_argument('--reward',                         type=float,         default = 10,
                    help='Extrinsic reward for choosing correct task, shape, and color.') 
parser.add_argument('--wrong_object_punishment',        type=float,         default = 0,
                    help='Negative reward for punishing doing anything to the wrong object (except watching).') 
parser.add_argument('--reward_inflation_type',          type=str,           default = "None",
                    help='How should reward increase?')   
parser.add_argument('--max_steps',                      type=int,           default = 30,     
                    help='How many steps the agent can make in one episode.')
parser.add_argument('--step_lim_punishment',            type=float,         default = 0,
                    help='Extrinsic punishment for taking max_steps steps.')
parser.add_argument('--step_cost',                      type=float,         default = .99,    
                    help='How much extrinsic rewards are reduced per step.')
parser.add_argument('--max_voice_len',                  type=int,           default = 3,
                    help='Maximum length of voice.')



parser.add_argument('--watch_duration',                 type=int,           default = 3,
                    help='How long must the agent watch the object to achieve watching.')
parser.add_argument('--be_near_duration',               type=int,           default = 3,
                    help='How long must the agent watch the object to achieve watching.')
parser.add_argument('--top_duration',                   type=int,           default = 3,   
                    help='How long must the agent watch the object to achieve watching.')
parser.add_argument('--push_duration',                  type=int,           default = 3,
                    help='How long must the agent watch the object to achieve watching.')
parser.add_argument('--left_duration',                  type=int,           default = 3,   
                    help='How long must the agent watch the object to achieve watching.')

parser.add_argument('--pointing_at_object_for_watch',   type=float,         default = pi/6,
                    help='How close must the agent watch the object to achieve watching or pushing.')
parser.add_argument('--pointing_at_object_for_left',    type=float,         default = pi/3,
                    help='How close must the agent watch the object to achieve pushing left or right.')

parser.add_argument('--watch_distance',                 type=float,         default = 8,
                    help='How close must the agent watch the object to achieve watching.')
parser.add_argument('--top_arm_min_angle',              type=float,         default = pi/12,
                    help='How elevated the agent\'s arm must be to touch the object from above.')


    # Module  
parser.add_argument('--hidden_size',                    type=int,           default = 64,
                    help='Parameters in hidden layers.')   
parser.add_argument('--pvrnn_mtrnn_size',               type=int,           default = 256,
                    help='Parameters in hidden layers 0f PVRNN\'s mtrnn.')   
parser.add_argument('--vision_state_size',                type=int,           default = 128,
                    help='Parameters in prior and posterior inner-states.')
parser.add_argument('--voice_state_size',               type=int,           default = 128,
                    help='Parameters in prior and posterior inner-states.')

parser.add_argument('--char_encode_size',               type=int,           default = 8,
                    help='Parameters in encoding.')   
parser.add_argument('--vision_encode_size',               type=int,           default = 128,
                    help='Parameters in encoding image.')   
parser.add_argument('--voice_encode_size',              type=int,           default = 128,
                    help='Parameters in encoding voice.')   
parser.add_argument('--wheels_joints_encode_size',   type=int,           default = 8,
                    help='Parameters in encoding wheels_joints.')   

parser.add_argument('--dropout',                        type=float,         default = .001,
                    help='Dropout percentage.')
parser.add_argument('--divisions',                      type=int,           default = 2,
                    help='How many times should RBGD_Out double size to image-size?')
parser.add_argument('--half',                           type=literal,       default = True,
                    help='Should the models use float16 instead of float32?')      



    # Training
parser.add_argument('--capacity',                       type=int,           default = 256,
                    help='How many episodes can the memory buffer contain.')
parser.add_argument('--batch_size',                     type=int,           default = 32, 
                    help='How many episodes are sampled for each epoch.')       
parser.add_argument('--weight_decay',                   type=float,         default = .00001,
                    help='Weight decay for modules.')       
parser.add_argument('--lr',                             type=float,         default = .0003,
                    help='Learning rate.')
parser.add_argument('--critics',                        type=int,           default = 2,
                    help='How many critics?')  
parser.add_argument("--tau",                            type=float,         default = .1,
                    help='Rate at which target-critics approach critics.')      
parser.add_argument('--GAMMA',                          type=float,         default = .9,
                    help='How heavily critics consider the future.')
parser.add_argument("--d",                              type=int,           default = 2,
                    help='Delay for training actors.') 



    # Entropy
parser.add_argument("--alpha",                          type=literal,       default = 0,
                    help='Nonnegative value, how much to consider entropy. Set to None to use target_entropy.')        
parser.add_argument("--target_entropy",                 type=float,         default = -1,
                    help='Target for choosing alpha if alpha set to None. Recommended: negative size of wheels_joints-space.')      
parser.add_argument("--alpha_text",                     type=literal,       default = 0,
                    help='Nonnegative value, how much to consider entropy regarding agent voice. Set to None to use target_entropy_text.')        
parser.add_argument("--target_entropy_text",            type=float,         default = -2,
                    help='Target for choosing alpha_text if alpha_text set to None. Recommended: negative size of voice_out-space.')      
parser.add_argument("--normal_alpha",                   type=float,         default = 0,
                    help='Nonnegative value, how much to consider policy prior.') 



    # Curiosity
parser.add_argument('--std_min',                        type=int,           default = exp(-20),
                    help='Minimum value for standard deviation.')
parser.add_argument('--std_max',                        type=int,           default = exp(2),
                    help='Maximum value for standard deviation.')
parser.add_argument("--curiosity",                      type=str,           default = "none",
                    help='Which kind of curiosity: none, prediction_error, or hidden_state.')  
parser.add_argument("--dkl_max",                        type=float,         default = 1,
                    help='Maximum value for clamping Kullback-Liebler divergence for hidden_state curiosity.')   



    # Vision
parser.add_argument('--vision_scaler',                    type=float,         default = 5, 
                    help='How much to consider vision prediction in accuracy compared to voice and touch.')   
parser.add_argument("--beta_vision",                      type=float,         default = .03,
                    help='Relative importance of complexity for vision.')
parser.add_argument("--prediction_error_eta_vision",      type=float,         default = .3,
                    help='Nonnegative value, how much to consider prediction_error curiosity for vision.')    
parser.add_argument("--hidden_state_eta_vision",          type=float,         default = .3,
                    help='Nonnegative values, how much to consider hidden_state curiosity for vision.') 


    # Touch
parser.add_argument('--touch_scaler',                 type=float,         default = .3, 
                    help='How much to consider touch prediction in accuracy compared to vision and voice.')   
parser.add_argument("--beta_touch",                   type=float,         default = .3,
                    help='Relative importance of complexity for touch.')     
parser.add_argument("--prediction_error_eta_touch",   type=float,         default = .03,
                    help='Nonnegative value, how much to consider prediction_error curiosity for touch.')   
parser.add_argument("--hidden_state_eta_touch",       type=float,         default = .03,
                    help='Nonnegative values, how much to consider hidden_state curiosity for touch.') 



    # Command Voice
parser.add_argument('--command_voice_scaler',            type=float,         default = 3,
                    help='How much to consider command voice prediction in accuracy compared to vision and touch.') 
parser.add_argument("--beta_command_voice",              type=float,         default = .1,
                    help='Relative importance of complexity for voice.')
parser.add_argument("--prediction_error_eta_command_voice", type=float,      default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for voice.')    
parser.add_argument("--hidden_state_eta_command_voice",  type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for voice.') 



    # Report Voice
parser.add_argument('--report_voice_scaler',            type=float,         default = 3, 
                    help='How much to consider report voice prediction in accuracy compared to vision and touch.')     
parser.add_argument("--beta_report_voice",              type=float,         default = .1,
                    help='Relative importance of complexity for voice.')
parser.add_argument("--prediction_error_eta_report_voice", type=float,      default = 1,
                    help='Nonnegative value, how much to consider prediction_error curiosity for voice.')     
parser.add_argument("--hidden_state_eta_report_voice",  type=float,         default = 1.5,
                    help='Nonnegative values, how much to consider hidden_state curiosity for voice.') 
parser.add_argument("--hidden_state_eta_report_voice_reduction_type",  type=str,         default = "None",
                    help='How should interest in report_voice chance?') 



    # Saving data
parser.add_argument('--keep_data',                      type=int,           default = 500,
                    help='How many epochs should pass before keep data.')
parser.add_argument('--temp',                           type=literal,       default = False,
                    help='Should this use data saved temporarily?')      

parser.add_argument('--epochs_per_gen_test',            type=int,           default = 50,
                    help='How many epochs should pass before trying generalization test.')

parser.add_argument('--agents_per_behavior_analysis',   type=int,           default = 1,
                    help='How many agents to save episodes.')

parser.add_argument('--epochs_per_agent_save',          type=int,           default = 25000,
                    help='How many epochs should pass before saving agent model.')
parser.add_argument('--agents_per_agent_save',          type=int,           default = 2,
                    help='How many epochs should pass before saving agent model.')

parser.add_argument('--epochs_per_component_data',      type=int,           default = 2500,
                    help='How many epochs should pass before saving an episode.')
parser.add_argument('--agents_per_component_data',      type=int,           default = 2,
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
    
    
    
# Checking robot parts.
def get_num_sensors(robot_name):
    urdf_path = "pybullet_data/robots/{}.urdf".format(args.robot_name)
    physicsClient = p.connect(p.DIRECT)
    default_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId = physicsClient)
    robot_index = p.loadURDF(urdf_path, (0, 0, 0), default_orn, useFixedBase=False, globalScaling = 1, physicsClientId = physicsClient)
    sensors = []
    for link_index in range(p.getNumJoints(robot_index, physicsClientId = physicsClient)):
        joint_info = p.getJointInfo(robot_index, link_index, physicsClientId = physicsClient)
        link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
        if("sensor" in link_name):
            sensors.append(link_name)
    p.disconnect(physicsClientId = physicsClient)
    num_sensors = len(sensors)
    return(num_sensors, sensors)

    
    
def extend_list_to_match_length(target_list, length, value):
    while len(target_list) < length:
        target_list.append(value)
    return target_list

def update_args(arg_set):
    if(arg_set.comp == "deigo"):
        arg_set.half = False
        
    arg_set.min_joint_1_angle = -arg_set.max_joint_1_angle
    arg_set.wheels_joints_shape = 4
       
    num_sensors, sensors = get_num_sensors(args.robot_name)
    arg_set.touch_state_size = num_sensors
    arg_set.touch_encode_size = num_sensors
    arg_set.touch_shape = num_sensors
    arg_set.sensor_names = sensors
    arg_set.joint_aspects = 4
    
    arg_set.steps_per_epoch = arg_set.max_steps
    arg_set.voice_shape = len(voice_map)
    arg_set.obs_encode_size = arg_set.vision_encode_size + arg_set.touch_encode_size + arg_set.voice_encode_size
    arg_set.h_w_wheels_joints_size = arg_set.pvrnn_mtrnn_size + arg_set.wheels_joints_encode_size
    arg_set.h_w_action_size = arg_set.pvrnn_mtrnn_size + arg_set.wheels_joints_encode_size + arg_set.voice_encode_size
    """arg_set.epochs = [epochs_for_processor[0] for epochs_for_processor in arg_set.epochs_per_processor]
    arg_set.processor_list = [epochs_for_processor[1] for epochs_for_processor in arg_set.epochs_per_processor]"""
    arg_set.right_duration = arg_set.left_duration
    return(arg_set)

for arg_set in [default_args, args]:
    default_args = update_args(default_args) 
    args = update_args(args)
        
args_not_in_title = ["arg_title", "id", "agents", "previous_agents", "init_seed", "keep_data", "epochs_per_pred_list", "episodes_in_pred_list", "agents_per_pred_list", "epochs_per_pos_list", "episodes_in_pos_list", "agents_per_pos_list"]
def get_args_title(default_args, args):
    if(args.arg_title[:3] == "___"): return(args.arg_title)
    name = "" ; first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index("arg_name")))
    for arg in arg_list:
        if(arg in args_not_in_title): pass 
        else: 
            default = getattr(default_args, arg)
            try:
                this_time = getattr(args, arg)
            except:
                this_time = "NONE"
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
os.makedirs(f"{save_file}", exist_ok=True)
os.makedirs(f"{save_file}/thesis_pics", exist_ok=True)
os.makedirs(f"{save_file}/thesis_pics/final", exist_ok=True)

def move_to_bucket(start_address):
    file_name = start_address.split("/")[-1]
    rest_of_address = "/".join(start_address.split("/")[:-1])
    target_address = os.path.join("sftp://theodore-tinker@deigo.oist.jp/bucket/TaniU/Members/ted/" + rest_of_address, file_name)
    print(file_name, rest_of_address, target_address)
    try:
        shutil.move(start_address, target_address)
        print(f"File moved to {target_address} successfully.")
    except PermissionError:
        print("Permission denied. Could not move the file to the target folder.")
    except Exception as e:
        print(f"An error occurred while moving the file: {e}")
    
folder = f"{save_file}/{args.arg_name}"
if(args.arg_title[:3] != "___" and not args.arg_name in ["default", "finishing_dictionaries", "plotting", "plotting_predictions", "plotting_positions"]):
    os.makedirs(f"{folder}", exist_ok=True)
    os.makedirs(f"{folder}/agents", exist_ok=True)
    with open(f"{folder}/agents/args.pickle", "wb") as handle:
        pickle.dump(args, handle)
if(default_args.alpha == "None"): 
    default_args.alpha = None
if(args.alpha == "None"):         
    args.alpha = None

if(args == default_args): print("Using default arguments.")
else:
    for arg in vars(default_args):
        default = getattr(default_args, arg)
        try:
            this_time = getattr(args, arg)
        except:
            this_time = "NONE"
        if(this_time != default):
            print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))
        elif(arg == "device"):
            print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))
            
            
            
if(not args.show_duration):
    def print_duration(start_time, end_time, text = None, end_text = ""):
        pass
     


#%% 



def wait_for_button_press(button_label="Continue"):
    """
    Displays a tkinter button and waits for the user to click it.

    Parameters:
    button_label (str): The label for the button.

    Returns:
    None
    """
    def on_button_click():
        nonlocal continue_simulation
        continue_simulation = True
        root.destroy()

    # Create the tkinter window
    root = tk.Tk()
    root.title("Wait for Input")
    root.geometry("200x100")

    # Add the button
    button = tk.Button(root, text=button_label, command=on_button_click)
    button.pack(expand=True)

    continue_simulation = False

    # Run the tkinter main loop
    root.mainloop()
    
    
    
import tkinter as tk
import torch
import numpy as np

def adjust_action(action_tensor):
    """
    Creates a user interface for adjusting values in 'action_tensor' within [-1, 1].
    Users can see real-time slider values, reset to original values, reset to zero,
    and confirm their final selection.
    """
    root = tk.Tk()
    root.title("Adjust Actions")
    
    # Flatten the tensor and convert to a NumPy array
    shape = action_tensor.shape
    flat_action = action_tensor.view(-1).detach().numpy()
    num_elements = flat_action.size
    
    # Keep track of scales and labels so we can reset/update them
    scales = []
    value_labels = []

    # Store original values so we can reset if needed
    original_values = flat_action.copy()

    def update_value_label(val, label):
        """Update the text of the label to display current slider value."""
        label.config(text=f"{float(val):.2f}")

    def confirm():
        """Close the GUI and allow the function to return the new tensor."""
        root.quit()

    def reset_to_original():
        """Reset all sliders to their original values."""
        for i, scale in enumerate(scales):
            scale.set(original_values[i])
            
    def reset_to_zero():
        """Reset all sliders to zero."""
        for scale in scales:
            scale.set(0.0)

    # Create rows of slider + current value label
    for i in range(num_elements):
        frame = tk.Frame(root, padx=5, pady=5)
        frame.pack(fill=tk.X)
        
        # Label for the slider name
        label = tk.Label(frame, text=f"Action[{i}]")
        label.pack(side=tk.LEFT)
        
        # Current value label on the right
        current_val_label = tk.Label(frame, width=5, anchor='e')
        current_val_label.pack(side=tk.RIGHT)
        
        # Create the scale itself (longer length for easier fine-tuning)
        scale = tk.Scale(
            frame, from_=-1.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=300,
            command=lambda val, lbl=current_val_label: update_value_label(val, lbl)
        )
        scale.set(flat_action[i])
        scale.pack(side=tk.RIGHT, padx=10)
        
        # Initialize the label to the current slider value
        current_val_label.config(text=f"{scale.get():.2f}")
        
        scales.append(scale)
        value_labels.append(current_val_label)

    # Button frame
    btn_frame = tk.Frame(root, pady=10)
    btn_frame.pack()

    # Reset buttons
    reset_orig_btn = tk.Button(btn_frame, text="Reset to Original", command=reset_to_original)
    reset_orig_btn.pack(side=tk.LEFT, padx=5)

    reset_zero_btn = tk.Button(btn_frame, text="Reset to Zero", command=reset_to_zero)
    reset_zero_btn.pack(side=tk.LEFT, padx=5)

    # Confirm button
    confirm_btn = tk.Button(btn_frame, text="Confirm", command=confirm)
    confirm_btn.pack(side=tk.LEFT, padx=5)

    # Start the Tkinter main loop
    root.mainloop()

    # Once the user clicks "Confirm," gather updated values
    updated_values = [scale.get() for scale in scales]
    root.destroy()

    # Convert updated values back to a PyTorch tensor
    return torch.tensor(updated_values).view(shape)



#%%



def wheels_joints_to_string(wheels_joints):
    while(len(wheels_joints.shape) > 1):
        wheels_joints = wheels_joints.squeeze(0)
    print(f"\n\nIN WHEEL_JOINTS_TO_STRING: {wheels_joints}\n\n")
    string = "Left Wheel: {} ".format(round(wheels_joints[0].item(),2))
    string += "Right Wheel: {} ".format(round(wheels_joints[1].item(),2))
    string += "Joint 1: {} ".format(round(wheels_joints[2].item(),2))
    if(len(wheels_joints) == 4):
        string += "Joint 2: {} ".format(round(wheels_joints[3].item(),2))
    return(string)



def plot_number_bars(numbers):
    numbers = [n for n in numbers if n != None]
    fontsize = 7
    plt.figure(figsize=(1.5,1.5))
    plt.bar(range(len(numbers)), numbers, color=['red' if x < 0 else 'blue' for x in numbers])
    
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Index", fontsize = fontsize)
    plt.ylabel("Value", fontsize = fontsize)
    plt.title("Bar Plot of Actions", fontsize = fontsize)
    plt.ylim(-1, 1) 
    xticks = ["left wheel", "right wheel"]
    i = 1
    while(len(xticks) < len(numbers)):
        xticks.append(f"joint {i}")
        i += 1
    plt.xticks(range(len(xticks)), xticks, rotation=45, ha='right', fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.show()



def relative_to(this, min, max):
    this = min + ((this + 1)/2) * (max - min)
    this = [min, max, this]
    this.sort()
    return(this[1])

def opposite_relative_to(this, min, max):
    return ((this - min) / (max - min)) * 2 - 1


    
def calculate_dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)



def rolling_average(lst, window_size=500):
    print(f"\n\nSOMETHING MIGHT BE GOING WRONG HERE! IN ROLLING AVERAGE :{lst}\n\n")
    try:
        new_list = [0 if lst[0] is None else float(lst[0])]
        for i in range(1, len(lst)):
            if lst[i] is None:
                new_list.append(new_list[-1])
            else:
                start_index = max(0, i - window_size + 1)
                window = [x for x in lst[start_index:i+1] if x is not None]
                if window:
                    new_value = sum(window) / len(window)
                else:
                    new_value = 0 
                new_list.append(new_value)
        return new_list
    except:
        print("\n\nYeah, it messed up.\n\n")



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
        if(not key in ["args", "arg_title", "arg_name", "all_task_names", "component_data", "episode_dicts", "agent_lists", "spot_names", "steps", "goal_task", "all_processor_names", "behavior"]):
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
