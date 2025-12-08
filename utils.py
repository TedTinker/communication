#%% 

import os
import pickle
import pybullet as p
from time import sleep
import builtins
import datetime 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import argparse, ast
from math import exp, log, pi
from random import choice, choices
import torch
import psutil
from itertools import product
import tkinter as tk
import numpy as np
import torch

# Find torch device.
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Choose correct folder.
if(os.getcwd().split("/")[-1] != "communication"): 
    os.chdir("communication")
print(f"\n\nWorking in: {os.getcwd()}\n\n")

# Adjusting font in PLT.
font = {'family' : 'sans-serif',
        #'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

# Adjusting printing for computer-cluster.
def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)
    
# For readable printing options.
torch.set_printoptions(precision=3, sci_mode=False)

# Functions to view durations.
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

# Options to view memory. 
def cpu_memory_usage():
    process = psutil.Process(os.getpid())
    mem_usage_bytes = process.memory_info().rss  # rss is the Resident Set Size
    mem_usage_gb = mem_usage_bytes / (1024 ** 3)  # Convert bytes to gigabytes
    print('memory use:', mem_usage_gb, "gigabytes")



#%%



# Class describing task.
class Task:
    def __init__(self, char, name):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
    def __str__(self):
        return(f"{self.char}, {self.name}")
    
# Mapping tasks to digits.
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


        
# Mapping describing color.
class Color:
    def __init__(self, char, name, rgba):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
    def __str__(self):
        return(f"{self.char}, {self.name}")
    
# Mapping colors to digits.
color_map = {
    0: Color("H", "RED",        (1,0,0,1)), 
    1: Color("I", "GREEN",      (0,1,0,1)),
    2: Color("J", "BLUE",       (0,0,1,1)),
    3: Color("K", "CYAN",       (0,1,1,1)), 
    4: Color("L", "MAGENTA",    (1,0,1,1)), 
    5: Color("M", "YELLOW",     (1,1,0,1))} 
max_len_color_name = max([len(c.name) for c in color_map.values()])
color_name_list = [c.name for c in color_map.values()]


        
# Class describing shapes.
class Shape:
    def __init__(self, char, file_name):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        self.name = file_name.split("_")[-1][:-5]
        
    def __str__(self):
        return(f"{self.char}, {self.name}")
    
# Mapping describing shapes.
shape_files = [f.name for f in os.scandir("pybullet_data/shapes") if f.name.endswith("urdf")] 
shape_files.sort()
shape_letter_file = [[f.split("_")[0], f] for f in shape_files]
shape_map = {i : Shape(l, f) for i, (l, f) in enumerate(shape_letter_file)} 
max_len_shape_name = max([len(s.name) for s in shape_map.values()])
shape_name_list = [s.name for s in shape_map.values()]



# In __main__, view all tasks/colors/shapes.
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


        
# Class combining tasks, colors, and shaped. 
# "Parenting" refers to the command voice. If we were using two agents in cooperation, parenting is false. 
class Goal:
    def __init__(self, task, color, shape, parenting):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
        # If the voice is silent, make sure all parts of the voice are silent.
        if(self.task.name == "SILENCE"):
            self.color = self.task 
            self.shape = self.task
        self.one_hots = torch.zeros((3, len(task_map) + len(color_map) + len(shape_map)))
        self.digits = ()
        self.make_texts()
    
    # Make text representing the goal.    
    def make_texts(self):

        for i, char in enumerate([self.task.char, self.color.char, self.shape.char]):
            index = ord(char) - ord('A')
            self.one_hots[i, index] = 1
            
        self.digits = ()
        task_index = ord(self.task.char) - ord('A')
        color_index = ord(self.color.char) - ord('A') - len(task_map)
        shape_index = ord(self.shape.char) - ord('A') - len(task_map) - len(color_map)
        self.digits = (task_index, color_index, shape_index)
            
        self.char_text = f"{self.task.char}{self.color.char}{self.shape.char}"
        self.human_text = f"{self.task.name} {self.color.name} {self.shape.name}"
        
    # Make text easier for humans to read.
    def human_friendly_text(self, command = True):
        return(f"{'Command' if command else 'Feedback'}: {self.human_text}")
        
# Goal representing silence.
empty_goal = Goal(task_map[0], task_map[0], task_map[0], parenting = False)



# Given a one-hot vector, make a goal.
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



# Given (x, y, z) digits, make a goal.
def get_goal_from_digits(digits):
    x, y, z = digits
    one_hots = torch.zeros((3, len(task_map) + len(color_map) + len(shape_map)))
    one_hots[0, x] = 1
    one_hots[1, len(task_map) + y] = 1
    one_hots[2, len(task_map) + len(color_map) + z] = 1
    return(get_goal_from_one_hots(one_hots))


        
# Class describing sensory observations. "prop" is proprioception.
class Obs:
    def __init__(self, vision, touch, prop, command_voice, feedback_voice):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
# Class describing motor commands.
class Action:
    def __init__(self, wheels_joints, voice_out):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
# Class describing transitions to be pushed into the recurrent replay buffer.
class To_Push:
    def __init__(self, obs, action, reward, next_obs, done):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
    def push(self, memory):
        memory.push(
            self.obs.vision.to("cpu"),
            self.obs.touch.to("cpu"),
            self.obs.prop.to("cpu"),
            self.obs.command_voice.to("cpu"),
            self.obs.feedback_voice.to("cpu"),
            self.action.wheels_joints.to("cpu"), 
            self.action.voice_out.to("cpu"),
            self.reward, 
            self.next_obs.vision.to("cpu"),
            self.next_obs.touch.to("cpu"),
            self.next_obs.prop.to("cpu"),
            self.next_obs.command_voice.to("cpu"), 
            self.next_obs.feedback_voice.to("cpu"), 
            self.done)

# Class describing prior, estimated posterior, and the kullback leibler divergence comparing them.
class Inner_States:
    def __init__(self, zp, zq, dkl):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})



# Mapping indexes and characters representing tasks, colors, and shapes.
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



# In __main__, view the one-hot version, character version, and human-friendly version of some example goals.
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



# Here we generate valid combinations in training versus testing generalization.



all_combos = list(product(task_map.keys(), color_map.keys(), shape_map.keys()))

def get_matrix_pattern(a_values, rows=5, cols=6):
    excluded = set()
    for a in a_values:
        for r in range(rows):
            c = (r + a) % cols  
            excluded.add((r, c))
    return [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in excluded]

pattern_lookup_3 = {
    1: set(get_matrix_pattern([0, 1, 2, 3])),
    2: set(get_matrix_pattern([1, 2, 3, 4])),
    3: set(get_matrix_pattern([2, 3, 4, 5])),
    4: set(get_matrix_pattern([3, 4, 5, 6])),
    5: set(get_matrix_pattern([-2, -1, 0, 1])),
    6: set(get_matrix_pattern([-1, 0, 1, 2]))}

def get_training_combos(pattern_lookup):
    training_combos = [(a, c, s) for (a, c, s) in all_combos if 
                        a == 0 or 
                        (a == 1 and (s, c) in pattern_lookup[1]) or
                        (a == 2 and (s, c) in pattern_lookup[2]) or
                        (a == 3 and (s, c) in pattern_lookup[3]) or
                        (a == 4 and (s, c) in pattern_lookup[4]) or
                        (a == 5 and (s, c) in pattern_lookup[5]) or
                        (a == 6 and (s, c) in pattern_lookup[6])]
    return(training_combos)



# 4 tasks, 4 colors, 3 shapes. 48 goals, 16 for training, 32 for testing.
training_combos_1 = [
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 3, 0), (0, 3, 1), (0, 3, 2), 
    (1, 2, 0), (1, 3, 0), (1, 0, 1), (1, 1, 2),
    (4, 0, 0), (4, 1, 1), (4, 3, 1), (4, 2, 2), 
    (5, 1, 0), (5, 2, 1), (5, 0, 2), (5, 3, 2), 
    (6, 1, 0), (6, 2, 0), (6, 3, 1), (6, 0, 2)]
testing_combos_1 = [combo for combo in all_combos if not combo in training_combos_1]

# 5 tasks, 5 colors, 3 shapes. 75 goals, 25 for training, 50 for testing.
training_combos_2 = [
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3), 
    (1, 3, 0), (1, 0, 1), (1, 4, 1), (1, 0, 2), (1, 1, 2), 
    (2, 0, 0), (2, 4, 0), (2, 1, 1), (2, 1, 2), (2, 2, 2), 
    (4, 0, 0), (4, 1, 0), (4, 1, 1), (4, 2, 1), (4, 3, 2), 
    (5, 1, 0), (5, 2, 0), (5, 2, 1), (5, 3, 1), (5, 3, 2), (5, 4, 2),
    (6, 2, 0), (6, 3, 1), (6, 4, 1), (6, 0, 2), (6, 4, 2)]
testing_combos_2 = [combo for combo in all_combos if not combo in training_combos_2]

# 6 tasks, 6 colors, 5 shapes. 180 goals, 60 for training, 120 for testing.
training_combos_3 = get_training_combos(pattern_lookup_3)
testing_combos_3 = [combo for combo in all_combos if not combo in training_combos_3]



# Jun's new request!
# Exceptions replace one goal with a completely different goal.
# His suggestions: 
#   Watch Magenta Pillar -> Push Blue Pole (or maybe just the other object)
#   Be Near Green Pole -> Touch the Top of the Red Dumbbell (or maybe just the other object)

# WE SHOULD ALSO HAVE "EXCEPTIONS" LEAD BACK TO THEMSELVES, SO SHOW A LACK OF U-SHAPE!

exceptions_dict = {
    0 : (                           # None
        [],            
        []),
    
    1 : (
        [(1, 4, 0), (2, 1, 1)],     # Swap Watch Magenta Pillar with Be Near Green Pole
        [(2, 1, 1), (1, 4, 0)]),
    
    3 : (
        [(1, 5, 1), (2, 2, 2)],     # Swap Watch Yellow Pole with Be Near Blue Dumbbell
        [(2, 2, 2), (1, 5, 1)]),
    
    5 : (
        [(1, 4, 0), (2, 1, 1), (1, 3, 4), (2, 2, 2)],     # Both of Those
        [(2, 1, 1), (1, 4, 0), (2, 2, 2), (1, 3, 4)]),
    
    7 : (
        [(3, 1, 1), (4, 3, 2)],     # Swap Touch the Top Green Pole with Push Forawrd Cyan Dumbbell
        [(4, 3, 2), (3, 1, 1)]),
    
    9 : (
        [(1, 4, 0), (2, 1, 1), (3, 5, 4), (4, 3, 2)],       # Swap Watch Magenta Pillar with Be Near Green Pole
        [(2, 1, 1), (1, 4, 0), (4, 3, 2), (3, 5, 4)]),      # Swap Touch the Top Green Pole with Push Forawrd Cyan Dumbbell
    
    11 : (
        [(1, 4, 0), (4, 3, 2)],     # Swap Watch Magenta Pillar with Push Forward Cyan Dumbbell
        [(4, 3, 2), (1, 4, 0)]),    
    
    13 : (
        [(3, 1, 1), (2, 2, 2)],     # Swap Be Near Blue Dumbbell with Touch the Top Green Pole
        [(2, 2, 2), (3, 1, 1)]),    
    
    15 : (
        [(1, 4, 0), (4, 5, 4), (3, 1, 1), (2, 2, 2)],     # Both of Those
        [(4, 5, 4), (1, 4, 0), (2, 2, 2), (3, 1, 1)]),
    
    
}



def add_control_exceptions(exceptions_dict):
    """
    For every odd-numbered key in exceptions_dict, add an even-numbered key
    that uses the same list for exception and correct goal.
    """
    new_dict = exceptions_dict.copy()
    for k in list(exceptions_dict.keys()):
        if k % 2 == 1:  # it's odd
            red = exceptions_dict[k][0]
            control_key = k + 1
            new_dict[control_key] = (red, red)
    return new_dict

exceptions_dict = add_control_exceptions(exceptions_dict)



# In __main__, view plots showing training and testing combinations.
if(__name__ == "__main__"):
    def plot_combined_training_grid(training_combos, exception_num, title="Training Set"):
        task_items = [(a, t) for a, t in task_map.items() if t.name != "SILENCE"]
        num_tasks = len(task_items)
        num_cols = 3
        num_rows = (num_tasks + num_cols - 1) // num_cols

        fig = plt.figure(figsize=(22, 12))
        fig.suptitle(title, fontsize=28)
        outer_grid = gridspec.GridSpec(num_rows, num_cols, wspace=0.5, hspace=0.5)

        # Map each (a, c, s) combo to its Axes so we can connect across subplots later
        ax_map = {}

        for i, (a, task) in enumerate(task_items):
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                len(shape_map), len(color_map),
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0
            )

            for s in range(len(shape_map)):
                for c in range(len(color_map)):
                    ax = fig.add_subplot(inner_grid[s, c])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                    combo = (a, c, s)
                    ax_map[combo] = ax  # remember where this combo lives

                    # Base cell
                    if combo in training_combos:
                        ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='gray', alpha=0.5))
                    else:
                        ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black'))

                    # Red exception (centered at (0.5, 0.5))
                    if combo in exceptions_dict[exception_num][0]:
                        ax.add_patch(patches.Rectangle((.25, .25), .5, .5, facecolor='red', alpha = .5, edgecolor='red'))

                    # Blue exception (also centered at (0.5, 0.5))
                    if combo in exceptions_dict[exception_num][1]:
                        ax.add_patch(patches.Rectangle((.375, .375), .25, .25, facecolor='blue', alpha = .3, edgecolor='blue'))

                    # Label
                    ax.text(0.5, 0.5, f"{color_map[c].name}\n{shape_map[s].name}",
                            va='center', ha='center', fontsize=9, wrap=True)

            # Task title in the middle column
            center_col = len(color_map) // 2
            title_ax = fig.add_subplot(inner_grid[0, center_col])
            title_ax.set_title(task.name, fontsize=14, pad=12)
            title_ax.axis('off')

        # Make sure layout is finalized before drawing connectors
        fig.canvas.draw()

        # Draw arrows: from each red combo to the blue combo at the same index
        for start_combo, end_combo in zip(exceptions_dict[exception_num][0], exceptions_dict[exception_num][1]):
            start_ax = ax_map.get(start_combo)
            end_ax   = ax_map.get(end_combo)
            if start_ax is None or end_ax is None:
                continue

            # Both colored squares are centered at (0.5, 0.5) in their own axes
            con = ConnectionPatch(
                xyA=(0.5, 0.5), 
                coordsA=start_ax.transData,   # start (red)
                xyB=(0.5, 0.5), 
                coordsB=end_ax.transData,     # end (blue)
                arrowstyle="-|>", 
                ls = "-",
                mutation_scale=25, 
                lw=1.8, 
                color="black",
                shrinkA=10, 
                shrinkB=10  # keep arrowheads off the colored squares
            )
            con.set_zorder(1000)
            con.set_clip_on(False)  # don't let axes clip the arrow
            fig.add_artist(con)

        plt.show()
        plt.close()
    
    #plot_combined_training_grid(training_combos_1, title="Training Set 1 – All Tasks")
    #plot_combined_training_grid(training_combos_2, title="Training Set 2 – All Tasks")
    
    for key in exceptions_dict.keys():
        #if(key % 2 != 0):
            plot_combined_training_grid(training_combos_3, title=f"Training Set 3 - All Tasks - Exceptions {key}", exception_num = key)
            
        
        
#%%



# These functions can make goals given which tasks, colors, and shapes are allowed.



# Choose color and shape, given tasks/colors/shapes in use.
def valid_color_shape(task_num, other_shape_colors, allowed_colors, allowed_shapes, test_train_num = 3, test = False):
    training_combos = training_combos_1 if test_train_num == 1 else training_combos_2 if test_train_num == 2 else training_combos_3 if test_train_num == 3 else training_combos_4
    testing_combos = [combo for combo in all_combos if not combo in training_combos]
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

# Return goals, given allowed tasks, colors, and shapes. 
# This returns two sets of colors and shapes. If using two robots in cooperation, the second set is for the second robot.
def make_objects_and_task(num_objects, allowed_tasks_and_weights, allowed_colors, allowed_shapes, test_train_num = 3, test = False):
    tasks   = [v for v, w in allowed_tasks_and_weights]
    weights = [w for v, w in allowed_tasks_and_weights]
    task_num = choices(tasks, weights=weights, k=1)[0]
    
    goal_object = valid_color_shape(task_num, [], allowed_colors, allowed_shapes, test_train_num, test = test)
    colors_shapes_1 = [goal_object]
    colors_shapes_2 = [goal_object]
    for n in range(num_objects-1):
        colors_shapes_1.append(valid_color_shape(task_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test_train_num, test = test))
    for n in range(num_objects-1):
        colors_shapes_2.append(valid_color_shape(task_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test_train_num, test = test))
    
    task = task_map[task_num]
    colors_shapes_1 = [(color_map[color_index], shape_map[shape_index]) for color_index, shape_index in colors_shapes_1]
    colors_shapes_2 = [(color_map[color_index], shape_map[shape_index]) for color_index, shape_index in colors_shapes_2]
    return(task, colors_shapes_1, colors_shapes_2)


        
# In __main__, view some example goals.
if(__name__ == "__main__"):
    print("Train")
    for i in range(1):
        task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(2, [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
        print(task.name, [(color.name, shape.name) for color, shape in colors_shapes_1], [(color.name, shape.name) for color, shape in colors_shapes_2])
    print("\nTest")
    for i in range(1):
        task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(2, [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4], test = True)
        print(task.name, [(color.name, shape.name) for color, shape in colors_shapes_1], [(color.name, shape.name) for color, shape in colors_shapes_2])
        
        
        
        
#%% 



# Type for booleons in arguments.
def literal(arg_string): 
    return(ast.literal_eval(arg_string))


# Arguments to parse. 
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
parser.add_argument("--init_seed",                      type=int,         default = 777,
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
parser.add_argument('--load_agents',                    type=literal,       default = False,
                    help='Are we loading agents?')      

    

    # Simulation details
parser.add_argument('--time_step',                      type=float,         default = .005,
                    help='Length of step in pybullet environment.')
parser.add_argument('--steps_per_step',                 type=int,           default = 20,
                    help='Agent-steps for each action.')
parser.add_argument('--numSolverIterations',            type=int,           default = 1,
                    help='Precision of steps in pybullet environment.')
parser.add_argument('--numSubSteps',                    type=int,           default = 1,
                    help='numSubSteps in pybullet environment.')
parser.add_argument('--force',                          type=float,         default = 30000,
                    help='Force for moving joints.') 
parser.add_argument('--gravity',                        type=float,         default = -9.8,
                    help='Force of gravity.') 



    # Which tasks/colors/shapes are allowed in this test_train_num?
parser.add_argument('--watch',                          type=literal,       default = True,
                    help='Allow watch task?')
parser.add_argument('--be_near',                        type=literal,       default = True,
                    help='Allow be_near task?')
parser.add_argument('--touch_top',                      type=literal,       default = True,
                    help='Allow touch_top task?')
parser.add_argument('--push_forward',                   type=literal,       default = True,
                    help='Allow push_forward task?')
parser.add_argument('--push_left',                      type=literal,       default = True,
                    help='Allow push_left task?')
parser.add_argument('--push_right',                     type=literal,       default = True,
                    help='Allow push_right task?')

parser.add_argument('--red',                            type=literal,       default = True,
                    help='Allow red color?')
parser.add_argument('--green',                          type=literal,       default = True,
                    help='Allow green color?')
parser.add_argument('--blue',                           type=literal,       default = True,
                    help='Allow blue color?')
parser.add_argument('--cyan',                           type=literal,       default = True,
                    help='Allow cyan color?')
parser.add_argument('--magenta',                        type=literal,       default = True,
                    help='Allow magenta color?')
parser.add_argument('--yellow',                         type=literal,       default = True,
                    help='Allow yellow color?')

parser.add_argument('--pillar',                         type=literal,       default = True,
                    help='Allow pillar shape?')
parser.add_argument('--pole',                           type=literal,       default = True,
                    help='Allow pole shape?')
parser.add_argument('--dumbbell',                       type=literal,       default = True,
                    help='Allow dumbbell shape?')
parser.add_argument('--cone',                           type=literal,       default = True,
                    help='Allow cone shape?')
parser.add_argument('--hourglass',                      type=literal,       default = True,
                    help='Allow hourglass shape?')



    # Agent details
parser.add_argument('--robot_name',                     type=str,           default = "robot",
                    help='Name of the robot\'s urdf file.')  
parser.add_argument('--body_size',                      type=float,         default = 2,
                    help='How large is the agent\'s body?')  
parser.add_argument('--image_size',                     type=int,           default = 16, 
                    help='Dimensions of the images observed.')
parser.add_argument('--max_wheel_speed',                type=float,         default = 10,
                    help='Max wheel speed.')
parser.add_argument('--angular_scaler',                 type=float,         default = .4,
                    help='How to scale angular velocity vs linear velocity.')
parser.add_argument('--max_joint_speed',                type=float,         default = 8,
                    help='Max joint speed.')
parser.add_argument('--max_joint_1_angle',              type=float,         default = pi/6,
                    help='Max yaw angle.')
parser.add_argument('--min_joint_2_angle',              type=float,         default = -pi/2,
                    help='Min pitch angle.')
parser.add_argument('--max_joint_2_angle',              type=float,         default = 0,
                    help='Max pitch angle.')



    # Arena/Processor details
parser.add_argument('--processor',                      type=str,       default = "all",
                help='List of processors. Agent trains on each processor based on epochs in epochs parameter.')
parser.add_argument('--min_object_distance',            type=float,         default = 4,
                    help='How far objects can start from the agent.')
parser.add_argument('--max_object_distance',            type=float,         default = 8,
                    help='How far objects can start from the agent.')
parser.add_argument('--min_object_angle',               type=float,         default = pi/2,
                    help='How far objects must be from one another.')
parser.add_argument('--object_size',                    type=float,         default = 2.5,
                    help='How large are objects?')          

parser.add_argument('--reward',                         type=float,         default = 10,
                    help='Extrinsic reward for choosing correct task, shape, and color.') 
parser.add_argument('--wrong_object_punishment',        type=float,         default = 0,
                    help='Negative reward for punishing doing anything to the wrong object (except watching).') 
parser.add_argument("--hidden_state_eta_feedback_voice_reduction_type",  type=str,         default = "None",
                    help='How should interest in feedback_voice chance?') 
parser.add_argument('--reward_inflation_type',          type=str,           default = "None",
                    help='How should reward increase?')   
parser.add_argument('--tanh_touch',                     type=literal,       default = True,
                    help='Do sensors measure contact with Tanh?')

parser.add_argument('--max_steps',                      type=int,           default = 30,     
                    help='How many steps the agent can make in one episode.')
parser.add_argument('--step_lim_punishment',            type=float,         default = 0,
                    help='Extrinsic punishment for taking max_steps steps.')
parser.add_argument('--step_cost',                      type=float,         default = .99,    
                    help='How much extrinsic rewards are reduced per step.')
parser.add_argument('--max_voice_len',                  type=int,           default = 3,
                    help='Maximum length of voice.')



    # Task details.
parser.add_argument('--watch_duration',                 type=int,           default = 6,
                    help='How long the agent must watch the object to achieve watching.')
parser.add_argument('--pointing_at_object_for_watch',   type=float,         default = pi/12,
                    help='How directly the agent must point to the object to achieve watching.')
parser.add_argument('--watch_distance',                 type=float,         default = 12,
                    help='How closely the agent must watch the object to achieve watching.')

parser.add_argument('--be_near_duration',               type=int,           default = 5,
                    help='How long the agent must be near the object to achieve be_near.')
parser.add_argument('--pointing_at_object_for_being_near',  type=float,     default = pi/6,
                    help='How directly the agent must point to the object to achieve be_near.')
parser.add_argument('--be_near_distance',               type=float,         default = 6,
                    help='How close the agent must be near the object to achieve be_near.')

parser.add_argument('--top_duration',                   type=int,           default = 3,   
                    help='How long the agent must touch the top of the object to achieve touch_top.')
parser.add_argument('--pointing_at_object_for_touch_top',  type=float,      default = pi/3,
                    help='How directly the agent must point to the object to achieve touch top.')
parser.add_argument('--touch_top_min_height',           type=float,         default = 3.75,
                    help='How elevated the agent\'s arm must be to touch the object from above.')

parser.add_argument('--push_duration',                  type=int,           default = 3,
                    help='How long the agent must push the object to achieve push_forward.')
parser.add_argument('--pointing_at_object_for_push',    type=float,         default = pi/12,
                    help='How directly the agent must point to the object to achieve push_forward.')
parser.add_argument('--global_push_amount',             type=float,         default = .1,
                    help='Needed distance of an object\'s movement for push_forward')
parser.add_argument('--local_push_limit',               type=float,         default = .3,
                    help='Prevent bogus pushing by requiring local stillness.')

parser.add_argument('--left_right_duration',            type=int,           default = 3,   
                    help='How long the agent must push the object to achieve push_left or push_right.')
parser.add_argument('--pointing_at_object_for_left_right', type=float,      default = pi/3,
                    help='How directly the agent must point to the object to achieve push_left or push_right.')
parser.add_argument('--global_left_right_amount',       type=float,         default = .2,
                    help='Needed distance of an object\'s movement for push_left or push_right.')
parser.add_argument('--local_left_right_amount',        type=float,         default = .25,
                    help='Prevent bogus pushing by requiring local movement.')
parser.add_argument('--max_wheel_speed_for_left_right', type=float,         default = 5,
                    help='How fast the agent\'s wheels may move for push_left or push_right.')
parser.add_argument('--min_arm_speed_for_left_right',   type=float,         default = .01,
                    help='How fast the agent\'s arm must move for push_left or push_right.')

parser.add_argument('--exceptions',                     type=literal,       default = 0,
                    help='Add exceptions to goals?')



    # Model architecture
parser.add_argument('--hidden_size',                    type=int,           default = 64,
                    help='Parameters in hidden layers.')   
parser.add_argument('--pvrnn_mtrnn_size',               type=int,           default = 256,
                    help='Parameters in hidden layers of PVRNN\'s mtrnn.')   

parser.add_argument('--wheels_joints_encode_size',      type=int,           default = 8,
                    help='Parameters in encoding wheels_joints.')   
parser.add_argument('--touch_encode_size',              type=int,           default = 20,
                    help='Parameters in encoding image.')  
parser.add_argument('--touch_state_size',               type=int,           default = 20,
                    help='Parameters in prior and posterior inner-states.')

parser.add_argument('--vision_encode_size',             type=int,           default = 128,
                    help='Parameters in encoding image.')   
parser.add_argument('--vision_state_size',              type=int,           default = 128,
                    help='Parameters in prior and posterior inner-states.')

parser.add_argument('--prop_encode_size',               type=int,           default = 4,
                    help='Parameters in encoding image.')  
parser.add_argument('--prop_state_size',                type=int,           default = 4,
                    help='Parameters in prior and posterior inner-states.')

parser.add_argument('--char_encode_size',               type=int,           default = 8,
                    help='Parameters in encoding.')   
parser.add_argument('--voice_encode_size',              type=int,           default = 256,
                    help='Parameters in encoding voice.')   
parser.add_argument('--voice_state_size',               type=int,           default = 256,
                    help='Parameters in prior and posterior inner-states.')

parser.add_argument('--dropout',                        type=float,         default = .001,
                    help='Dropout percentage.')
parser.add_argument('--divisions',                      type=int,           default = 2,
                    help='How many times should RBGD_Out double size to image-size?')
parser.add_argument('--half',                           type=literal,       default = True,
                    help='Should the models use float16 instead of float32?')   



    # Training
parser.add_argument('--epochs',                         type=int,           default = 60000,
                    help='List of processors. Agent trains on each processor based on epochs in epochs parameter.')
parser.add_argument('--test_train_num',                 type=int,           default = 3,
                    help='Which collects of tasks/colors/shapes are used?')
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
parser.add_argument("--normal_alpha",                   type=float,         default = 0,
                    help='Nonnegative value, how much to consider policy prior.') 
parser.add_argument("--alpha",                          type=literal,       default = 0,
                    help='Nonnegative value, how much to consider entropy. Set to None to use target_entropy.')        
parser.add_argument("--target_entropy",                 type=float,         default = 0,
                    help='Target for choosing alpha if alpha set to None. Recommended: negative size of action-space.')      
parser.add_argument("--alpha_text",                     type=literal,       default = 0,
                    help='Nonnegative value, how much to consider entropy regarding agent voice. Set to None to use target_entropy_text.')        
parser.add_argument("--target_entropy_text",            type=float,         default = 0,
                    help='Target for choosing alpha_text if alpha_text set to None. Recommended: negative size of voice_out-space.')     



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
parser.add_argument('--vision_scaler',                  type=float,         default = 5, 
                    help='How much to consider vision prediction in accuracy compared to voice and touch.')   
parser.add_argument("--beta_vision",                    type=float,         default = .03,
                    help='Relative importance of complexity for vision.')
parser.add_argument("--prediction_error_eta_vision",    type=float,         default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for vision.')    
parser.add_argument("--hidden_state_eta_vision",        type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for vision.') 



    # Touch
parser.add_argument('--touch_scaler',                   type=float,         default = .3, 
                    help='How much to consider touch prediction in accuracy compared to vision and voice.')   
parser.add_argument("--beta_touch",                     type=float,         default = .3,
                    help='Relative importance of complexity for touch.')     
parser.add_argument("--prediction_error_eta_touch",     type=float,         default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for touch.')   
parser.add_argument("--hidden_state_eta_touch",         type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for touch.') 



    # Proprioception
parser.add_argument('--prop_scaler',                    type=float,         default = .01, 
                    help='How much to consider proprioception prediction in accuracy compared to vision and voice.')   
parser.add_argument("--beta_prop",                      type=float,         default = .3,
                    help='Relative importance of complexity for proprioception.')     
parser.add_argument("--prediction_error_eta_prop",      type=float,         default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for proprioception.')   
parser.add_argument("--hidden_state_eta_prop",          type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for proprioception.') 



    # Command Voice
parser.add_argument('--command_voice_scaler',            type=float,         default = 3,
                    help='How much to consider command voice prediction in accuracy compared to vision and touch.') 
parser.add_argument("--beta_command_voice",              type=float,         default = .1,
                    help='Relative importance of complexity for voice.')
parser.add_argument("--prediction_error_eta_command_voice", type=float,      default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for voice.')    
parser.add_argument("--hidden_state_eta_command_voice",  type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for voice.') 



    # Feedback Voice
parser.add_argument('--feedback_voice_scaler',            type=float,         default = 3, 
                    help='How much to consider feedback voice prediction in accuracy compared to vision and touch.')     
parser.add_argument("--beta_feedback_voice",              type=float,         default = .1,
                    help='Relative importance of complexity for voice.')
parser.add_argument("--prediction_error_eta_feedback_voice", type=float,      default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for voice.')     
parser.add_argument("--hidden_state_eta_feedback_voice",  type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for voice.') 



    # Saving data
parser.add_argument('--keep_data',                      type=int,           default = 500,
                    help='How many epochs should pass before keeping data.')
parser.add_argument('--temp',                           type=literal,       default = False,
                    help='Should this use data saved temporarily?')      
parser.add_argument('--agents_for_plotting',            type=int,           default = 9999,
                    help='How many agents should be used in plotting?')      

parser.add_argument('--epochs_per_gen_test',            type=int,           default = 50,
                    help='How many epochs should pass before trying generalization test.')

parser.add_argument('--save_agents',                    type=literal,       default = True,
                    help='Do you save agents?')
parser.add_argument('--epochs_per_agent_save',          type=int,           default = 10000,
                    help='How many epochs should pass before saving agent model.')
parser.add_argument('--agents_per_agent_save',          type=int,           default = 2,
                    help='How many epochs should pass before saving agent model.')

parser.add_argument('--save_behaviors',                 type=literal,       default = True,
                    help='How many agents to save episodes.')
parser.add_argument('--episodes_per_behavior_analysis', type=int,           default = 10,
                    help='How many agents to save episodes.')
parser.add_argument('--agents_per_behavior_analysis',   type=int,           default = 1,
                    help='How many agents to save episodes.')

parser.add_argument('--save_compositions',              type=literal,       default = True,
                    help='How many agents to save episodes.')
parser.add_argument('--epochs_per_composition_data',    type=int,           default = 2500,
                    help='How many epochs should pass before saving an episode.')
parser.add_argument('--agents_per_composition_data',    type=int,           default = 2,
                    help='How many agents to save episodes.')



# Make arguments.
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



# Based on arguments, adjust other arguments.
def update_args(arg_set):
    if(arg_set.comp == "deigo"):
        arg_set.half = False
        
    arg_set.min_joint_1_angle = -arg_set.max_joint_1_angle
    arg_set.wheels_joints_shape = 4
       
    num_sensors, sensors = get_num_sensors(args.robot_name)
    arg_set.touch_shape = num_sensors
    arg_set.sensor_names = sensors
    arg_set.joint_aspects = 4
    
    arg_set.steps_per_epoch = arg_set.max_steps
    arg_set.voice_shape = len(voice_map)
    arg_set.obs_encode_size = arg_set.vision_encode_size + arg_set.touch_encode_size + arg_set.voice_encode_size
    arg_set.h_w_wheels_joints_size = arg_set.pvrnn_mtrnn_size + arg_set.wheels_joints_encode_size
    arg_set.h_w_action_size = arg_set.pvrnn_mtrnn_size + arg_set.wheels_joints_encode_size + arg_set.voice_encode_size
    
    allowed_task_dict = {
        1 : arg_set.watch,
        2 : arg_set.be_near,
        3 : arg_set.touch_top,
        4 : arg_set.push_forward,
        5 : arg_set.push_left,
        6 : arg_set.push_right}
    arg_set.allowed_tasks = [key for key, value in allowed_task_dict.items() if value]
    
    allowed_color_dict = {
        0 : arg_set.red,
        1 : arg_set.green,
        2 : arg_set.blue,
        3 : arg_set.cyan,
        4 : arg_set.magenta,
        5 : arg_set.yellow}
    arg_set.allowed_colors = [key for key, value in allowed_color_dict.items() if value]
    
    allowed_shape_dict = {
        0 : arg_set.pillar,
        1 : arg_set.pole,
        2 : arg_set.dumbbell,
        3 : arg_set.cone,
        4 : arg_set.hourglass}
    arg_set.allowed_shapes = [key for key, value in allowed_shape_dict.items() if value]

    return(arg_set)

for arg_set in [default_args, args]:
    default_args = update_args(default_args) 
    args = update_args(args)
        
# Make a title for these arguments based on comparing it to the default arguments, without including these parameters.
args_not_in_title = [
    "arg_title", "id", "agents", "previous_agents", "init_seed", "keep_data", "epochs_per_pred_list", 
    "episodes_in_pred_list", "agents_per_pred_list", "epochs_per_pos_list", "episodes_in_pos_list", "agents_per_pos_list",
    "watch", "be_near", "touch_top", "push_forward", "push_left", "push_right", "red", "green", "blue", "cyan", "magenta", "yellow", "pillar", "pole", "dumbbell", "cone", "hourglass"]

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

# Generate some folders for saving agents and plots.
save_file = f"saved_{args.comp}"
os.makedirs(f"{save_file}", exist_ok=True)
os.makedirs(f"{save_file}/thesis_pics", exist_ok=True)
os.makedirs(f"{save_file}/thesis_pics/final", exist_ok=True)
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

# Print information about arguments.
if(args == default_args): 
    print("Using default arguments.")
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
            
            
            
# If we are not showing durations, remove influence of this function.
if(not args.show_duration):
    def print_duration(start_time, end_time, text = None, end_text = ""):
        pass
     


#%% 



# Buttons are used in some tkinter GUIs.
def wait_for_button_press(button_label="Continue"):
    def on_button_click():
        nonlocal continue_simulation
        continue_simulation = True
        root.destroy()

    root = tk.Tk()
    root.title("Wait for Input")
    root.geometry("200x100")
    button = tk.Button(root, text=button_label, command=on_button_click)
    button.pack(expand=True)
    continue_simulation = False
    root.mainloop()
    
    

# GUI for users to input custom motor commands.
def adjust_action(action_tensor):
    root = tk.Tk()
    root.title("Adjust Actions")
    shape = action_tensor.shape
    flat_action = action_tensor.view(-1).detach().numpy()
    num_elements = flat_action.size
    scales = []
    value_labels = []
    original_values = flat_action.copy()

    def update_value_label(val, label):
        label.config(text=f"{float(val):.2f}")

    def confirm():
        root.quit()

    def reset_to_original():
        for i, scale in enumerate(scales):
            scale.set(original_values[i])
            
    def reset_to_zero():
        for scale in scales:
            scale.set(0.0)

    for i in range(num_elements):
        frame = tk.Frame(root, padx=5, pady=5)
        frame.pack(fill=tk.X)
        label = tk.Label(frame, text=f"Action[{i}]")
        label.pack(side=tk.LEFT)
        current_val_label = tk.Label(frame, width=5, anchor='e')
        current_val_label.pack(side=tk.RIGHT)
        scale = tk.Scale(
            frame, from_=-1.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=300,
            command=lambda val, lbl=current_val_label: update_value_label(val, lbl))
        scale.set(flat_action[i])
        scale.pack(side=tk.RIGHT, padx=10)
        current_val_label.config(text=f"{scale.get():.2f}")
        scales.append(scale)
        value_labels.append(current_val_label)

    btn_frame = tk.Frame(root, pady=10)
    btn_frame.pack()
    reset_orig_btn = tk.Button(btn_frame, text="Reset to Original", command=reset_to_original)
    reset_orig_btn.pack(side=tk.LEFT, padx=5)
    reset_zero_btn = tk.Button(btn_frame, text="Reset to Zero", command=reset_to_zero)
    reset_zero_btn.pack(side=tk.LEFT, padx=5)
    confirm_btn = tk.Button(btn_frame, text="Confirm", command=confirm)
    confirm_btn.pack(side=tk.LEFT, padx=5)
    root.mainloop()
    updated_values = [scale.get() for scale in scales]
    root.destroy()
    return torch.tensor(updated_values).view(shape)



#%%



# Make human-readable text describing robot's wheels and joints.
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



# Make human-readable plot of robot motor commands.
def plot_number_bars(numbers):
    numbers = [n for n in numbers if n != None]
    fontsize = 7
    plt.figure(figsize=(1.5,1.5))
    plt.bar(range(len(numbers)), numbers, color=['red' if x < 0 else 'blue' for x in numbers])
    
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Index", fontsize = fontsize)
    plt.ylabel("Value", fontsize = fontsize)
    plt.title("Bar Plot of Motor Commands", fontsize = fontsize)
    plt.ylim(-1, 1) 
    xticks = ["left wheel", "right wheel"]
    i = 1
    while(len(xticks) < len(numbers)):
        xticks.append(f"joint {i}")
        i += 1
    plt.xticks(range(len(xticks)), xticks, rotation=45, ha='right', fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.show()



# Given minimum and maximum, find proportional value of "this" in [-1, 1].
def relative_to(this, min, max):
    this = min + ((this + 1)/2) * (max - min)
    this = [min, max, this]
    this.sort()
    return(this[1])

# Do the reverse.
def opposite_relative_to(this, min, max):
    return ((this - min) / (max - min)) * 2 - 1


    
# Calculate Kullback-Leibler divergence.
def calculate_dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)



# Find rolling average.
def rolling_average(lst, window_size=500):
    # print(f"\nSometimes this may result in error. In rolling average :{lst}\n")
    print("Rolling...", end = " ")
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
        print("\n\nRolling average failed.\n\n")



# Load dictionaries for plotting robot data.
def load_dicts(args):
    if(os.getcwd().split("/")[-1] != save_file): os.chdir(save_file)
    plot_dicts = [] ; min_max_dicts = []
        
    if(type(args) == dict):
        complete_order = args["titles"]
    else:
        complete_order = args.arg_title[3:-3].split("+")
    order = [o for o in complete_order if not o in ["empty_space", "break"]]
                
    for name in order:
        print(f"Loading dictionaries for {name}...")
        got_plot_dicts = False ; got_min_max_dicts = False
        while(not got_plot_dicts):
            with open(name + "/" + "plot_dict.pickle", "rb") as handle: 
                plot_dicts.append(pickle.load(handle)) ; got_plot_dicts = True
        while(not got_min_max_dicts):
            try:
                with open(name + "/" + "min_max_dict.pickle", "rb") as handle: 
                    min_max_dicts.append(pickle.load(handle)) ; got_min_max_dicts = True 
            except: 
                print("Stuck trying to get {}'s min_max_dicts...".format(name)) ; sleep(1)
    print("Loaded all dicts! Making min/max dict...")
    
    min_max_dict = {}
    for key in plot_dicts[0].keys():
        if(not key in ["args", "arg_title", "arg_name", "all_task_names", "composition_data", "component_data", "episode_dicts", "agent_lists", "spot_names", "steps", "goal_task", "all_processor_names", "behavior"]):
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
                if(plot_dict["args"].arg_name == arg_name or plot_dict["args"].arg_name + "_old" == arg_name):    
                    final_complete_order.append(arg_name) 
                    final_plot_dicts.append(plot_dict)
                    
    while(len(final_complete_order) > 0 and final_complete_order[0] in ["break", "empty_space"]): 
        final_complete_order.pop(0)    
        
    print("Done with Load Dicts!")          
    
    return(final_plot_dicts, min_max_dict, complete_order)
# %%
