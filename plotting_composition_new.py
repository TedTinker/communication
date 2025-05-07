#%%

# Now make plot_by_epoch_vs_epoch:
#   Input component to get data_dict and recorder_dict.
#   Input data_epochs, recorder_epoch.
#   If data_epochs is an integert, plot that.
#   If data_epochs is a list, plot the first smoothly shifting to the last.


import os
import random
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import torch
from torch import nn 
import torch.optim as optim
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from collections import defaultdict

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from utils import testing_combos, args, duration, load_dicts, print, task_map, color_map, shape_map
from utils_submodule import  init_weights


print("name:\n{}\n".format(args.arg_name),)



# Make tools for plotting task/color/shape.
task_mapping_color = {
    'WATCH':            '#FF0000',  # Red
    'BE NEAR':          '#00FF00',  # Green
    'TOUCH THE TOP':    '#0000FF',  # Blue
    'PUSH FORWARD':     '#00DDDD',  # Cyan
    'PUSH LEFT':        '#FF00FF',  # Magenta
    'PUSH RIGHT':       '#DDDD00'}  # Yellow

task_mapping_letter = {
    'WATCH':            "W",
    'BE NEAR':          "B",
    'TOUCH THE TOP':    "T",
    'PUSH FORWARD':     "F",
    'PUSH LEFT':        "L",
    'PUSH RIGHT':       "R"}

color_mapping_color = {
    'RED':              '#FF0000',           
    'GREEN':            '#00FF00',
    'BLUE':             '#0000FF',
    'CYAN':             '#00DDDD',
    'MAGENTA':          '#FF00FF',
    'YELLOW':           '#DDDD00'}

def darken_hex_color(hex_color, factor=0.8):
    """
    Darkens the given hex color by the specified factor.
    Factor should be between 0 (black) and 1 (no change).
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    return f'#{r:02X}{g:02X}{b:02X}'

color_mapping_color_dark = {name: darken_hex_color(hex_code) for name, hex_code in color_mapping_color.items()}

shape_mapping_color = {
    'PILLAR':           '#FF0000',  # Red
    'POLE':             '#00FF00',  # Green
    'DUMBBELL':         '#0000FF',  # Blue
    'CONE':             '#00DDDD',  # Cyan
    'HOURGLASS':        '#FF00FF'}  # Magenta

shape_mapping_marker = {
    'PILLAR':           mpimg.imread('pybullet_data/shapes/pillar.png'),   
    'POLE':             mpimg.imread('pybullet_data/shapes/pole.png'),   
    'DUMBBELL':         mpimg.imread('pybullet_data/shapes/dumbbell.png'),    
    'CONE':             mpimg.imread('pybullet_data/shapes/cone.png'),   
    'HOURGLASS':        mpimg.imread('pybullet_data/shapes/hourglass.png')}   

dpi = 400  
letter_size = 120
letter_w_size = 200
shape_size = .6
test_size = 120
color_size = 120
shape_size = .4
fontsize = 10

    
    
def colorize_marker_image(marker_img, hex_color, alpha = .3):
    hex_color = hex_color.lstrip('#')
    target_rgb = np.array([
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0])
    colorized = marker_img.copy()
    mask = colorized[..., 3] > 0 
    colorized[mask, 0] = target_rgb[0]
    colorized[mask, 1] = target_rgb[1]
    colorized[mask, 2] = target_rgb[2]
    colorized[mask, 3] = alpha
    return colorized

shape_mapping_colored_marker = {}
for shape_name, shape_marker in shape_mapping_marker.items():
    shape_mapping_colored_marker[shape_name] = {}
    shape_mapping_colored_marker[shape_name]["BLACK"] = colorize_marker_image(shape_marker, "#000000")
    for color_name, color_color in color_mapping_color.items():
        shape_mapping_colored_marker[shape_name][color_name] = colorize_marker_image(shape_marker, color_color)



def print_dict_keys(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('    ' * indent + str(key))
            print_dict_keys(value, indent + 1)
        elif hasattr(value, 'shape'):
            print('    ' * indent + f"{key} : shape {value.shape}")
        else:
            print('    ' * indent + str(key))



# Collect all data.
meta_data_dict = {}

def get_all_data(plot_dict, component):
    args = plot_dict["args"]
    print(f"Getting {args.arg_name}'s {component} data...")
    
    # Iterate over agents.
    for agent_num, values_for_composition in enumerate(plot_dict["composition_data"]):
        if(values_for_composition == {}):
            break
        print(f"\tAgent {agent_num}...")
        meta_data_dict[(args.arg_name, agent_num, component)] = {}
        
        # Iterate over epochs.
        for epochs, comp_dict in values_for_composition.items():
            print(f"\t\tEpoch {epochs}...")
            all_mask = comp_dict["all_mask"].astype(bool)                
            max_episode_len = all_mask.shape[1]
            all_mask = all_mask.reshape(-1, all_mask.shape[-1]).squeeze()
            one_episode = np.arange(max_episode_len)
            steps = np.broadcast_to(one_episode.reshape(1, max_episode_len, 1), (180, max_episode_len, 1))
            steps = steps.reshape(-1, steps.shape[-1]).squeeze()
            steps = steps[all_mask]
                                                            
            def process_component(key):
                data = comp_dict[key]
                data = data.reshape(-1, data.shape[-1])
                data = data[all_mask]
                return data

            data_dict = {"labels" : process_component("labels"), "component" : process_component(component)}
            meta_data_dict[(args.arg_name, agent_num, component)][epochs] = data_dict
    print(f"Got all {component} data for {args.arg_name}")
    """print("\nKeys in meta_data_dict!")
    print_dict_keys(meta_data_dict)
    print("\n")"""
            
            
            
# Make all reducers.
meta_reducer_dict = {}

def make_all_reducers(plot_dict, component, reducer_type, these_epochs):
    args = plot_dict["args"]
    print(f"Making {args.arg_name}'s {component} reducers...")
    
    # Iterate over agents.
    for agent_num, values_for_composition in enumerate(plot_dict["composition_data"]):
        if(values_for_composition == {}):
            break
        print(f"\tAgent {agent_num}...")
        meta_reducer_dict[(args.arg_name, agent_num, component, reducer_type)] = {}
        
        # Iterate over epochs.
        for epochs in these_epochs:
            print(f"\t\tEpoch {epochs}...")
            data_dict = meta_data_dict[(args.arg_name, agent_num, component)][epochs]
            meta_reducer_dict[(args.arg_name, agent_num, component, reducer_type)][epochs] = make_reducer(data_dict, reducer_type)
    print(f"Made {component} reducers for {args.arg_name}.")
    """print("\nKeys in meta_reducer_dict!")
    print_dict_keys(meta_reducer_dict)
    print("\n")"""



# For stable reducer-production. 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        


# How to make one reducer.
def make_reducer(data_dict, reducer_type):
    reducer_dict = {}
    # Iterate over three combinations of goal-parts.
    for classes in [("task", "color"), ("task", "shape"), ("color", "shape")]:
        print(f"\t\t\tClasses {classes}...")
        set_seed(42)
        scaler = StandardScaler()
        labels = data_dict["labels"]
        data_scaled = scaler.fit_transform(data_dict["component"])
        
        if(reducer_type == "pca"):
            reducer = PCA(n_components=2, random_state=42) 
            reducer.fit(data_scaled)
            
        if(reducer_type == "lda"):
            i = 0
            these_labels = np.zeros_like(labels)
            if("task" in classes):
                these_labels[:, i] = labels[:, 0]
                i += 1
            if("color" in classes):
                these_labels[:, i] = labels[:, 1]
                i += 1
            if("shape" in classes):
                these_labels[:, i] = labels[:, 2]                
            class_labels = [
                f"{class_1}_{class_2}"
                for class_1, class_2 in zip(these_labels[:, 0], these_labels[:, 1])]
            reducer = LDA(n_components=2)
            reducer.fit(data_scaled, class_labels)
            
        reducer_dict[classes] = {"scaler": scaler, "reducer" : reducer}
    return reducer_dict



meta_reduced_data_dict = {}

# Making all reduced data.
def make_all_reduced_data(plot_dict, component, reducer_type):
    args = plot_dict["args"]
    print(f"Reducing {args.arg_name}'s {component} data with {reducer_type}...")
    
    # Iterate over agents.
    for agent_num, values_for_composition in enumerate(plot_dict["composition_data"]):
        if(values_for_composition == {}):
            break
        print(f"\tAgent {agent_num}...")
        meta_reduced_data_dict[(args.arg_name, agent_num, component, reducer_type)] = {}
        
        # Iterate over epochs.
        for data_epochs in meta_data_dict[(args.arg_name, agent_num, component)].keys():
            for reducer_epochs in meta_reducer_dict[(args.arg_name, agent_num, component, reducer_type)].keys():
                print(f"\t\tData epochs {data_epochs}, Reducer epochs {reducer_epochs}...")
                data_dict = meta_data_dict[(args.arg_name, agent_num, component)][data_epochs]
                reducer_dict = meta_reducer_dict[(args.arg_name, agent_num, component, reducer_type)][reducer_epochs]
                meta_reduced_data_dict[(args.arg_name, agent_num, component, reducer_type)][data_epochs] = use_reducer(data_dict, reducer_dict)
    print(f"Reduced {component} data with {reducer_type} for {args.arg_name}.")
    """print("\nKeys in meta_reduced_data_dict!")
    print_dict_keys(meta_reduced_data_dict)
    print("\n")"""
    
    
    
def use_reducer(data_dict, reducer_dict):
    reduced_data_dict = {}
    labels = data_dict["labels"]
    reduced_data_dict["tasks"] = labels[:, 0]
    reduced_data_dict["colors"] = labels[:, 1]
    reduced_data_dict["shapes"] = labels[:, 2]
    reduced_data_dict["unique_tasks"] = np.unique(reduced_data_dict["tasks"])
    reduced_data_dict["unique_colors"] = np.unique(reduced_data_dict["colors"])
    reduced_data_dict["unique_shapes"] = np.unique(reduced_data_dict["shapes"])
    for classes in [("task", "color"), ("task", "shape"), ("color", "shape")]:  
        data_scaled = reducer_dict[classes]["scaler"].transform(data_dict["component"])
        reduced = reducer_dict[classes]["reducer"].transform(data_scaled)
        reduced_data_dict[classes] = reduced
    return(reduced_data_dict)

    
    
def smooth_plots(plot_dict, component, reducer_type, reducer_epochs, smooth_frames):
    args = plot_dict["args"]
    print(f"Plotting {args.arg_name}'s {component} data with {reducer_type}...")
    
    for agent_num, values_for_composition in enumerate(plot_dict["composition_data"]):
        if(values_for_composition == {}):
            break
        print(f"\tAgent {agent_num}...")
        reduced_data_dict = meta_reduced_data_dict[(args.arg_name, agent_num, component, reducer_type)][reducer_epochs]
        stopping_epochs = list(meta_data_dict[(args.arg_name, agent_num, component)].keys())
        starting_epochs = [None] + stopping_epochs[:-1]
        for start_epochs, stop_epochs in zip(starting_epochs, stopping_epochs):
            print(f"\t\tEpochs {start_epochs} to {stop_epochs}...")
            if(start_epochs == None):
                plot(reduced_data_dict, None, 1, component, reducer_type, stop_epochs, None, agent_num, reducer_epochs, args.arg_name)
            else:
                start_reduced_data_dict = meta_reduced_data_dict[(args.arg_name, agent_num, component, reducer_type)][start_epochs]
                stop_reduced_data_dict  = meta_reduced_data_dict[(args.arg_name, agent_num, component, reducer_type)][stop_epochs]
                for i in range(smooth_frames):
                    fraction_of_start = (smooth_frames - (i+1)) / smooth_frames
                    plot(start_reduced_data_dict, stop_reduced_data_dict, fraction_of_start, component, reducer_type, stop_epochs, i+1, agent_num, reducer_epochs, args.arg_name)

    
    
def plot(start_reduced_data_dict, stop_reduced_data_dict, fraction_of_start, component, reducer_type, data_epochs, smooth_frame, agent_num, reducer_epochs, arg_name):
    print(f"\t\t\tPlot {data_epochs}.{smooth_frame} of component {component} reducer_type {reducer_type} for agent {agent_num}")
    
    """print("\nKeys in reduced_data_dict!")
    print_dict_keys(reduced_data_dict)
    print("\n")"""
    
    fig, axes = plt.subplots(
        1, 4, 
        figsize=(19, 6), 
        dpi=dpi, 
        sharex=False, 
        sharey=False, 
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.4]})
    
    plt.suptitle(f"Tests for Compositionality\nAgent {agent_num}. data from {data_epochs} epochs, frame {smooth_frame}. {reducer_type} from {reducer_epochs} epochs, with {component}.", fontsize=16)
    
    plot_by_attribute(axes[0], start_reduced_data_dict, stop_reduced_data_dict, fraction_of_start, ("task", "color"),  "Task and Color")
    plot_by_attribute(axes[1], start_reduced_data_dict, stop_reduced_data_dict, fraction_of_start, ("task", "shape"),  "Task and Shape")
    plot_by_attribute(axes[2], start_reduced_data_dict, stop_reduced_data_dict, fraction_of_start, ("color", "shape"), "Color and Shape")

    ax = axes[3]

    y = 10
    for task_name, letter in task_mapping_letter.items():
        ax.scatter(.1, y, color="black", marker=f'${letter}$', alpha=0.4, s=letter_w_size if letter == "W" else letter_size, edgecolor='none')
        ax.text(.3, y, s = task_name, horizontalalignment='left', verticalalignment='center', fontsize = fontsize)
        y -= .5
        
    for color_name, color in color_mapping_color.items():
        ax.scatter(.1, y, facecolors=color, edgecolors='none', s=color_size, alpha=0.4)
        ax.text(.3, y, s = color_name, horizontalalignment='left', verticalalignment='center', fontsize = fontsize)
        y -= .5
        
    for shape_name, marker in shape_mapping_marker.items():
        colored_marker = shape_mapping_colored_marker[shape_name]["BLACK"]
        imagebox = OffsetImage(colored_marker, zoom=shape_size)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (.1, y), frameon=False, alpha=0.4, zorder=1)
        ax.add_artist(ab)
        ax.text(.3, y, s = shape_name, horizontalalignment='left', verticalalignment='center', fontsize = fontsize)
        y -= .5
        
    ax.scatter(.1, y, facecolors='none', edgecolors="#000000", linewidths=0.5, marker='o', alpha=.5, s=test_size, zorder=0, linestyle='dotted')
    ax.text(.3, y, s = "TEST", horizontalalignment='left', verticalalignment='center', fontsize = fontsize)
        
    ax.set_xticks([])      # remove x-axis ticks
    ax.set_yticks([])      # remove y-axis ticks
    ax.set_xticklabels([]) # remove x-axis tick labels
    ax.set_yticklabels([]) # remove y-axis tick labels
    ax.set_xlim([0, .8])
    ax.set_ylim([1, 10.5])
        
    os.makedirs(f"thesis_pics/composition/{arg_name}/agent_{agent_num}/{reducer_type}/{component}/{reducer_type}_{str(reducer_epochs).zfill(6)}", exist_ok = True)
    plt.savefig(f"thesis_pics/composition/{arg_name}/agent_{agent_num}/{reducer_type}/{component}/{reducer_type}_{str(reducer_epochs).zfill(6)}/data_{str(data_epochs).zfill(6)}.{str(smooth_frame).zfill(3)}.png", bbox_inches="tight")
    plt.close()
        
        
        
def plot_by_attribute(ax, start_reduced_data_dict, stop_reduced_data_dict, fraction_of_start, classes, title): 
    print(f"\t\t\t\t{title}...") 
    
    fraction_of_stop = 1 - fraction_of_start
    
    if(stop_reduced_data_dict == None):
        stop_reduced_data_dict = start_reduced_data_dict
        
    xs_for_min_max = []
    ys_for_min_max = []
    
    def make_groups(reduced_data_dict):
        grouped_points = defaultdict(list)
        grouped_letters = {}
        for task, color, shape, (x, y) in zip(reduced_data_dict["tasks"], reduced_data_dict["colors"], reduced_data_dict["shapes"], reduced_data_dict[classes]):
            if(not "task" in classes):
                task = None
            if(not "color" in classes):
                color = None
            if(not "shape" in classes):
                shape = None
            key = (task, color, shape)
            grouped_points[key].append((x, y))
            if key not in grouped_letters and task != None:
                grouped_letters[key] = task_mapping_letter[task_map[task].name]
        return(grouped_points, grouped_letters)
    
    start_grouped_points, start_grouped_letters = make_groups(start_reduced_data_dict)
    stop_grouped_points, stop_grouped_letters = make_groups(stop_reduced_data_dict)
    
    for ((task, color, shape), start_coords), ((_, _, _), stop_coords) in zip(start_grouped_points.items(), stop_grouped_points.items()):
        if("task" in classes):
            task_name = task_map[task].name
            letter = start_grouped_letters[(task, color, shape)]
        if("color" in classes):
            color_name = color_map[color].name
            color_val = color_mapping_color[color_name]
            if("task" in classes):
                text_color_val = color_mapping_color[color_name] # color_mapping_color_dark[color_name]
        else:
            text_color_val = "black"
        if("shape" in classes):
            shape_name = shape_map[shape].name
            marker = shape_mapping_marker[shape_name]
            if("color" in classes):
                colored_marker = shape_mapping_colored_marker[shape_name][color_name]
            else:
                colored_marker = shape_mapping_colored_marker[shape_name]["BLACK"]
        else:
            marker = "." # Might need to make this into a circle from image
        
        start_xs, start_ys = zip(*start_coords)  
        start_x = sum(start_xs) / len(start_xs) # Maybe median would be better?
        start_y = sum(start_ys) / len(start_ys)
        
        stop_xs, stop_ys = zip(*stop_coords)  
        stop_x = sum(stop_xs) / len(stop_xs) # Maybe median would be better?
        stop_y = sum(stop_ys) / len(stop_ys)
        #x = np.median(xs)
        #y = np.median(ys)
        
        x = fraction_of_start * start_x + fraction_of_stop * stop_x
        y = fraction_of_start * start_y + fraction_of_stop * stop_y
        xs_for_min_max.append(x)
        ys_for_min_max.append(y)

        if("shape" in classes):
            imagebox = OffsetImage(colored_marker, zoom=shape_size)  # Adjust zoom as needed
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
            ax.add_artist(ab)
        if("task" in classes):
            ax.scatter(x, y, color=text_color_val, marker=f'${letter}$', alpha=1, s=letter_w_size if letter == "W" else letter_size, edgecolor='none', zorder=2)
                        
    min_x = min(xs_for_min_max)
    max_x = max(xs_for_min_max)
    min_y = min(ys_for_min_max)
    max_y = max(ys_for_min_max)
    
    padding_ratio = 0.1  

    x_range = max_x - min_x
    y_range = max_y - min_y

    min_x -= x_range * padding_ratio
    max_x += x_range * padding_ratio
    min_y -= y_range * padding_ratio
    max_y += y_range * padding_ratio
    
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    if title == "Task and Color":
        ax.set_ylabel("Component 2")
    ax.set_xticks([])      # remove x-axis ticks
    ax.set_yticks([])      # remove y-axis ticks
    ax.set_xticklabels([]) # remove x-axis tick labels
    ax.set_yticklabels([]) # remove y-axis tick labels
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.grid(False)
        
        
    
#these_epochs = [0, 2500, 10000, 20000, 30000, 40000, 50000]
these_epochs = [i for i in range(0, 52500, 2500)]



plot_dicts, min_max_dict, complete_order = load_dicts(args)
for plot_dict in plot_dicts:
    for component in [
        #"hq", 
        "command_voice_zq"
        ]:
        get_all_data(
            plot_dict = plot_dict, 
            component = component)
        for reducer_type in ["lda"]:
            make_all_reducers(
                plot_dict = plot_dict, 
                component = component, 
                reducer_type = reducer_type, 
                these_epochs = [these_epochs[-1]])
            make_all_reduced_data(
                plot_dict = plot_dict, 
                component = component, 
                reducer_type = reducer_type)
            smooth_plots(
                plot_dict = plot_dict, 
                component = component, 
                reducer_type = reducer_type, 
                reducer_epochs = these_epochs[-1], 
                smooth_frames = 15)
print(f"\nDuration: {duration()}. Done!")
# %%