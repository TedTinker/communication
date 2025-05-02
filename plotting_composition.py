#%%
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



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        

    
def make_reducer(data, labels, reducer_dict, args):
    for classes in [("task", "color"), ("task", "shape"), ("color", "shape")]:
        set_seed(42)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        if(args.reducer_type == "pca"):
            reducer = PCA(n_components=2, random_state=42) # I might need to do something totally different for LDA?
            reducer.fit(data_scaled)
        if(args.reducer_type == "lda"):
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
        reducer_dict[args.epochs][classes] = {"scaler": scaler, "reducer" : reducer}
    return reducer_dict



def use_reducer(data, reducer_dict):
    reduced_dict = {}
    for classes in [("task", "color"), ("task", "shape"), ("color", "shape")]:  
        data_scaled = reducer_dict[classes]["scaler"].transform(data)
        reduced = reducer_dict[classes]["reducer"].transform(data_scaled)
        reduced_dict["_".join(classes)] = reduced
    return(reduced_dict)



def use_reducer_by_epoch(data_dict, data_epochs, reducer_dict, reducer_epochs, agent_num):
    reduced_dict = {}
    data, labels = data_dict[data_epochs]
    reduced_dict["tasks"] = labels[:, 0]
    reduced_dict["colors"] = labels[:, 1]
    reduced_dict["shapes"] = labels[:, 2]
    reduced_dict["unique_tasks"] = np.unique(reduced_dict["tasks"])
    reduced_dict["unique_colors"] = np.unique(reduced_dict["colors"])
    reduced_dict["unique_shapes"] = np.unique(reduced_dict["shapes"])
    reducer_dict = reducer_dict[reducer_epochs]
    reduced_dict["reduced"] = use_reducer(data, reducer_dict)
    return reduced_dict





def plot_by_epoch_vs_epoch(reduced_dict, args):
    
    reduced_task_color  = reduced_dict["reduced"]["task_color"]
    reduced_task_shape  = reduced_dict["reduced"]["task_shape"]
    reduced_color_shape = reduced_dict["reduced"]["color_shape"]
    
    dpi = 400  
    letter_size = 120
    letter_w_size = 200
    shape_size = .6
    test_size = 120
    
    fig, axes = plt.subplots(
        1, 4, 
        figsize=(19, 6), 
        dpi=dpi, 
        sharex=False, 
        sharey=False, 
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.4]}  # Make the last (legend) plot thinner
    )

    def plot_by_attribute(ax, reduced, classes, title): 
        grouped_points = defaultdict(list)
        grouped_letters = {}
        classes_title = "_".join(classes)
        
        xs_for_min_max = []
        ys_for_min_max = []
        
        for task, color, shape, (x, y) in zip(reduced_dict["tasks"], reduced_dict["colors"], reduced_dict["shapes"], reduced_dict["reduced"][classes_title]):
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
                
        for (task, color, shape), coords in grouped_points.items():
            if("task" in classes):
                task_name = task_map[task].name
                letter = grouped_letters[(task, color, shape)]
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

            xs, ys = zip(*coords)  
            x = sum(xs) / len(xs) # Maybe mean would be better?
            y = sum(ys) / len(ys)
            #x = np.median(xs)
            #y = np.median(ys)
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
        
        padding_ratio = 0.1  # 5% of the range

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

    plot_by_attribute(axes[0], reduced_task_color, ("task", "color"),  "Task and Color")
    plot_by_attribute(axes[1], reduced_task_shape, ("task", "shape"),  "Task and Shape")
    plot_by_attribute(axes[2], reduced_color_shape, ("color", "shape"), "Color and Shape")
    
    # Make legend
    ax = axes[3]
    letter_size = 120
    letter_w_size = 200
    color_size = 120
    shape_size = .4
    test_size = 120
    fontsize = 12
    
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

    # Save
    plt.suptitle(f"Tests for Generalization\nAgent {args.agent_num}, data epoch {args.data_epochs}, {args.reducer_type} epochs {args.reducer_epochs}", fontsize=16)
    os.makedirs(f"thesis_pics/composition/{args.arg_name}/agent_{args.agent_num}/{args.reducer_type}/{args.reducer_type}_{args.reducer_epochs}", exist_ok = True)
    plt.savefig(f"thesis_pics/composition/{args.arg_name}/agent_{args.agent_num}/{args.reducer_type}/{args.reducer_type}_{args.reducer_epochs}/data_{args.data_epochs}.png", bbox_inches="tight")
    plt.close()

                    
                    
def plot_dimension_reduction(plot_dict, reducer_type):
    args = plot_dict["args"]
    args.reducer_type = reducer_type
    for agent_num, values_for_composition in enumerate(plot_dict["composition_data"]):
        args.agent_num = agent_num

        if(values_for_composition != {}):
            print("\nAGENT NUM", agent_num)
            reducer_dict = {}
            data_dict = {}
        
            for epochs, comp_dict in values_for_composition.items():
                print("EPOCHS", epochs)
                args.epochs = epochs
                reducer_dict[epochs] = {}

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

                components = ["labels", "hq"] # Add zq for command here
                results = {}
                for key in components:
                    results[key] = process_component(key)                    
                
                labels = results["labels"]
                                                    
                os.makedirs(f"thesis_pics/composition/{args.arg_name}/agent_{agent_num}", exist_ok=True)  
                    
                data = results["hq"] # Need to add zq
                data_dict[epochs] = (data, labels)

                reducer_dict = make_reducer(data, labels, reducer_dict, args)
                    
            for data_epochs in values_for_composition.keys():
                args.data_epochs = data_epochs
                for reducer_epochs in values_for_composition.keys():
                    args.reducer_epochs = reducer_epochs
                    os.makedirs(f"thesis_pics/composition/{args.arg_name}/agent_{agent_num}/{reducer_type}/{reducer_type}_{reducer_epochs}", exist_ok=True)  
                    print(f"DATA: {data_epochs}. {reducer_type}: {reducer_epochs}")
                    reduced_dict = use_reducer_by_epoch(data_dict, data_epochs, reducer_dict, reducer_epochs, agent_num)
                    plot_by_epoch_vs_epoch(reduced_dict, args)
                        
                        
                            
                    


def plots(plot_dicts):
    for i, plot_dict in enumerate(plot_dicts):
        args = plot_dict["args"]
        os.makedirs(f"saved_deigo/thesis_pics/composition/{args.arg_name}", exist_ok = True)
        for reducer_type in [
            "pca", 
            #"lda"
            ]:
            os.makedirs(f"saved_deigo/thesis_pics/composition/{reducer_type}/{args.arg_name}", exist_ok = True)
            print(f"\nStarting {args.arg_name}, {reducer_type}")        
            plot_dimension_reduction(plot_dict, reducer_type)
            print(f"Finished.")
        print(f"Duration: {duration()}")
        
    print(f"\tFinished composition.")



if(os.getcwd().split("/")[-1] != "communication"): 
    os.chdir("communication")
os.chdir(f"saved_{args.comp}")
os.makedirs("thesis_pics/composition", exist_ok=True)
    
plot_dicts, min_max_dict, complete_order = load_dicts(args)
plots(plot_dicts)
print(f"\nDuration: {duration()}. Done!")
# %%