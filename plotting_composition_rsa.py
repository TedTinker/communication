#%%

import os
import random
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # Without this, pyplot crashes the kernal
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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import pdist, squareform

from utils import args, duration, load_dicts, print, task_map, color_map, shape_map, training_combos_3, testing_combos_3
from utils_submodule import  init_weights

# This file makes a series of RSA compositions, showing how an agent considered the relationships between tasks and colors over time.



print('name:\n{}\n'.format(args.arg_name),)



# Make tools for plotting task/color/shape.
task_priority = {
    "WATCH": 0,
    "BE NEAR": 1,
    "TOUCH THE TOP": 2,
    "PUSH FORWARD": 3,
    "PUSH LEFT": 4,
    "PUSH RIGHT": 5,
}

color_priority = {
    "RED": 0,
    "GREEN": 1,
    "BLUE": 2,
    "CYAN": 3,
    "MAGENTA": 4,
    "YELLOW": 5,
}

shape_priority = {
    "PILLAR": 0,
    "POLE": 1,
    "DUMBBELL": 2,
    "CONE": 3,
    "HOURGLASS": 4,
}

task_mapping_color = {
    'WATCH':            '#FF0000',  # Red
    'BE NEAR':          '#00FF00',  # Green
    'TOUCH THE TOP':    '#0000FF',  # Blue
    'PUSH FORWARD':     '#00DDDD',  # Cyan
    'PUSH LEFT':        '#FF00FF',  # Magenta
    'PUSH RIGHT':       '#DDDD00'}  # Yellow

task_mapping_letter = {
    'WATCH':            'W',
    'BE NEAR':          'N',
    'TOUCH THE TOP':    'T',
    'PUSH FORWARD':     'F',
    'PUSH LEFT':        'L',
    'PUSH RIGHT':       'R'}

color_mapping_color = {
    'RED':              '#FF0000',           
    'GREEN':            '#00FF00',
    'BLUE':             '#0000FF',
    'CYAN':             '#00DDDD',
    'MAGENTA':          '#FF00FF',
    'YELLOW':           '#DDDD00'}

def darken_hex_color(hex_color, factor=0.8):
    '''
    Darkens the given hex color by the specified factor.
    Factor should be between 0 (black) and 1 (no change).
    '''
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
letter_size = 550
letter_w_size = 600
shape_size = 1.2
test_size = 120
color_size = 120
fontsize = 20
step_y_scaler = 0.2

max_agent_num = 1

    
    
# Given a shape, apply a color.
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
    shape_mapping_colored_marker[shape_name]['BLACK'] = colorize_marker_image(shape_marker, '#000000')
    for color_name, color_color in color_mapping_color.items():
        shape_mapping_colored_marker[shape_name][color_name] = colorize_marker_image(shape_marker, color_color)



# Iteratively print keys and values in nested dictionaries.
def print_dict_keys(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('    ' * indent + str(key))
            print_dict_keys(value, indent + 1)
        elif hasattr(value, 'shape'):
            print('    ' * indent + f'{key} : shape {value.shape}')
        else:
            print('    ' * indent + str(key))



# How to remove unneeded data.
def filter_data_by_split(episode_labels, component_data, trained_untrained):

    if trained_untrained == "trained":
        allowed = set(training_combos_3)

    elif trained_untrained == "untrained":
        allowed = set(testing_combos_3)

    else:  # "all"
        return episode_labels, component_data

    to_keep = []

    for i, (t, c, s) in enumerate(episode_labels):
        if (t, c, s) in allowed:
            to_keep.append(i)

    to_keep = np.array(to_keep)

    return episode_labels[to_keep], component_data[to_keep]



# Collect all data.
meta_data_dict = {}

def get_all_data(plot_dict, component, trained_untrained):
    args = plot_dict['args']
    print(f'Getting {args.arg_name}\'s {component} data...')
    
    # Iterate over agents.
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if(values_for_composition == {} or agent_num > max_agent_num):
            break
        print(f'\tAgent {agent_num}...')
        meta_data_dict[(args.arg_name, agent_num, component)] = {}
        
        # Iterate over epochs.
        for epochs, comp_dict in values_for_composition.items():
            #if(epochs > 300):
            #    break
            print(f'\t\tEpoch {epochs}...')
            all_mask = comp_dict['all_mask'].astype(bool)   # (episodes, T, 1)
            all_mask = all_mask.squeeze(-1)                 # (episodes, T)

            def process_component_first_step(key):
                data = comp_dict[key]                       # (episodes, T, D)
                first_vectors = []

                for ep in range(data.shape[0]):
                    valid_steps = np.where(all_mask[ep])[0]
                    if len(valid_steps) == 0:
                        continue
                    first_vectors.append(data[ep, valid_steps[-1]])

                return np.stack(first_vectors, axis=0)

            # Labels assumed constant per episode
            episode_labels = comp_dict['labels'][:, 0, :]   # (episodes, 3)
            print(episode_labels)
            
            print(training_combos_3)
            
            component_data = process_component_first_step(component)
            print(component_data.shape)
            
            episode_labels, component_data = filter_data_by_split(
                episode_labels,
                component_data,
                trained_untrained)
            
            data_dict = {
                'labels': episode_labels,
                'component': component_data
            }
            meta_data_dict[(args.arg_name, agent_num, component)][epochs] = data_dict
    print('\nKeys in meta_data_dict!')
    print_dict_keys(meta_data_dict)
    print('\n')
    
    
    
def compute_rsm(data_dict, metric='cosine'):
    """
    Returns representational similarity matrix (RSM)
    """
    X = data_dict['component']  # (N, D)

    if metric == 'cosine':
        # cosine distance → convert to similarity
        D = squareform(pdist(X, metric='cosine'))
        S = 1 - D

    elif metric == 'euclidean':
        D = squareform(pdist(X, metric='euclidean'))
        S = -D  # or np.exp(-D)

    elif metric == 'correlation':
        D = squareform(pdist(X, metric='correlation'))
        S = 1 - D

    else:
        raise ValueError(metric)

    return S
    
    
    
def average_these_labels(data_dict, label_averaging_tuple=('task','color','shape')):
    label_index_map = {'task':0, 'color':1, 'shape':2}
    label_dims = [label_index_map[k] for k in label_averaging_tuple]

    labels = data_dict["labels"]
    vectors = data_dict["component"]

    group_to_vecs = defaultdict(list)

    for i in range(len(vectors)):
        key = tuple(labels[i, d] for d in label_dims)
        group_to_vecs[key].append(vectors[i])

    new_vectors = []
    new_labels = []

    for key, vecs in group_to_vecs.items():
        mean_vec = np.mean(vecs, axis=0)

        label_row = np.array([-1, -1, -1])
        for dim, val in zip(label_dims, key):
            label_row[dim] = val

        new_vectors.append(mean_vec)
        new_labels.append(label_row)

    return {
        "labels": np.array(new_labels),
        "component": np.array(new_vectors)
    }

            
            
            
# Make all reducers.
meta_reducer_dict = {}


# For stable reducer-production. 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        


# How to make one reducer.
def make_reducer(data_dict):

    print(f'\t\t\tComputing HCA...')
    set_seed(42)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_dict['component'])
    Z = linkage(data_scaled, method='ward')

    # Compute pairwise distances
    distances = pdist(data_scaled, metric='euclidean')
    
    # Hierarchical clustering (Ward requires euclidean)
    Z = linkage(data_scaled, method='ward')

    reducer_dict = {
        'scaler': scaler,
        'linkage': Z,
        'distances': distances
    }

    return reducer_dict

    

def plot(data_dict, epoch, agent_num, arg_name, label_averaging_tuple, trained_untrained):
    S = compute_rsm(data_dict, metric='cosine')

    # reorder for structure
    D = pdist(data_dict['component'], metric='cosine')
    Z = linkage(D, method='average')  # NOT ward
    order = leaves_list(Z)

    S = S[order][:, order]
    labels = data_dict['labels'][order]

    readable = []
    for t, c, s in labels:
        parts = []
        if t != -1:
            parts.append(task_map[t].name)
        if c != -1:
            parts.append(color_map[c].name)
        if s != -1:
            parts.append(shape_map[s].name)
        readable.append("_".join(parts))

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(S, aspect='equal')

    # ticks
    ax.set_xticks(np.arange(len(readable)))
    ax.set_yticks(np.arange(len(readable)))

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=90)

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Similarity')
    
    for i, (t, c, s) in enumerate(labels):

        # --- TASK LETTER ---
        if t != -1:
            task_name = task_map[t].name
            letter = task_mapping_letter[task_name]
        else:
            letter = None

        # --- COLOR ---
        if c != -1:
            color_name = color_map[c].name
            color_val = color_mapping_color[color_name]
        else:
            color_val = 'black'

        # --- SHAPE ---
        if s != -1:
            shape_name = shape_map[s].name
            colored_marker = shape_mapping_colored_marker[shape_name][color_name]
        else:
            colored_marker = None

        # =========================
        # X AXIS (top)
        # =========================
        if letter is not None:
            ax.text(
                i, -1.5, letter,
                ha='center', va='center',
                color=color_val,
                fontsize=10
            )

        if colored_marker is not None:
            imagebox = OffsetImage(colored_marker, zoom=0.4)
            ab = AnnotationBbox(imagebox, (i, -2.5), frameon=False)
            ax.add_artist(ab)

        # =========================
        # Y AXIS (left)
        # =========================
        if letter is not None:
            ax.text(
                -1.5, i, letter,
                ha='center', va='center',
                color=color_val,
                fontsize=10
            )

        if colored_marker is not None:
            imagebox = OffsetImage(colored_marker, zoom=0.4)
            ab = AnnotationBbox(imagebox, (-2.5, i), frameon=False)
            ax.add_artist(ab)

    # title
    ax.set_title(f'RSA ({trained_untrained})\n{arg_name} | Agent {agent_num} | Epoch {epoch}')

    plt.tight_layout()

    plt.title(f'RSA ({trained_untrained})\n{arg_name} | Agent {agent_num} | Epoch {epoch}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_xlim(-3, S.shape[1])
    ax.set_ylim(S.shape[0], -3)

    avg_tag = "_".join(label_averaging_tuple)
    save_dir = f'thesis_pics/rsa/{arg_name}/agent_{agent_num}/avg_{avg_tag}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f'{save_dir}/{trained_untrained}_epoch_{epoch:06d}.png', bbox_inches='tight')
    plt.close()


        
    
# Iterate over components, etc.
plot_dicts, min_max_dict, complete_order = load_dicts(args)

component = 'hq'
label_averaging_tuple = (
    'task',
    'color',
    'shape',
    )
trained_untrained = "untrained" # "trained" "untrained" "both"
for plot_dict in plot_dicts:

    get_all_data(plot_dict, component, trained_untrained)

    args = plot_dict['args']
    arg_name = args.arg_name

    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break

        for epoch, data_dict in meta_data_dict[(arg_name, agent_num, component)].items():

            averaged = average_these_labels(
                data_dict,
                label_averaging_tuple=('task','color','shape')
            )

            plot(
                averaged,
                epoch=epoch,
                agent_num=agent_num,
                arg_name=arg_name,
                label_averaging_tuple=label_averaging_tuple,
                trained_untrained = trained_untrained
            )
print(f'\nDuration: {duration()}. Done!')
# %%