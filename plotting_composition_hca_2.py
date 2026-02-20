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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

from utils import args, duration, load_dicts, print, task_map, color_map, shape_map
from utils_submodule import  init_weights

# This file makes a series of PCA or LDA compositions, showing how an agent considered the relationships between tasks and colors over time.



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
letter_size = 120
letter_w_size = 200
shape_size = .6
test_size = 120
color_size = 120
shape_size = .4
fontsize = 10

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



# Collect all data.
meta_data_dict = {}

def get_all_data(plot_dict, component):
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
                    first_step = valid_steps[-1]
                    first_vectors.append(data[ep, first_step])

                return np.stack(first_vectors, axis=0)

            # Labels assumed constant per episode
            episode_labels = comp_dict['labels'][:, 0, :]   # (episodes, 3)

            data_dict = {
                'labels': episode_labels,
                'component': process_component_first_step(component)
            }
            meta_data_dict[(args.arg_name, agent_num, component)][epochs] = data_dict
    print('\nKeys in meta_data_dict!')
    print_dict_keys(meta_data_dict)
    print('\n')
    
    
    
def average_these_labels(plot_dict, component, label_averaging_tuple = ('task','color','shape')):
    """
    Collapses meta_data_dict vectors by averaging across selected label dimensions.

    label_averaging_tuple can contain any of:
        ('task',)
        ('color',)
        ('shape',)
        ('task','color')
        ('task','shape')
        ('color','shape')
        ('task','color','shape')

    Result overwrites meta_data_dict in-place, replacing episode-level vectors
    with group-mean vectors.
    """

    args = plot_dict['args']
    arg_name = args.arg_name

    label_index_map = {
        'task': 0,
        'color': 1,
        'shape': 2
    }

    # sanity check
    for key in label_averaging_tuple:
        if key not in label_index_map:
            raise ValueError(f"Unknown label type {key}. Must be one of {list(label_index_map.keys())}")

    label_dims = [label_index_map[k] for k in label_averaging_tuple]

    print(f"Averaging labels {label_averaging_tuple} for {arg_name} ({component})...")

    for (arg_name2, agent_num, comp_name), epoch_dict in meta_data_dict.items():
        if arg_name2 != arg_name or comp_name != component:
            continue

        for epoch, data_dict in epoch_dict.items():

            labels = data_dict["labels"]       # (N,3)
            vectors = data_dict["component"]   # (N,D)

            group_to_vecs = defaultdict(list)

            for i in range(len(vectors)):
                group_key = tuple(labels[i, d] for d in label_dims)
                group_to_vecs[group_key].append(vectors[i])

            new_vectors = []
            new_labels = []
            counts = []

            for group_key, vec_list in group_to_vecs.items():
                vec_array = np.stack(vec_list, axis=0)
                mean_vec = vec_array.mean(axis=0)

                # reconstruct a (task,color,shape) label row
                label_row = np.array([-1, -1, -1], dtype=int)

                for dim, val in zip(label_dims, group_key):
                    label_row[dim] = val

                new_vectors.append(mean_vec)
                new_labels.append(label_row)
                counts.append(len(vec_list))

            new_vectors = np.stack(new_vectors, axis=0)
            new_labels = np.stack(new_labels, axis=0)
            counts = np.array(counts)

            meta_data_dict[(arg_name, agent_num, component)][epoch] = {
                "labels": new_labels,
                "component": new_vectors,
                "counts": counts
            }

    print("Done averaging.\n")

            
            
            
# Make all reducers.
meta_reducer_dict = {}

def make_all_reducers(plot_dict, component, these_epochs):
    args = plot_dict['args']
    print(f'Making {args.arg_name}\'s {component} HCA reducers...')
    
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if(values_for_composition == {} or agent_num > max_agent_num):
            break

        print(f'\tAgent {agent_num}...')
        meta_reducer_dict[(args.arg_name, agent_num, component)] = {}
        
        for epochs in these_epochs:
            print(f'\t\tEpoch {epochs}...')
            data_dict = meta_data_dict[(args.arg_name, agent_num, component)][epochs]
            meta_reducer_dict[(args.arg_name, agent_num, component)][epochs] = make_reducer(data_dict)

    print(f'Made {component} HCA reducers for {args.arg_name}.')



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

    # Compute pairwise distances
    distances = pdist(data_scaled, metric='euclidean')
    
    # Hierarchical clustering (Ward requires euclidean)
    Z = linkage(distances, method='ward')

    reducer_dict = {
        'scaler': scaler,
        'linkage': Z,
        'distances': distances
    }

    return reducer_dict

    

# Plot the components of all pairs of tasks/colors/shapes, with legend.
# Plot the components of all pairs of tasks/colors/shapes, with legend.
def plot(component, data_epochs, agent_num, arg_name, K=10, max_labels_per_cluster=None):

    print(f'\t\t\tPlot {data_epochs} of component {component} for agent {agent_num} (HCA, {K} clusters)...')

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 8),
        dpi=dpi,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [4, 1]}
    )

    ax_tree = axes[0]
    ax_legend = axes[1]

    plt.suptitle(
        f'Hierarchical Clustering (HCA)\n'
        f'{arg_name} | Agent {agent_num} | Epoch {data_epochs} | {component}',
        fontsize=16
    )

    # ======================
    # GET LINKAGE
    # ======================

    reducer_info = meta_reducer_dict[(arg_name, agent_num, component)][data_epochs]
    Z = reducer_info['linkage']

    # ======================
    # TRUE LEAF ORDER (FULL TREE, NO TRUNCATION)
    # ======================

    d_full = dendrogram(
        Z,
        no_plot=True,
        no_labels=True
    )
    leaf_order = d_full["leaves"]   # safe: indices 0..(n_obs-1)

    # ======================
    # DENDROGRAM (TRUNCATED DISPLAY)
    # ======================

    dendro = dendrogram(
        Z,
        ax=ax_tree,
        no_labels=True,
        truncate_mode='lastp',
        p=K,
        color_threshold=None
    )

    ax_tree.set_title(f"Hierarchical Clustering (showing {K} clusters)")
    ax_tree.set_ylabel("Distance")
    ax_tree.set_xticks([])

    # ======================
    # CLUSTER MEMBERSHIPS (ORDERED BY TRUE LEAF ORDER)
    # ======================

    labels = meta_data_dict[(arg_name, agent_num, component)][data_epochs]['labels']
    cluster_ids = fcluster(Z, t=K, criterion="maxclust")

    # cluster_to_indices[cid] = [episode indices...], ordered left->right in tree
    cluster_to_indices = {}
    for leaf_idx in leaf_order:
        cid = cluster_ids[leaf_idx]
        cluster_to_indices.setdefault(cid, []).append(leaf_idx)

    # cluster_order = cluster IDs in left->right dendrogram order
    cluster_order = []
    seen = set()
    for leaf_idx in leaf_order:
        cid = cluster_ids[leaf_idx]
        if cid not in seen:
            seen.add(cid)
            cluster_order.append(cid)

    # ======================
    # PLACE SYMBOL COLUMNS (LEFT->RIGHT CLUSTER ORDER)
    # ======================

    x_min, x_max = ax_tree.get_xlim()
    cluster_width = (x_max - x_min) / len(cluster_order)

    y_min, y_max = ax_tree.get_ylim()
    base_y = -0.06 * y_max
    step_y = -0.04 * y_max  # vertical spacing

    max_cluster_size_seen = 0

    for col, cid in enumerate(cluster_order):

        x_pos = x_min + (col + 0.5) * cluster_width

        members = cluster_to_indices[cid]

        # Sort members for readability: task -> color -> shape
        members = sorted(
            members,
            key=lambda idx: (
                task_priority.get(task_map[labels[idx][0]].name, 999) if labels[idx][0] != -1 else 999,
                color_priority.get(color_map[labels[idx][1]].name, 999) if labels[idx][1] != -1 else 999,
                shape_priority.get(shape_map[labels[idx][2]].name, 999) if labels[idx][2] != -1 else 999,
            )
        )
        if len(members) == 0:
            continue

        max_cluster_size_seen = max(max_cluster_size_seen, len(members))

        # optional cap for readability
        if max_labels_per_cluster is not None and len(members) > max_labels_per_cluster:
            members_to_plot = members[:max_labels_per_cluster]
            truncated = True
        else:
            members_to_plot = members
            truncated = False

        for j, idx in enumerate(members_to_plot):

            task_id, color_id, shape_id = labels[idx]

            # --------------------------
            # Determine task
            # --------------------------
            if task_id != -1:
                task_name = task_map[task_id].name
                letter = task_mapping_letter[task_name]
            else:
                task_name = None
                letter = None

            # --------------------------
            # Determine color
            # --------------------------
            if color_id != -1:
                color_name = color_map[color_id].name
                color_val = color_mapping_color[color_name]
            else:
                color_name = None
                color_val = "#777777"   # gray if no color

            # --------------------------
            # Determine shape
            # --------------------------
            if shape_id != -1:
                shape_name = shape_map[shape_id].name
            else:
                shape_name = None

            y = base_y + j * step_y

            # =====================================================
            # DRAWING LOGIC (covers all combinations)
            # =====================================================

            # ---- Case 1: shape exists ----
            if shape_name is not None:

                if color_name is not None:
                    colored_marker = shape_mapping_colored_marker[shape_name][color_name]
                else:
                    colored_marker = shape_mapping_colored_marker[shape_name]["BLACK"]

                imagebox = OffsetImage(colored_marker, zoom=shape_size)
                ab = AnnotationBbox(imagebox, (x_pos, y), frameon=False, zorder=1)
                ax_tree.add_artist(ab)

            # ---- Case 2: task letter exists ----
            if letter is not None:
                ax_tree.scatter(
                    x_pos,
                    y,
                    color=color_val,
                    marker=f'${letter}$',
                    s=letter_size,
                    edgecolor='none',
                    zorder=2
                )

            # ---- Case 3: neither task nor shape exists ----
            if letter is None and shape_name is None:
                ax_tree.scatter(
                    x_pos,
                    y,
                    color=color_val,
                    marker='o',
                    s=30,
                    edgecolor='none',
                    zorder=2
                )

        # if truncated, add "..." marker
        if truncated:
            y = base_y + len(members_to_plot) * step_y
            ax_tree.text(x_pos, y, "...", ha='center', va='top', fontsize=10)

    # ======================
    # FIX Y LIMITS SO NOTHING IS CLIPPED
    # ======================

    lowest_y = base_y + (max_cluster_size_seen + 2) * step_y
    ax_tree.set_ylim(bottom=lowest_y)

    # ======================
    # LEGEND PANEL (UNCHANGED)
    # ======================

    y = 10

    for task_name, letter in task_mapping_letter.items():
        ax_legend.scatter(
            .1, y,
            color='black',
            marker=f'${letter}$',
            alpha=0.4,
            s=letter_w_size if letter == 'W' else letter_size,
            edgecolor='none'
        )
        ax_legend.text(.3, y, task_name,
                       horizontalalignment='left',
                       verticalalignment='center',
                       fontsize=fontsize)
        y -= .5

    for color_name, color in color_mapping_color.items():
        ax_legend.scatter(
            .1, y,
            facecolors=color,
            edgecolors='none',
            s=color_size,
            alpha=0.4
        )
        ax_legend.text(.3, y, color_name,
                       horizontalalignment='left',
                       verticalalignment='center',
                       fontsize=fontsize)
        y -= .5

    for shape_name, marker in shape_mapping_marker.items():
        colored_marker = shape_mapping_colored_marker[shape_name]['BLACK']
        imagebox = OffsetImage(colored_marker, zoom=shape_size)
        ab = AnnotationBbox(imagebox, (.1, y),
                            frameon=False,
                            alpha=0.4,
                            zorder=1)
        ax_legend.add_artist(ab)
        ax_legend.text(.3, y, shape_name,
                       horizontalalignment='left',
                       verticalalignment='center',
                       fontsize=fontsize)
        y -= .5

    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax_legend.set_xlim([0, .8])
    ax_legend.set_ylim([1, 10.5])

    # ======================
    # SAVE
    # ======================

    avg_tag = "_".join(label_averaging_tuple)
    save_dir = f'thesis_pics/composition/{arg_name}/agent_{agent_num}/{component}/hca_avg_{avg_tag}'
    
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        f'{save_dir}/epoch_{str(data_epochs).zfill(6)}_clusters_{K}_symbols.png',
        bbox_inches='tight'
    )
    plt.close()


        
    
# Iterate over components, etc.
these_epochs = [i for i in range(0, 60001, 10000)]

plot_dicts, min_max_dict, complete_order = load_dicts(args)

component = 'hq'
label_averaging_tuple = ('task', 'color', 'shape')
for plot_dict in plot_dicts:

    get_all_data(
        plot_dict=plot_dict,
        component=component
    )
    
    average_these_labels(
        plot_dict=plot_dict,
        component=component,
        label_averaging_tuple=label_averaging_tuple
    )

    make_all_reducers(
        plot_dict=plot_dict,
        component=component,
        these_epochs=these_epochs
    )

    # Now plot each epoch
    args = plot_dict['args']
    arg_name = args.arg_name

    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break

        for data_epochs in these_epochs:
            if data_epochs not in meta_reducer_dict[(arg_name, agent_num, component)]:
                continue

            plot(
                component=component,
                data_epochs=data_epochs,
                agent_num=agent_num,
                arg_name=arg_name
            )
print(f'\nDuration: {duration()}. Done!')
# %%