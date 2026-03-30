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
import pickle

from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
from sklearn.decomposition import PCA, KernelPCA

from utils import args, duration, load_dicts, print, task_map, color_map, shape_map, testing_combos_3
from utils_submodule import  init_weights

# This file makes a series of PCA compositions, showing how an agent considered the relationships between tasks and colors over time.



print('name:\n{}\n'.format(args.arg_name),)



# Make tools for plotting task/color/shape.
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



GOAL_HIGHLIGHTS = [
    ('WATCH', 'MAGENTA', 'PILLAR'),
    ('BE NEAR', 'GREEN', 'POLE'),
]

skip_these_labels = [] # [(1, 4, 0), (2, 1, 1)] +  testing_combos_3
#dont_plot_these_labels = testing_combos_3



dpi = 400  
letter_size = 120
letter_w_size = 200
shape_size = .6
test_size = 120
color_size = 120
shape_size = .4
fontsize = 10

max_agent_num = 0

    
    
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
    print('HOW MUCH DATA:', len(plot_dict['composition_data']))
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        print(f'\tAgent {agent_num}...')
        meta_data_dict[(args.arg_name, agent_num, component)] = {}
        
        # Iterate over epochs.
        for epochs, comp_dict in values_for_composition.items():
            #if(epochs > 300):
            #    break
            print(f'\t\tEpoch {epochs}...')
            all_mask = comp_dict['all_mask'].astype(bool)                
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

            data_dict = {
                'labels' : process_component('labels'), 
                'component' : process_component(component)}
            meta_data_dict[(args.arg_name, agent_num, component)][epochs] = data_dict
    print('\nKeys in meta_data_dict!')
    print_dict_keys(meta_data_dict)
    print('\n')
            
            
            
# Make all reducers.
meta_reducer_dict = {}

def make_all_reducers(plot_dict, component, these_epochs):
    args = plot_dict['args']
    print(f'Making {args.arg_name}\'s {component} reducers...')
    
    # Iterate over agents.
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if(values_for_composition == {} or agent_num > max_agent_num):
            break
        print(f'\tAgent {agent_num}...')
        meta_reducer_dict[(args.arg_name, agent_num, component)] = {}
        
        # Iterate over epochs.
        for epochs in these_epochs:
            print(f'\t\tEpoch {epochs}...')
            data_dict = meta_data_dict[(args.arg_name, agent_num, component)][epochs]
            meta_reducer_dict[(args.arg_name, agent_num, component)][epochs] = make_reducer(data_dict)
    print(f'Made {component} reducers for {args.arg_name}.')
    '''print('\nKeys in meta_reducer_dict!')
    print_dict_keys(meta_reducer_dict)
    print('\n')'''



# For stable reducer-production. 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        


# How to make one reducer.
def make_reducer(data_dict):
    reducer_dict = {}
    # Iterate over three combinations of goal-parts.
    for classes in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:

        seed = 1111
        set_seed(seed)
        scaler = StandardScaler()
        labels = data_dict['labels']        # shape [N,3]
        components = data_dict['component']
        tuple_labels = [tuple(row) for row in labels]
        skip_these = np.array([lbl not in skip_these_labels for lbl in tuple_labels])
        training_components = components[skip_these]
        data_scaled = scaler.fit_transform(training_components)
        reducer = PCA(n_components=2, random_state=seed, svd_solver='randomized').fit(data_scaled)
        #reducer = KernelPCA(n_components=None, kernel='rbf', gamma=10, fit_inverse_transform=True, alpha=0.1).fit(data_scaled)
        reducer_dict[classes] = {'scaler': scaler, 'reducer': reducer}

    return reducer_dict



meta_reduced_data_dict = {}

# Making all reduced data.
def make_all_reduced_data(plot_dict, component):
    args = plot_dict['args']
    print(f'Reducing {args.arg_name}\'s {component} data...')
    
    # Iterate over agents.
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if(values_for_composition == {} or agent_num > max_agent_num):
            break
        print(f'\tAgent {agent_num}...')
        meta_reduced_data_dict[(args.arg_name, agent_num, component)] = {}
        
        # Iterate over epochs.
        for data_epochs in meta_data_dict[(args.arg_name, agent_num, component)].keys():
            print(f'\t\tData epochs {data_epochs}, Reducer epochs {data_epochs}...')
            data_dict = meta_data_dict[(args.arg_name, agent_num, component)][data_epochs]
            reducer_dict = meta_reducer_dict[(args.arg_name, agent_num, component)][data_epochs]
            meta_reduced_data_dict[(args.arg_name, agent_num, component)][data_epochs, data_epochs] = use_reducer(data_dict, reducer_dict)
    print(f'Reduced {component} data for {args.arg_name}.')
    '''print('\nKeys in meta_reduced_data_dict!')
    print_dict_keys(meta_reduced_data_dict)
    print('\n')'''
    
    

# Apply reducer.
def use_reducer(data_dict, reducer_dict):
    reduced_data_dict = {}
    labels = data_dict['labels']
    reduced_data_dict['labels'] = labels
    reduced_data_dict['tasks'] = labels[:, 0]
    reduced_data_dict['colors'] = labels[:, 1]
    reduced_data_dict['shapes'] = labels[:, 2]
    reduced_data_dict['unique_tasks'] = np.unique(reduced_data_dict['tasks'])
    reduced_data_dict['unique_colors'] = np.unique(reduced_data_dict['colors'])
    reduced_data_dict['unique_shapes'] = np.unique(reduced_data_dict['shapes'])
    for classes in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:  
        data_scaled = reducer_dict[classes]['scaler'].transform(data_dict['component'])
        reduced = reducer_dict[classes]['reducer'].transform(data_scaled)
        reduced_data_dict[classes] = reduced
    return(reduced_data_dict)



meta_aligned_data_dict = {}

# For all analysed epochs, find compositionality of all data combinations.
def make_all_aligned_data(plot_dict, component):
    args = plot_dict['args']
    print(f'Aligning {args.arg_name}\'s {component} data...')
    
    # Iterate over agents.
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if(values_for_composition == {} or agent_num > max_agent_num):
            break
        print(f'\tAgent {agent_num}...')
        meta_aligned_data_dict[(args.arg_name, agent_num, component)] = {}
        
        anchor_key = (0, 0) 
        for data_epochs, reducer_epochs in meta_reduced_data_dict[(args.arg_name, agent_num, component)].keys():
            if data_epochs > anchor_key[0] or reducer_epochs > anchor_key[1]:
                anchor_key = (data_epochs, reducer_epochs)
                
        print(f'\n\nanchor key: {anchor_key}\n\n')
        
        anchor_dict = meta_reduced_data_dict[(args.arg_name, agent_num, component)][anchor_key[0], anchor_key[1]]
        
        # Iterate over epochs.
        for data_epochs, reducer_epochs in meta_reduced_data_dict[(args.arg_name, agent_num, component)].keys():
            print(f'\t\tData epochs {data_epochs}, Reducer epochs {reducer_epochs}...')
            reduced_data_dict = meta_reduced_data_dict[(args.arg_name, agent_num, component)][data_epochs, reducer_epochs]
            meta_aligned_data_dict[(args.arg_name, agent_num, component)][data_epochs, reducer_epochs] = align_data(reduced_data_dict, anchor_dict)
    print(f'Aligned {component} data for {args.arg_name}.')
    print('\nKeys in meta_aligned_data_dict!')
    print_dict_keys(meta_aligned_data_dict)
    print('\n')
    
    

# Apply procrustes, finding similarity between datasets.
def align_data(reduced_data_dict_1, reduced_data_dict_2):
    aligned_data_dict = {}
    labels = reduced_data_dict_1['labels']
    aligned_data_dict['labels'] = labels
    aligned_data_dict['tasks'] = labels[:, 0]
    aligned_data_dict['colors'] = labels[:, 1]
    aligned_data_dict['shapes'] = labels[:, 2]
    aligned_data_dict['unique_tasks'] = np.unique(aligned_data_dict['tasks'])
    aligned_data_dict['unique_colors'] = np.unique(aligned_data_dict['colors'])
    aligned_data_dict['unique_shapes'] = np.unique(aligned_data_dict['shapes'])
    for classes in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:  
        aligned_data, _, _ = procrustes(reduced_data_dict_1[classes], reduced_data_dict_2[classes])
        aligned_data_dict[classes] = aligned_data
    return(aligned_data_dict)


    
# Make frames between two recorded epochs, to make a smooth video.
def smooth_plots(plot_dict, component, anchor_epochs, smooth_frames):
    args = plot_dict['args']
    print(f'Plotting {args.arg_name}\'s {component} data...')
    
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if(values_for_composition == {} or agent_num > max_agent_num):
            break
        print(f'\tAgent {agent_num}...')
        reduced_data_dict = meta_reduced_data_dict[(args.arg_name, agent_num, component)][0, 0]
        final_reduced_data_dict = meta_reduced_data_dict[(args.arg_name, agent_num, component)][anchor_epochs, anchor_epochs]
        stopping_epochs = list(meta_data_dict[(args.arg_name, agent_num, component)].keys())
        starting_epochs = [None] + stopping_epochs[:-1]
        for start_epochs, stop_epochs in zip(starting_epochs, stopping_epochs):
            print(f'\t\tEpochs {start_epochs} to {stop_epochs}...')
            if(start_epochs == None):
                pass
            else:
                start_aligned_data = meta_aligned_data_dict[(args.arg_name, agent_num, component)][start_epochs, start_epochs]
                stop_aligned_data = meta_aligned_data_dict[(args.arg_name, agent_num, component)][stop_epochs, stop_epochs]
                for i in range(smooth_frames):
                    fraction_of_start = (smooth_frames - (i+1)) / smooth_frames
                    plot_one(
                        start_aligned_data = start_aligned_data,
                        stop_aligned_data  = stop_aligned_data,
                        fraction_of_start  = fraction_of_start,
                        component          = component,
                        data_epochs        = stop_epochs,
                        smooth_frame       = i+1,
                        agent_num          = agent_num,
                        anchor_epochs      = anchor_epochs,
                        arg_name           = args.arg_name
                    )

    

# Plot the components of all pairs of tasks/colors/shapes, with legend.
def plot_marginal(ax, start_aligned_data, stop_aligned_data, fraction_of_start, average_over):
    """
    Like plot_all_attributes but collapses one attribute by averaging coordinates.
    average_over: 'task', 'color', or 'shape'
    """
    print(f'\t\t\t\tMarginal plot averaging over {average_over}...')

    classes_key = ('task', 'shape')
    f0 = fraction_of_start
    f1 = 1.0 - f0

    if stop_aligned_data is None:
        stop_aligned_data = start_aligned_data

    coords = f0 * start_aligned_data[classes_key] + f1 * stop_aligned_data[classes_key]

    tasks  = start_aligned_data['tasks']
    colors = start_aligned_data['colors']
    shapes = start_aligned_data['shapes']

    # Build groups keyed by the two attributes we're NOT averaging over.
    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        #if (t, c, s) in dont_plot_these_labels:
        #    continue
        if average_over == 'task':
            key = (c, s)
        elif average_over == 'color':
            key = (t, s)
        else:  # average_over == 'shape'
            key = (t, c)
        grouped[key].append((x, y, t, c, s))

    xs_for_min_max, ys_for_min_max = [], []
    for key, pts in grouped.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x, y = sum(xs)/len(xs), sum(ys)/len(ys)
        xs_for_min_max.append(x); ys_for_min_max.append(y)

        # Decode names from the first point (all share the non-averaged attributes).
        t, c, s = pts[0][2], pts[0][3], pts[0][4]
        task_name  = task_map[t].name
        color_name = color_map[c].name
        shape_name = shape_map[s].name

        # Draw shape icon — omit if averaging over shape.
        if average_over != 'shape':
            icon_color = color_name if average_over != 'color' else 'BLACK'
            colored_marker = shape_mapping_colored_marker[shape_name][icon_color]
            imagebox = OffsetImage(colored_marker, zoom=shape_size)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
            ax.add_artist(ab)

        # Draw task letter — omit if averaging over task.
        if average_over != 'task':
            letter = task_mapping_letter[task_name]
            text_color_val = color_mapping_color[color_name] if average_over != 'color' else '#000000'
            ax.scatter(x, y,
                       color=text_color_val,
                       marker=f'${letter}$',
                       alpha=1.0,
                       s=letter_w_size if letter == 'W' else letter_size,
                       edgecolor='none',
                       zorder=2)

    if xs_for_min_max and ys_for_min_max:
        min_x, max_x = min(xs_for_min_max), max(xs_for_min_max)
        min_y, max_y = min(ys_for_min_max), max(ys_for_min_max)
        pad = 0.10
        xr, yr = (max_x - min_x), (max_y - min_y)
        ax.set_xlim([min_x - xr*pad, max_x + xr*pad])
        ax.set_ylim([min_y - yr*pad, max_y + yr*pad])

    title_map = {
        'task':  'Color × Shape (averaged over task)',
        'color': 'Task × Shape (averaged over color)',
        'shape': 'Task × Color (averaged over shape)',
    }
    ax.set_title(title_map[average_over])
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(False)
    
    
    
def plot_one(start_aligned_data, stop_aligned_data, fraction_of_start,
             component, data_epochs, smooth_frame, agent_num, anchor_epochs, arg_name):

    # --- Original: all attributes ---
    print(f'\t\t\tPlot {data_epochs}.{smooth_frame} of component {component} for agent {agent_num} (ALL ATTRIBUTES)...')
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=dpi, constrained_layout=True)
    plt.suptitle(
        f'Compositionality with {arg_name}\n'
        f'Agent {agent_num} • epoch {data_epochs} • frame {smooth_frame} • {component}',
        fontsize=14)
    plot_all_attributes(ax, start_aligned_data, stop_aligned_data, fraction_of_start)
    outdir = f'thesis_pics/composition/{arg_name}/agent_{agent_num}/{component}_all'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/data_{str(data_epochs).zfill(6)}.{str(smooth_frame).zfill(3)}.png', bbox_inches='tight')
    plt.close()

    # --- Marginal plots ---
    for average_over in ['task', 'color', 'shape']:
        print(f'\t\t\tPlot {data_epochs}.{smooth_frame} • averaging over {average_over}...')
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=dpi, constrained_layout=True)
        plt.suptitle(
            f'Compositionality with {arg_name}\n'
            f'Agent {agent_num} • epoch {data_epochs} • frame {smooth_frame} • {component}',
            fontsize=14)
        plot_marginal(ax, start_aligned_data, stop_aligned_data, fraction_of_start, average_over=average_over)
        outdir = f'thesis_pics/composition/{arg_name}/agent_{agent_num}/{component}_avg_{average_over}'
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f'{outdir}/data_{str(data_epochs).zfill(6)}.{str(smooth_frame).zfill(3)}.png', bbox_inches='tight')
        plt.close()
        
        
        
# Plot those values with corresponding mapping.
def plot_all_attributes(ax, start_aligned_data, stop_aligned_data, fraction_of_start):
    '''
    Draw one plot with all (task, color, shape) combos at once.
    Uses the aligned coordinates from one pair-key (they're numerically the same
    pre-Procrustes, and post-Procrustes we just choose a canonical one).
    '''
    print(f'\t\t\t\tAll attributes (180 combos)...')

    # pick a canonical class key for coordinates (already aligned)
    classes_key = ('task', 'shape')
    f0 = fraction_of_start
    f1 = 1.0 - f0

    if stop_aligned_data is None:
        stop_aligned_data = start_aligned_data

    # interpolate coordinates in aligned space
    coords = f0 * start_aligned_data[classes_key] + f1 * stop_aligned_data[classes_key]

    tasks  = start_aligned_data['tasks']
    colors = start_aligned_data['colors']
    shapes = start_aligned_data['shapes']

    # group by full triple (task,color,shape) and average each cluster to one glyph
    from collections import defaultdict
    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        #if (t, c, s) in dont_plot_these_labels:
        #    continue
        grouped[(t, c, s)].append((x, y))

    xs_for_min_max, ys_for_min_max = [], []
    for (t, c, s), pts in grouped.items():
        xs, ys = zip(*pts)
        x, y = sum(xs)/len(xs), sum(ys)/len(ys)
        xs_for_min_max.append(x); ys_for_min_max.append(y)

        # decode human-readable names
        task_name  = task_map[t].name
        color_name = color_map[c].name
        shape_name = shape_map[s].name

        # glyph components
        letter           = task_mapping_letter[task_name]
        text_color_val   = color_mapping_color[color_name]
        colored_marker   = shape_mapping_colored_marker[shape_name][color_name]

        # draw shape icon (background)
        imagebox = OffsetImage(colored_marker, zoom=shape_size)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
        ax.add_artist(ab)
        
        if (task_name, color_name, shape_name) in GOAL_HIGHLIGHTS:
            # radius relative to data range
            r  = 0.001   # tweak thickness/scale as you like

            from matplotlib.patches import Circle
            circle = Circle((x, y), r,
                            fill=False,
                            linewidth=2.0,
                            edgecolor='#000000',
                            zorder=4)
            ax.add_patch(circle)

        # draw task letter (foreground) tinted by color
        ax.scatter(x, y,
                   color=text_color_val,
                   marker=f'${letter}$',
                   alpha=1.0,
                   s=letter_w_size if letter == 'W' else letter_size,
                   edgecolor='none',
                   zorder=2)

    # tidy axes
    if xs_for_min_max and ys_for_min_max:
        min_x, max_x = min(xs_for_min_max), max(xs_for_min_max)
        min_y, max_y = min(ys_for_min_max), max(ys_for_min_max)
        pad = 0.10
        xr, yr = (max_x - min_x), (max_y - min_y)
        ax.set_xlim([min_x - xr*pad, max_x + xr*pad])
        ax.set_ylim([min_y - yr*pad, max_y + yr*pad])

    ax.set_title('All task–color–shape combinations (180 glyphs)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(False)
        
        
    
# Iterate over components, etc.
these_epochs = [i for i in range(0, 60001, 2500)]

plot_dicts, min_max_dict, complete_order = load_dicts(args)

for plot_dict in plot_dicts:
    for component in [
        'hq', 'vision_zq', 'encoded_command'
        ]:
        print(f"\n\nHERE! {plot_dict.keys()} \n\n{plot_dict['composition_data'][0].keys()}\n\n")
        get_all_data(
            plot_dict = plot_dict, 
            component = component)
        make_all_reducers(
            plot_dict = plot_dict, 
            component = component, 
            these_epochs = these_epochs)
        make_all_reduced_data(
            plot_dict = plot_dict, 
            component = component)
        make_all_aligned_data(
            plot_dict = plot_dict, 
            component = component)
        smooth_plots(
            plot_dict = plot_dict, 
            component = component, 
            anchor_epochs = these_epochs[-1], 
            smooth_frames = 1)
        
print(f'\nDuration: {duration()}. Done!')
# %%