#%%
import os
import pickle
import random
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle

from utils import task_map, color_map, shape_map, duration, print

# ============================
# CONFIGURATION
# ============================
hyper_parameters = 'ef_pb2_3keys_q2_2'
agent_num_str = '0010'
epochs_str = '052089'
saved_file = 'saved_deigo'

# ============================
# LOAD SAVED PLOT DICT
# ============================
load_path = f'{saved_file}/{hyper_parameters}/plot_dict_agent_{agent_num_str}_epoch_{epochs_str}_composition.pickle'
with open(load_path, 'rb') as f:
    raw_plot_dict = pickle.load(f)

plot_dict = raw_plot_dict.copy()
plot_dict['composition_data'] = [raw_plot_dict['composition_data']]

the_epoch = int(epochs_str)
these_epochs = [the_epoch]

# ============================
# PLOTTING CONSTANTS
# ============================
GOAL_HIGHLIGHTS = [
    ('WATCH', 'MAGENTA', 'PILLAR'),
    ('BE NEAR', 'GREEN', 'POLE'),
]
skip_these_labels = []
dpi = 400
letter_size = 120
letter_w_size = 220
shape_size = 0.5
max_agent_num = 0

task_mapping_color = {
    'WATCH': '#FF0000', 'BE NEAR': '#00FF00', 'TOUCH THE TOP': '#0000FF',
    'PUSH FORWARD': '#00DDDD', 'PUSH LEFT': '#FF00FF', 'PUSH RIGHT': '#DDDD00'}

task_mapping_letter = {
    'WATCH': 'W', 'BE NEAR': 'N', 'TOUCH THE TOP': 'T',
    'PUSH FORWARD': 'F', 'PUSH LEFT': 'L', 'PUSH RIGHT': 'R'}

color_mapping_color = {
    'RED': '#FF0000', 'GREEN': '#00FF00', 'BLUE': '#0000FF',
    'CYAN': '#00DDDD', 'MAGENTA': '#FF00FF', 'YELLOW': '#DDDD00'}

shape_mapping_marker = {
    'PILLAR':    mpimg.imread('pybullet_data/shapes/pillar.png'),
    'POLE':      mpimg.imread('pybullet_data/shapes/pole.png'),
    'DUMBBELL':  mpimg.imread('pybullet_data/shapes/dumbbell.png'),
    'CONE':      mpimg.imread('pybullet_data/shapes/cone.png'),
    'HOURGLASS': mpimg.imread('pybullet_data/shapes/hourglass.png')}

def colorize_marker_image(marker_img, hex_color, alpha=0.3):
    hex_color = hex_color.lstrip('#')
    target_rgb = np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
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

# ============================
# PIPELINE FUNCTIONS
# ============================

meta_data_dict = {}

# Only first step.
def get_all_data(plot_dict, component):
    args = plot_dict['args']
    print(f'Getting {args.arg_name}\'s {component} data...')
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        meta_data_dict[(args.arg_name, agent_num, component)] = {}
        for epochs, comp_dict in values_for_composition.items():
            all_mask = comp_dict['all_mask'].astype(bool)
            all_mask = all_mask.reshape(-1, all_mask.shape[-1]).squeeze()
            def process_component(key):
                data = comp_dict[key]  # shape (episodes, steps, D)
                data = data[:, 0, :]   # just take first timestep, shape (episodes, D)
                return data
            labels = comp_dict['labels'][:, 0, :]
            meta_data_dict[(args.arg_name, agent_num, component)][epochs] = {
                'labels': process_component('labels'),
                'component': process_component(component)}
                
# All steps.
"""def get_all_data(plot_dict, component):
    args = plot_dict['args']
    print(f'Getting {args.arg_name}\'s {component} data...')
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        meta_data_dict[(args.arg_name, agent_num, component)] = {}
        for epochs, comp_dict in values_for_composition.items():
            all_mask = comp_dict['all_mask'].astype(bool)  # (episodes, steps, 1)

            def process_component(key):
                data = comp_dict[key]  # (episodes, steps, D)
                # Flatten episodes and steps, then mask out padding
                episodes, steps, D = data.shape
                mask_flat = all_mask.reshape(episodes * steps, -1).squeeze(-1)  # (episodes*steps,)
                data_flat = data.reshape(episodes * steps, D)
                return data_flat[mask_flat]  # only real (non-padded) timesteps

            labels_data = comp_dict['labels']  # (episodes, steps, 3)
            episodes, steps, _ = labels_data.shape
            mask_flat = all_mask.reshape(episodes * steps, -1).squeeze(-1).astype(bool)
            labels_flat = labels_data.reshape(episodes * steps, 3)
            labels_masked = labels_flat[mask_flat]

            meta_data_dict[(args.arg_name, agent_num, component)][epochs] = {
                'labels': labels_masked,
                'component': process_component(component)
            }"""


meta_reducer_dict = {}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_reducer(data_dict):
    reducer_dict = {}
    for classes in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:
        seed = 1111
        set_seed(seed)
        scaler = StandardScaler()
        labels = data_dict['labels']
        components = data_dict['component']
        tuple_labels = [tuple(row) for row in labels]
        skip_these = np.array([lbl not in skip_these_labels for lbl in tuple_labels])
        data_scaled = scaler.fit_transform(components[skip_these])
        
        reducer_2d = PCA(n_components=2, random_state=seed, svd_solver='randomized').fit(data_scaled)
        explained = reducer_2d.explained_variance_ratio_
        total_explained = explained.sum()

        print("Two components:")
        print(f"[PCA {classes}] Explained variance ratio: {explained}")
        print(f"[PCA {classes}] Total explained (2D): {total_explained:.4f}")
        
        reducer_3d = PCA(n_components=3, random_state=seed, svd_solver='randomized').fit(data_scaled)
        explained = reducer_3d.explained_variance_ratio_
        total_explained = explained.sum()

        print("Three components")
        print(f"[PCA {classes}] Explained variance ratio: {explained}")
        print(f"[PCA {classes}] Total explained (3D): {total_explained:.4f}")

        reducer_dict[classes] = {'scaler': scaler, 'reducer': reducer_2d}
    return reducer_dict



def make_all_reducers(plot_dict, component, these_epochs):
    args = plot_dict['args']
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break
        meta_reducer_dict[(args.arg_name, agent_num, component)] = {}
        for epochs in these_epochs:
            data_dict = meta_data_dict[(args.arg_name, agent_num, component)][epochs]
            meta_reducer_dict[(args.arg_name, agent_num, component)][epochs] = make_reducer(data_dict)


meta_reduced_data_dict = {}

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
        reduced_data_dict[classes] = reducer_dict[classes]['reducer'].transform(data_scaled)
    return reduced_data_dict

def make_all_reduced_data(plot_dict, component):
    args = plot_dict['args']
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break
        meta_reduced_data_dict[(args.arg_name, agent_num, component)] = {}
        for data_epochs in meta_data_dict[(args.arg_name, agent_num, component)].keys():
            data_dict = meta_data_dict[(args.arg_name, agent_num, component)][data_epochs]
            reducer_dict = meta_reducer_dict[(args.arg_name, agent_num, component)][data_epochs]
            meta_reduced_data_dict[(args.arg_name, agent_num, component)][data_epochs, data_epochs] = use_reducer(data_dict, reducer_dict)


meta_aligned_data_dict = {}

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
    return aligned_data_dict

def make_all_aligned_data(plot_dict, component):
    args = plot_dict['args']
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break
        meta_aligned_data_dict[(args.arg_name, agent_num, component)] = {}
        for data_epochs, reducer_epochs in meta_reduced_data_dict[(args.arg_name, agent_num, component)].keys():
            reduced_data_dict = meta_reduced_data_dict[(args.arg_name, agent_num, component)][data_epochs, reducer_epochs]
            meta_aligned_data_dict[(args.arg_name, agent_num, component)][data_epochs, reducer_epochs] = align_data(reduced_data_dict, reduced_data_dict)


# ============================
# PLOT FUNCTIONS
# ============================

def plot_all_attributes(ax, aligned_data):
    classes_key = ('task', 'shape')
    coords = aligned_data[classes_key]
    tasks  = aligned_data['tasks']
    colors = aligned_data['colors']
    shapes = aligned_data['shapes']
    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        grouped[(t, c, s)].append((x, y))
    xs_all, ys_all = [], []
    for (t, c, s), pts in grouped.items():
        xs, ys = zip(*pts)
        x, y = sum(xs)/len(xs), sum(ys)/len(ys)
        xs_all.append(x); ys_all.append(y)
        task_name  = task_map[t].name
        color_name = color_map[c].name
        shape_name = shape_map[s].name
        letter         = task_mapping_letter[task_name]
        text_color_val = color_mapping_color[color_name]
        colored_marker = shape_mapping_colored_marker[shape_name][color_name]
        imagebox = OffsetImage(colored_marker, zoom=shape_size)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
        ax.add_artist(ab)
        if (task_name, color_name, shape_name) in GOAL_HIGHLIGHTS:
            ax.add_patch(Circle((x, y), 0.001, fill=False, linewidth=2.0, edgecolor='#000000', zorder=4))
        ax.scatter(x, y, color=text_color_val,
                   marker=f'${letter}$', alpha=1.0,
                   s=letter_w_size if letter == 'W' else letter_size,
                   edgecolor='none', zorder=2)
    if xs_all:
        pad = 0.10
        xr, yr = max(xs_all)-min(xs_all), max(ys_all)-min(ys_all)
        ax.set_xlim([min(xs_all)-xr*pad, max(xs_all)+xr*pad])
        ax.set_ylim([min(ys_all)-yr*pad, max(ys_all)+yr*pad])
    ax.set_title('All task–color–shape combinations (180 glyphs)')
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    
    
    
def plot_single_attribute_value(ax, aligned_data, fix_attribute, fix_value):
    """Plot only the points where one attribute is fixed to a specific value."""
    classes_key = ('task', 'shape')
    coords = aligned_data[classes_key]
    tasks  = aligned_data['tasks']
    colors = aligned_data['colors']
    shapes = aligned_data['shapes']

    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        if fix_attribute == 'task'  and t != fix_value: continue
        if fix_attribute == 'color' and c != fix_value: continue
        if fix_attribute == 'shape' and s != fix_value: continue
        grouped[(t, c, s)].append((x, y))

    xs_all, ys_all = [], []
    for (t, c, s), pts in grouped.items():
        xs, ys = zip(*pts)
        x, y = sum(xs)/len(xs), sum(ys)/len(ys)
        xs_all.append(x); ys_all.append(y)
        task_name  = task_map[t].name
        color_name = color_map[c].name
        shape_name = shape_map[s].name
        letter         = task_mapping_letter[task_name]
        text_color_val = color_mapping_color[color_name]
        colored_marker = shape_mapping_colored_marker[shape_name][color_name]
        imagebox = OffsetImage(colored_marker, zoom=shape_size)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
        ax.add_artist(ab)
        ax.scatter(x, y, color=text_color_val,
                   marker=f'${letter}$', alpha=1.0,
                   s=letter_w_size if letter == 'W' else letter_size,
                   edgecolor='none', zorder=2)

    if xs_all:
        pad = 0.10
        xr, yr = max(xs_all)-min(xs_all), max(ys_all)-min(ys_all)
        ax.set_xlim([min(xs_all)-xr*pad, max(xs_all)+xr*pad])
        ax.set_ylim([min(ys_all)-yr*pad, max(ys_all)+yr*pad])

    id_map = {'task': task_map, 'color': color_map, 'shape': shape_map}[fix_attribute]
    attr_name = id_map[fix_value].name
    ax.set_title(attr_name, fontsize=6)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)



def plot_two_fixed_attributes(ax, aligned_data, fix_attribute_1, fix_value_1, fix_attribute_2, fix_value_2):
    classes_key = ('task', 'shape')
    coords = aligned_data[classes_key]
    tasks  = aligned_data['tasks']
    colors = aligned_data['colors']
    shapes = aligned_data['shapes']

    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        attrs = {'task': t, 'color': c, 'shape': s}
        if attrs[fix_attribute_1] != fix_value_1: continue
        if attrs[fix_attribute_2] != fix_value_2: continue
        grouped[(t, c, s)].append((x, y))

    xs_all, ys_all = [], []
    for (t, c, s), pts in grouped.items():
        xs, ys = zip(*pts)
        x, y = sum(xs)/len(xs), sum(ys)/len(ys)
        xs_all.append(x); ys_all.append(y)
        task_name  = task_map[t].name
        color_name = color_map[c].name
        shape_name = shape_map[s].name
        letter         = task_mapping_letter[task_name]
        text_color_val = color_mapping_color[color_name]
        colored_marker = shape_mapping_colored_marker[shape_name][color_name]
        imagebox = OffsetImage(colored_marker, zoom=shape_size * 2)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
        ax.add_artist(ab)
        ax.scatter(x, y, color=text_color_val,
                   marker=f'${letter}$', alpha=1.0,
                   s=letter_w_size * 4 if letter == 'W' else letter_size * 4,
                   edgecolor='none', zorder=2)

    if xs_all:
        pad = 0.10
        xr, yr = max(xs_all)-min(xs_all), max(ys_all)-min(ys_all)
        ax.set_xlim([min(xs_all)-xr*pad, max(xs_all)+xr*pad])
        ax.set_ylim([min(ys_all)-yr*pad, max(ys_all)+yr*pad])

    id_map_1 = {'task': task_map, 'color': color_map, 'shape': shape_map}[fix_attribute_1]
    id_map_2 = {'task': task_map, 'color': color_map, 'shape': shape_map}[fix_attribute_2]
    name_1 = id_map_1[fix_value_1].name
    name_2 = id_map_2[fix_value_2].name
    ax.set_title(f'{name_1}, {name_2}', fontsize=8)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    
    

def plot_marginal(ax, aligned_data, average_over, epoch):
    args = plot_dict['args']
    classes_key = ('task', 'shape')
    coords = aligned_data[classes_key]
    tasks  = aligned_data['tasks']
    colors = aligned_data['colors']
    shapes = aligned_data['shapes']
    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        if average_over == 'task':    key = (c, s)
        elif average_over == 'color': key = (t, s)
        else:                         key = (t, c)
        grouped[key].append((x, y, t, c, s))
    xs_all, ys_all = [], []
    for key, pts in grouped.items():
        x = sum(p[0] for p in pts) / len(pts)
        y = sum(p[1] for p in pts) / len(pts)
        xs_all.append(x); ys_all.append(y)
        t, c, s = pts[0][2], pts[0][3], pts[0][4]
        task_name  = task_map[t].name
        color_name = color_map[c].name
        shape_name = shape_map[s].name
        if average_over != 'shape':
            icon_color = color_name if average_over != 'color' else 'BLACK'
            colored_marker = shape_mapping_colored_marker[shape_name][icon_color]
            imagebox = OffsetImage(colored_marker, zoom=shape_size)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
            ax.add_artist(ab)
        if average_over != 'task':
            letter = task_mapping_letter[task_name]
            text_color_val = color_mapping_color[color_name] if average_over != 'color' else '#000000'
            ax.scatter(x, y, color=text_color_val,
                       marker=f'${letter}$', alpha=1.0,
                       s=letter_w_size if letter == 'W' else letter_size,
                       edgecolor='none', zorder=2)
    if xs_all:
        pad = 0.10
        xr, yr = max(xs_all)-min(xs_all), max(ys_all)-min(ys_all)
        ax.set_xlim([min(xs_all)-xr*pad, max(xs_all)+xr*pad])
        ax.set_ylim([min(ys_all)-yr*pad, max(ys_all)+yr*pad])
    title_map = {
        'task':  'Color × Shape (averaged over task)',
        'color': 'Task × Shape (averaged over color)',
        'shape': 'Task × Color (averaged over shape)'}
    ax.set_title(title_map[average_over])
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    
    # Single-fixed-attribute plots (subplots)
    for fix_attribute, unique_key, id_map in [
        ('task',  'unique_tasks',  task_map),
        ('color', 'unique_colors', color_map),
        ('shape', 'unique_shapes', shape_map),
    ]:
        unique_vals = aligned_data[unique_key]
        n_cols = len(unique_vals)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), dpi=dpi, constrained_layout=True)
        plt.suptitle(
            f'Compositionality with {args.arg_name}\n'
            f'Agent {agent_num_str} • epoch {epoch} • {component}\n'
            f'Fixed {fix_attribute.capitalize()}', fontsize=14)
        for j, fix_value in enumerate(unique_vals):
            ax = axes[j] if n_cols > 1 else axes
            plot_single_attribute_value(ax, aligned_data, fix_attribute, fix_value)
        outdir = f'thesis_pics/composition/{args.arg_name}/agent_{agent_num_str}/{component}/pca'
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f'{outdir}/{component}_fixed_{fix_attribute}_data_{str(epoch).zfill(6)}.png', bbox_inches='tight')
        plt.close()
        print(f'Saved fixed-{fix_attribute} subplot grid for {component} epoch {epoch}.')
        
        
        
def plot_marginal_split(ax, aligned_data, average_over, split_by):
    """
    Like plot_marginal, but restricted to a single value of split_by.
    average_over and split_by must be two different attributes from {'task','color','shape'}.
    The third attribute is what gets plotted (both its letter and icon).
    split_value is the actual integer id to filter on.
    
    This function plots into a single ax — the caller creates the subplot grid.
    """
    assert average_over != split_by
    classes_key = ('task', 'shape')
    coords = aligned_data[classes_key]
    tasks  = aligned_data['tasks']
    colors = aligned_data['colors']
    shapes = aligned_data['shapes']

    attr_arrays = {'task': tasks, 'color': colors, 'shape': shapes}

    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        attrs = {'task': t, 'color': c, 'shape': s}
        # Average over `average_over`, so group key excludes it
        remaining = {k: v for k, v in attrs.items() if k != average_over}
        key = (remaining['task'] if 'task' in remaining else None,
               remaining['color'] if 'color' in remaining else None,
               remaining['shape'] if 'shape' in remaining else None,
               t, c, s)
        grouped[key].append((x, y, t, c, s))

    # Re-group more cleanly: key = tuple of non-averaged attributes
    grouped2 = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        attrs = {'task': t, 'color': c, 'shape': s}
        key = tuple(v for k, v in sorted(attrs.items()) if k != average_over)
        grouped2[key].append((x, y, t, c, s))

    xs_all, ys_all = [], []
    for key, pts in grouped2.items():
        x = sum(p[0] for p in pts) / len(pts)
        y = sum(p[1] for p in pts) / len(pts)
        xs_all.append(x); ys_all.append(y)
        # Representative point for labels
        t, c, s = pts[0][2], pts[0][3], pts[0][4]
        task_name  = task_map[t].name
        color_name = color_map[c].name
        shape_name = shape_map[s].name

        if average_over != 'shape':
            icon_color = color_name if average_over != 'color' else 'BLACK'
            colored_marker = shape_mapping_colored_marker[shape_name][icon_color]
            imagebox = OffsetImage(colored_marker, zoom=shape_size)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
            ax.add_artist(ab)
        if average_over != 'task':
            letter = task_mapping_letter[task_name]
            text_color_val = color_mapping_color[color_name] if average_over != 'color' else '#000000'
            ax.scatter(x, y, color=text_color_val,
                       marker=f'${letter}$', alpha=1.0,
                       s=letter_w_size if letter == 'W' else letter_size,
                       edgecolor='none', zorder=2)

    if xs_all:
        pad = 0.10
        xr = max(xs_all) - min(xs_all)
        yr = max(ys_all) - min(ys_all)
        ax.set_xlim([min(xs_all) - xr * pad, max(xs_all) + xr * pad])
        ax.set_ylim([min(ys_all) - yr * pad, max(ys_all) + yr * pad])

    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def plot_marginal_split_grid(plot_dict, aligned_data, component, epoch, average_over, split_by):
    """
    Create a 1×N subplot grid: same marginal plot (averaging over `average_over`),
    one subplot per unique value of `split_by`.
    """
    args = plot_dict['args']
    split_key = {'task': 'unique_tasks', 'color': 'unique_colors', 'shape': 'unique_shapes'}[split_by]
    split_id_map = {'task': task_map, 'color': color_map, 'shape': shape_map}[split_by]
    unique_split_vals = aligned_data[split_key]
    n_cols = len(unique_split_vals)

    attr_arrays = {'task': aligned_data['tasks'], 'color': aligned_data['colors'], 'shape': aligned_data['shapes']}

    average_label_map = {
        'task':  'avg over task',
        'color': 'avg over color',
        'shape': 'avg over shape',
    }
    split_label_map = {
        'task':  'split by task',
        'color': 'split by color',
        'shape': 'split by shape',
    }

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), dpi=dpi, constrained_layout=True)
    plt.suptitle(
        f'Compositionality with {args.arg_name}\n'
        f'Agent {agent_num_str} • epoch {epoch} • {component}\n'
        f'{average_label_map[average_over]}, {split_label_map[split_by]}',
        fontsize=14)

    for j, split_val in enumerate(unique_split_vals):
        ax = axes[j] if n_cols > 1 else axes
        split_name = split_id_map[split_val].name

        # Build a filtered view of aligned_data for this split value
        mask = attr_arrays[split_by] == split_val
        filtered = {}
        filtered['tasks']  = aligned_data['tasks'][mask]
        filtered['colors'] = aligned_data['colors'][mask]
        filtered['shapes'] = aligned_data['shapes'][mask]
        filtered['labels'] = aligned_data['labels'][mask]
        filtered['unique_tasks']  = np.unique(filtered['tasks'])
        filtered['unique_colors'] = np.unique(filtered['colors'])
        filtered['unique_shapes'] = np.unique(filtered['shapes'])
        for ck in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:
            filtered[ck] = aligned_data[ck][mask]

        plot_marginal_split(ax, filtered, average_over=average_over, split_by=split_by)
        ax.set_title(split_name, fontsize=9)

    outdir = f'thesis_pics/composition/{args.arg_name}/agent_{agent_num_str}/{component}/pca'
    os.makedirs(outdir, exist_ok=True)
    fname = f'{outdir}/{component}_avg_{average_over}_split_{split_by}_data_{str(epoch).zfill(6)}.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Saved: avg {average_over}, split by {split_by} — epoch {epoch}.')
    


def plot_color_consistency(ax, aligned_data, average_over_shape=True):
    """
    Subtract each task's centroid then overlay all tasks in one space.
    If color is compositional, same-color letters cluster together regardless of task.
    average_over_shape: if True, average over shapes before plotting (one glyph per task×color).
    """
    classes_key = ('task', 'color')
    coords = aligned_data[classes_key]
    tasks  = aligned_data['tasks']
    colors = aligned_data['colors']
    shapes = aligned_data['shapes']

    # Average over shape within each (task, color) group first
    tc_grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        tc_grouped[(t, c)].append((x, y))

    tc_coords = {}
    for (t, c), pts in tc_grouped.items():
        xs, ys = zip(*pts)
        tc_coords[(t, c)] = (sum(xs)/len(xs), sum(ys)/len(ys))

    # Subtract per-task centroid
    unique_tasks = np.unique(tasks)
    task_centroids = {}
    for t in unique_tasks:
        pts = [tc_coords[(t, c)] for c in np.unique(colors) if (t, c) in tc_coords]
        xs, ys = zip(*pts)
        task_centroids[t] = (sum(xs)/len(xs), sum(ys)/len(ys))

    centered = {}
    for (t, c), (x, y) in tc_coords.items():
        cx, cy = task_centroids[t]
        centered[(t, c)] = (x - cx, y - cy)

    # Draw glyphs
    xs_all, ys_all = [], []
    for (t, c), (x, y) in centered.items():
        xs_all.append(x); ys_all.append(y)
        task_name  = task_map[t].name
        color_name = color_map[c].name
        letter     = task_mapping_letter[task_name]
        color_val  = color_mapping_color[color_name]
        ax.scatter(x, y, color=color_val,
                   marker=f'${letter}$', alpha=0.7,
                   s=letter_w_size if letter == 'W' else letter_size,
                   edgecolor='none', zorder=2)

    # Draw large faint dot at mean position per color across all tasks
    color_grouped = defaultdict(list)
    for (t, c), (x, y) in centered.items():
        color_grouped[c].append((x, y))
    for c, pts in color_grouped.items():
        xs, ys = zip(*pts)
        x, y = sum(xs)/len(xs), sum(ys)/len(ys)
        color_name = color_map[c].name
        color_val  = color_mapping_color[color_name]
        ax.scatter(x, y, color=color_val, s=500, alpha=0.2, edgecolor='none', zorder=1)

    if xs_all:
        pad = 0.15
        xr, yr = max(xs_all)-min(xs_all), max(ys_all)-min(ys_all)
        ax.set_xlim([min(xs_all)-xr*pad, max(xs_all)+xr*pad])
        ax.set_ylim([min(ys_all)-yr*pad, max(ys_all)+yr*pad])

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.4, zorder=0)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.4, zorder=0)
    ax.set_title('Task × Color (task-centroid subtracted)')
    ax.set_xlabel('Component 1 (task-centered)')
    ax.set_ylabel('Component 2 (task-centered)')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def plot_single_epoch(plot_dict, component, epoch):
    args = plot_dict['args']
    agent_num = 0
    aligned_data = meta_aligned_data_dict[(args.arg_name, agent_num, component)][epoch, epoch]

    # All attributes
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=dpi, constrained_layout=True)
    plt.suptitle(
        f'Compositionality with {args.arg_name}\n'
        f'Agent {agent_num_str} • epoch {epoch} • {component}', fontsize=14)
    plot_all_attributes(ax, aligned_data)
    outdir = f'thesis_pics/composition/{args.arg_name}/agent_{agent_num_str}/{component}/pca'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/{component}_all_data_{str(epoch).zfill(6)}.001.png', bbox_inches='tight')
    plt.close()
    print(f'Saved all-attributes plot for {component} epoch {epoch}.')

    # Marginal plots
    for average_over in ['task', 'color', 'shape']:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=dpi, constrained_layout=True)
        plt.suptitle(
            f'Compositionality with {args.arg_name}\n'
            f'Agent {agent_num_str} • epoch {epoch} • {component}', fontsize=14)
        plot_marginal(ax, aligned_data, average_over=average_over, epoch = epoch)
        outdir = f'thesis_pics/composition/{args.arg_name}/agent_{agent_num_str}/{component}/pca'
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f'{outdir}//{component}_avg_{average_over}_data_{str(epoch).zfill(6)}.001.png', bbox_inches='tight')
        plt.close()
        print(f'Saved marginal ({average_over}) plot for {component} epoch {epoch}.')

    # Color consistency plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=dpi, constrained_layout=True)
    plt.suptitle(
        f'Color Consistency with {args.arg_name}\n'
        f'Agent {agent_num_str} • epoch {epoch} • {component}', fontsize=14)
    plot_color_consistency(ax, aligned_data)
    outdir = f'thesis_pics/composition/{args.arg_name}/agent_{agent_num_str}/{component}/pca'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/{component}_color_consistency_data_{str(epoch).zfill(6)}.001.png', bbox_inches='tight')
    plt.close()
    print(f'Saved color consistency plot for {component} epoch {epoch}.')
    
    # Two-fixed-attribute plots (subplots)
    attribute_pairs = [
        ('task',  'unique_tasks',  task_map,  'color', 'unique_colors', color_map),
        ('task',  'unique_tasks',  task_map,  'shape', 'unique_shapes', shape_map),
        ('color', 'unique_colors', color_map, 'shape', 'unique_shapes', shape_map),
    ]
    for fix_attr_1, unique_key_1, id_map_1, fix_attr_2, unique_key_2, id_map_2 in attribute_pairs:
        unique_vals_1 = aligned_data[unique_key_1]
        unique_vals_2 = aligned_data[unique_key_2]
        n_rows = len(unique_vals_1)
        n_cols = len(unique_vals_2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=dpi, constrained_layout=True)
        plt.suptitle(
            f'Compositionality with {args.arg_name}\n'
            f'Agent {agent_num_str} • epoch {epoch} • {component}\n'
            f'{fix_attr_1.capitalize()} (rows) × {fix_attr_2.capitalize()} (cols)', fontsize=14)
        for i, fix_value_1 in enumerate(unique_vals_1):
            for j, fix_value_2 in enumerate(unique_vals_2):
                ax = axes[i, j] if n_rows > 1 else axes[j]
                plot_two_fixed_attributes(ax, aligned_data, fix_attr_1, fix_value_1, fix_attr_2, fix_value_2)
        outdir = f'thesis_pics/composition/{args.arg_name}/agent_{agent_num_str}/{component}/pca'
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f'{outdir}/{component}_fixed_{fix_attr_1}_{fix_attr_2}_data_{str(epoch).zfill(6)}.png', bbox_inches='tight')
        plt.close()
        print(f'Saved fixed-{fix_attr_1}/{fix_attr_2} subplot grid for {component} epoch {epoch}.')
        
    # ── 6 new marginal-split plots ──────────────────────────────────────────
    marginal_split_combos = [
        ('shape', 'color'),   # avg shapes, split by color
        ('shape', 'task'),    # avg shapes, split by task
        ('color', 'shape'),   # avg color,  split by shape
        ('color', 'task'),    # avg color,  split by task
        ('task',  'color'),   # avg task,   split by color
        ('task',  'shape'),   # avg task,   split by shape
    ]
    for average_over, split_by in marginal_split_combos:
        plot_marginal_split_grid(plot_dict, aligned_data, component, epoch,
                                 average_over=average_over, split_by=split_by)


# ============================
# RUN
# ============================
for component in ['command_voice_zq']:
    get_all_data(plot_dict, component)
    make_all_reducers(plot_dict, component, these_epochs)
    make_all_reduced_data(plot_dict, component)
    make_all_aligned_data(plot_dict, component)
    plot_single_epoch(plot_dict, component, the_epoch)

print(f'\nDuration: {duration()}. Done!')
# %%