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
from sklearn.linear_model import LogisticRegression
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle

from utils import task_map, color_map, shape_map, duration, print

# ============================
# CONFIGURATION
# ============================
hyper_parameters = 'ef_keys_q1_7'
agent_num_str = '0001'
epochs_str = '060000'
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

def get_all_data(plot_dict, component):
    args = plot_dict['args']
    print(f'Getting {args.arg_name}\'s {component} data...')
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        meta_data_dict[(args.arg_name, agent_num, component)] = {}
        for epochs, comp_dict in values_for_composition.items():
            def process_component(key):
                data = comp_dict[key]  # shape (episodes, steps, D)
                data = data[:, 0, :]   # just take first timestep, shape (episodes, D)
                return data
            meta_data_dict[(args.arg_name, agent_num, component)][epochs] = {
                'labels':    process_component('labels'),
                'component': process_component(component)}


# ============================
# UNSUPERVISED (PCA) PIPELINE
# ============================

meta_reducer_dict = {}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_reducer(data_dict):
    reducer_dict = {}
    seed = 1111
    set_seed(seed)
    scaler = StandardScaler()
    components = data_dict['component']
    data_scaled = scaler.fit_transform(components)

    reducer_2d = PCA(n_components=2, random_state=seed, svd_solver='randomized').fit(data_scaled)
    explained = reducer_2d.explained_variance_ratio_
    print(f'[PCA] Explained variance ratio: {explained}, Total (2D): {explained.sum():.4f}')

    reducer_3d = PCA(n_components=3, random_state=seed, svd_solver='randomized').fit(data_scaled)
    explained = reducer_3d.explained_variance_ratio_
    print(f'[PCA] Explained variance ratio: {explained}, Total (3D): {explained.sum():.4f}')

    # Store one shared reducer (same data regardless of classes pair)
    for classes in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:
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
    reduced_data_dict['labels']        = labels
    reduced_data_dict['tasks']         = labels[:, 0]
    reduced_data_dict['colors']        = labels[:, 1]
    reduced_data_dict['shapes']        = labels[:, 2]
    reduced_data_dict['unique_tasks']  = np.unique(reduced_data_dict['tasks'])
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
            data_dict    = meta_data_dict[(args.arg_name, agent_num, component)][data_epochs]
            reducer_dict = meta_reducer_dict[(args.arg_name, agent_num, component)][data_epochs]
            meta_reduced_data_dict[(args.arg_name, agent_num, component)][data_epochs, data_epochs] = use_reducer(data_dict, reducer_dict)


meta_aligned_data_dict = {}

def align_data(reduced_data_dict_1, reduced_data_dict_2):
    aligned_data_dict = {}
    labels = reduced_data_dict_1['labels']
    aligned_data_dict['labels']        = labels
    aligned_data_dict['tasks']         = labels[:, 0]
    aligned_data_dict['colors']        = labels[:, 1]
    aligned_data_dict['shapes']        = labels[:, 2]
    aligned_data_dict['unique_tasks']  = np.unique(aligned_data_dict['tasks'])
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
# SUPERVISED (PROBE) PIPELINE
# ============================

meta_probe_dict = {}
meta_probe_reduced_dict = {}

def train_probes(data_dict):
    """Train one linear probe per attribute, return dict of fitted classifiers."""
    components = data_dict['component']  # (N, 256)
    labels     = data_dict['labels']     # (N, 3)
    probe_dict = {}
    for attr_idx, attr_name in enumerate(['task', 'color', 'shape']):
        y   = labels[:, attr_idx]
        clf = LogisticRegression(max_iter=1000, random_state=1111)
        clf.fit(components, y)
        acc = clf.score(components, y)
        print(f'[Probe {attr_name}] Train accuracy: {acc:.4f}, coef shape: {clf.coef_.shape}')
        probe_dict[attr_name] = clf
    return probe_dict

def probe_reduce(data_dict, probe_dict):
    """Project data into each probe's class space, then PCA to 2D."""
    components = data_dict['component']  # (N, 256)
    labels     = data_dict['labels']

    reduced = {}
    reduced['labels']        = labels
    reduced['tasks']         = labels[:, 0]
    reduced['colors']        = labels[:, 1]
    reduced['shapes']        = labels[:, 2]
    reduced['unique_tasks']  = np.unique(reduced['tasks'])
    reduced['unique_colors'] = np.unique(reduced['colors'])
    reduced['unique_shapes'] = np.unique(reduced['shapes'])

    for attr_name, clf in probe_dict.items():
        W         = clf.coef_                        # (n_classes, 256)
        projected = components @ W.T                 # (N, n_classes)
        scaler    = StandardScaler()
        projected_scaled = scaler.fit_transform(projected)
        pca       = PCA(n_components=2, random_state=1111)
        coords_2d = pca.fit_transform(projected_scaled)
        explained = pca.explained_variance_ratio_
        print(f'[Probe PCA {attr_name}] Explained: {explained}, Total: {explained.sum():.4f}')
        # Store under all three classes keys so plot functions can use the same key
        for classes in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:
            reduced[f'probe_{attr_name}_{classes}'] = coords_2d
    return reduced

def make_all_probes(plot_dict, component, these_epochs):
    args = plot_dict['args']
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break
        meta_probe_dict[(args.arg_name, agent_num, component)]         = {}
        meta_probe_reduced_dict[(args.arg_name, agent_num, component)] = {}
        for epochs in these_epochs:
            data_dict  = meta_data_dict[(args.arg_name, agent_num, component)][epochs]
            probe_dict = train_probes(data_dict)
            meta_probe_dict[(args.arg_name, agent_num, component)][epochs]         = probe_dict
            meta_probe_reduced_dict[(args.arg_name, agent_num, component)][epochs] = probe_reduce(data_dict, probe_dict)


# ============================
# SHARED PLOT HELPERS
# ============================

def _get_coords(data, probe_attr=None, classes_key=('task', 'shape')):
    """Return 2D coordinates from either aligned (PCA) or probe-reduced data."""
    if probe_attr is not None:
        return data[f'probe_{probe_attr}_{classes_key}']
    return data[classes_key]

def _draw_glyph(ax, x, y, task_name, color_name, shape_name,
                icon_zoom=None, letter_scale=1.0, show_icon=True, show_letter=True):
    if icon_zoom is None:
        icon_zoom = shape_size
    if show_icon:
        colored_marker = shape_mapping_colored_marker[shape_name][color_name]
        imagebox = OffsetImage(colored_marker, zoom=icon_zoom)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
        ax.add_artist(ab)
    if show_letter:
        letter        = task_mapping_letter[task_name]
        text_color    = color_mapping_color[color_name]
        s             = (letter_w_size if letter == 'W' else letter_size) * letter_scale
        ax.scatter(x, y, color=text_color, marker=f'${letter}$',
                   alpha=1.0, s=s, edgecolor='none', zorder=2)

def _set_limits(ax, xs, ys, pad=0.10):
    if not xs:
        return
    xr = max(xs) - min(xs)
    yr = max(ys) - min(ys)
    ax.set_xlim([min(xs) - xr * pad, max(xs) + xr * pad])
    ax.set_ylim([min(ys) - yr * pad, max(ys) + yr * pad])


# ============================
# PLOT FUNCTIONS
# ============================

def plot_all_attributes(ax, data, probe_attr=None):
    classes_key = ('task', 'shape')
    coords = _get_coords(data, probe_attr, classes_key)
    tasks  = data['tasks']
    colors = data['colors']
    shapes = data['shapes']

    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        grouped[(t, c, s)].append((x, y))

    xs_all, ys_all = [], []
    for (t, c, s), pts in grouped.items():
        x, y = np.mean(pts, axis=0)
        xs_all.append(x); ys_all.append(y)
        task_name  = task_map[t].name
        color_name = color_map[c].name
        shape_name = shape_map[s].name
        if (task_name, color_name, shape_name) in GOAL_HIGHLIGHTS:
            ax.add_patch(Circle((x, y), 0.001, fill=False, linewidth=2.0, edgecolor='#000000', zorder=4))
        _draw_glyph(ax, x, y, task_name, color_name, shape_name)

    _set_limits(ax, xs_all, ys_all)
    prefix = f'Probe ({probe_attr}) — ' if probe_attr else ''
    ax.set_title(f'{prefix}All task–color–shape combinations (180 glyphs)')
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def plot_single_attribute_value(ax, data, fix_attribute, fix_value, probe_attr=None):
    classes_key = ('task', 'shape')
    coords = _get_coords(data, probe_attr, classes_key)
    tasks  = data['tasks']
    colors = data['colors']
    shapes = data['shapes']

    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        if fix_attribute == 'task'  and t != fix_value: continue
        if fix_attribute == 'color' and c != fix_value: continue
        if fix_attribute == 'shape' and s != fix_value: continue
        grouped[(t, c, s)].append((x, y))

    xs_all, ys_all = [], []
    for (t, c, s), pts in grouped.items():
        x, y = np.mean(pts, axis=0)
        xs_all.append(x); ys_all.append(y)
        _draw_glyph(ax, x, y, task_map[t].name, color_map[c].name, shape_map[s].name)

    _set_limits(ax, xs_all, ys_all)
    id_map    = {'task': task_map, 'color': color_map, 'shape': shape_map}[fix_attribute]
    attr_name = id_map[fix_value].name
    ax.set_title(attr_name, fontsize=6)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def plot_two_fixed_attributes(ax, data, fix_attribute_1, fix_value_1, fix_attribute_2, fix_value_2, probe_attr=None):
    classes_key = ('task', 'shape')
    coords = _get_coords(data, probe_attr, classes_key)
    tasks  = data['tasks']
    colors = data['colors']
    shapes = data['shapes']

    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        attrs = {'task': t, 'color': c, 'shape': s}
        if attrs[fix_attribute_1] != fix_value_1: continue
        if attrs[fix_attribute_2] != fix_value_2: continue
        grouped[(t, c, s)].append((x, y))

    xs_all, ys_all = [], []
    for (t, c, s), pts in grouped.items():
        x, y = np.mean(pts, axis=0)
        xs_all.append(x); ys_all.append(y)
        _draw_glyph(ax, x, y, task_map[t].name, color_map[c].name, shape_map[s].name,
                    icon_zoom=shape_size * 2, letter_scale=4.0)

    _set_limits(ax, xs_all, ys_all)
    id_map_1 = {'task': task_map, 'color': color_map, 'shape': shape_map}[fix_attribute_1]
    id_map_2 = {'task': task_map, 'color': color_map, 'shape': shape_map}[fix_attribute_2]
    name_1   = id_map_1[fix_value_1].name
    name_2   = id_map_2[fix_value_2].name
    ax.set_title(f'{name_1}, {name_2}', fontsize=8)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def plot_marginal(ax, data, average_over, probe_attr=None):
    classes_key = ('task', 'shape')
    coords = _get_coords(data, probe_attr, classes_key)
    tasks  = data['tasks']
    colors = data['colors']
    shapes = data['shapes']

    grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        if average_over == 'task':    key = (c, s)
        elif average_over == 'color': key = (t, s)
        else:                         key = (t, c)
        grouped[key].append((x, y, t, c, s))

    xs_all, ys_all = [], []
    for key, pts in grouped.items():
        x  = sum(p[0] for p in pts) / len(pts)
        y  = sum(p[1] for p in pts) / len(pts)
        xs_all.append(x); ys_all.append(y)
        t, c, s    = pts[0][2], pts[0][3], pts[0][4]
        task_name  = task_map[t].name
        color_name = color_map[c].name
        shape_name = shape_map[s].name
        show_icon   = average_over != 'shape'
        show_letter = average_over != 'task'
        icon_color  = color_name if average_over != 'color' else 'BLACK'
        colored_marker = shape_mapping_colored_marker[shape_name][icon_color]
        if show_icon:
            imagebox = OffsetImage(colored_marker, zoom=shape_size)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
            ax.add_artist(ab)
        if show_letter:
            letter        = task_mapping_letter[task_name]
            text_color    = color_mapping_color[color_name] if average_over != 'color' else '#000000'
            s_val         = letter_w_size if letter == 'W' else letter_size
            ax.scatter(x, y, color=text_color, marker=f'${letter}$',
                       alpha=1.0, s=s_val, edgecolor='none', zorder=2)

    _set_limits(ax, xs_all, ys_all)
    title_map = {
        'task':  'Color × Shape (averaged over task)',
        'color': 'Task × Shape (averaged over color)',
        'shape': 'Task × Color (averaged over shape)'}
    prefix = f'Probe ({probe_attr}) — ' if probe_attr else ''
    ax.set_title(f'{prefix}{title_map[average_over]}')
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def plot_color_consistency(ax, data, probe_attr=None):
    classes_key = ('task', 'color')
    coords = _get_coords(data, probe_attr, classes_key)
    tasks  = data['tasks']
    colors = data['colors']
    shapes = data['shapes']

    tc_grouped = defaultdict(list)
    for t, c, s, (x, y) in zip(tasks, colors, shapes, coords):
        tc_grouped[(t, c)].append((x, y))

    tc_coords = {k: np.mean(v, axis=0) for k, v in tc_grouped.items()}

    unique_tasks   = np.unique(tasks)
    unique_colors  = np.unique(colors)
    task_centroids = {}
    for t in unique_tasks:
        pts = [tc_coords[(t, c)] for c in unique_colors if (t, c) in tc_coords]
        task_centroids[t] = np.mean(pts, axis=0)

    centered = {(t, c): (xy - task_centroids[t]) for (t, c), xy in tc_coords.items()}

    xs_all, ys_all = [], []
    for (t, c), (x, y) in centered.items():
        xs_all.append(x); ys_all.append(y)
        letter    = task_mapping_letter[task_map[t].name]
        color_val = color_mapping_color[color_map[c].name]
        ax.scatter(x, y, color=color_val, marker=f'${letter}$',
                   alpha=0.7, s=letter_w_size if letter == 'W' else letter_size,
                   edgecolor='none', zorder=2)

    color_grouped = defaultdict(list)
    for (t, c), (x, y) in centered.items():
        color_grouped[c].append((x, y))
    for c, pts in color_grouped.items():
        x, y      = np.mean(pts, axis=0)
        color_val = color_mapping_color[color_map[c].name]
        ax.scatter(x, y, color=color_val, s=500, alpha=0.2, edgecolor='none', zorder=1)

    _set_limits(ax, xs_all, ys_all, pad=0.15)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.4, zorder=0)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.4, zorder=0)
    prefix = f'Probe ({probe_attr}) — ' if probe_attr else ''
    ax.set_title(f'{prefix}Task × Color (task-centroid subtracted)')
    ax.set_xlabel('Component 1 (task-centered)')
    ax.set_ylabel('Component 2 (task-centered)')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


# ============================
# SAVE HELPERS
# ============================

def _outdir(args, component):
    path = f'thesis_pics/composition/{args.arg_name}/agent_{agent_num_str}'
    os.makedirs(path, exist_ok=True)
    return path

def _epoch_str(epoch):
    return str(epoch).zfill(6)

def _save(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ============================
# MAIN PLOT FUNCTION
# ============================

def plot_single_epoch(plot_dict, component, epoch, probe_attr=None):
    """
    Generate all plots for one epoch.
    If probe_attr is one of 'task', 'color', 'shape', use probe-reduced coords.
    If probe_attr is None, use standard PCA coords.
    """
    args      = plot_dict['args']
    agent_num = 0
    ep        = _epoch_str(epoch)
    outdir    = _outdir(args, component)
    prefix    = f'probe_{probe_attr}_' if probe_attr else ''

    if probe_attr is not None:
        data = meta_probe_reduced_dict[(args.arg_name, agent_num, component)][epoch]
    else:
        data = meta_aligned_data_dict[(args.arg_name, agent_num, component)][epoch, epoch]

    sup_title_base = (
        f'Compositionality with {args.arg_name}\n'
        f'Agent {agent_num_str} • epoch {epoch} • {component}'
        + (f' • probe ({probe_attr})' if probe_attr else ''))

    # --- All attributes ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=dpi, constrained_layout=True)
    fig.suptitle(sup_title_base, fontsize=14)
    plot_all_attributes(ax, data, probe_attr=probe_attr)
    _save(fig, f'{outdir}/{prefix}{component}_all_{ep}.png')

    # --- Marginal plots ---
    for average_over in ['task', 'color', 'shape']:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=dpi, constrained_layout=True)
        fig.suptitle(sup_title_base, fontsize=14)
        plot_marginal(ax, data, average_over=average_over, probe_attr=probe_attr)
        _save(fig, f'{outdir}/{prefix}{component}_avg_{average_over}_{ep}.png')

    # --- Color consistency ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=dpi, constrained_layout=True)
    fig.suptitle(sup_title_base, fontsize=14)
    plot_color_consistency(ax, data, probe_attr=probe_attr)
    _save(fig, f'{outdir}/{prefix}{component}_color_consistency_{ep}.png')

    # --- Single-fixed-attribute subplot grids ---
    for fix_attribute, unique_key, id_map in [
        ('task',  'unique_tasks',  task_map),
        ('color', 'unique_colors', color_map),
        ('shape', 'unique_shapes', shape_map),
    ]:
        unique_vals = data[unique_key]
        n_cols = len(unique_vals)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), dpi=dpi, constrained_layout=True)
        fig.suptitle(f'{sup_title_base}\nFixed {fix_attribute.capitalize()}', fontsize=14)
        for j, fix_value in enumerate(unique_vals):
            ax = axes[j] if n_cols > 1 else axes
            plot_single_attribute_value(ax, data, fix_attribute, fix_value, probe_attr=probe_attr)
        _save(fig, f'{outdir}/{prefix}{component}_fixed_{fix_attribute}_{ep}.png')

    # --- Two-fixed-attribute subplot grids ---
    attribute_pairs = [
        ('task',  'unique_tasks',  task_map,  'color', 'unique_colors', color_map),
        ('task',  'unique_tasks',  task_map,  'shape', 'unique_shapes', shape_map),
        ('color', 'unique_colors', color_map, 'shape', 'unique_shapes', shape_map),
    ]
    for fix_attr_1, unique_key_1, id_map_1, fix_attr_2, unique_key_2, id_map_2 in attribute_pairs:
        unique_vals_1 = data[unique_key_1]
        unique_vals_2 = data[unique_key_2]
        n_rows = len(unique_vals_1)
        n_cols = len(unique_vals_2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=dpi, constrained_layout=True)
        fig.suptitle(
            f'{sup_title_base}\n{fix_attr_1.capitalize()} (rows) × {fix_attr_2.capitalize()} (cols)',
            fontsize=14)
        for i, fix_value_1 in enumerate(unique_vals_1):
            for j, fix_value_2 in enumerate(unique_vals_2):
                ax = axes[i, j] if n_rows > 1 else axes[j]
                plot_two_fixed_attributes(ax, data, fix_attr_1, fix_value_1, fix_attr_2, fix_value_2,
                                          probe_attr=probe_attr)
        _save(fig, f'{outdir}/{prefix}{component}_fixed_{fix_attr_1}_{fix_attr_2}_{ep}.png')


# ============================
# RUN
# ============================
for component in ['command_voice_zq']:
    get_all_data(plot_dict, component)

    # Unsupervised PCA pipeline
    make_all_reducers(plot_dict, component, these_epochs)
    make_all_reduced_data(plot_dict, component)
    make_all_aligned_data(plot_dict, component)

    # Supervised probe pipeline
    make_all_probes(plot_dict, component, these_epochs)

    # Save plots — PCA version and one probe version per attribute
    plot_single_epoch(plot_dict, component, the_epoch, probe_attr=None)
    for probe_attr in ['task', 'color', 'shape']:
        plot_single_epoch(plot_dict, component, the_epoch, probe_attr=probe_attr)

print(f'\nDuration: {duration()}. Done!')
# %%