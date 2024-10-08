import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

from utils import args, duration, load_dicts

print("name:\n{}".format(args.arg_name))

os.chdir(f"saved_{args.comp}")
try: os.mkdir("thesis_pics/behavior")
except: pass



task_names = ['WATCH', 'PUSH', 'PULL', 'LEFT', 'RIGHT']
color_names = ['RED', 'GREEN', 'BLUE', 'CYAN', 'PINK', 'YELLOW']
shape_names = ['PILLAR', 'POLE', 'DUMBBELL', 'DELTA', 'HOURGLASS']



def plot_behaviors(plot_dict):
    all_behaviors = plot_dict["behavior"]
    
    start_stop_indexes = []
    # TO DO: actually save mother_comm
    """start_index = 0 
    stop_index = start_index + 100 
    while(stop_index < len(all_behaviors[0])):
        start_stop_indexes.append((start_index, stop_index))
        start_index = stop_index 
        stop_index += 100"""
        
    start_stop_indexes = [(0, 1000), (1001, 2000), (2001, 3000)]
    
    nrows = len(start_stop_indexes) * 2
    ncols = len(task_names)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 10 * len(start_stop_indexes)))
    
    for i, (start, stop) in enumerate(start_stop_indexes):
        # Remove the axis for all columns in this row (use fig.text for centered label)
        for j in range(ncols):
            ax = axes[i * 2, j]
            ax.axis('off')  # Turn off all axes in the row for the label

        # Place the label above the row of subplots using fig.text
        fig.text(0.5, 1 - (i * 2 + 1) / nrows, f'{start}-{stop}\n\n\n', 
                ha='center', va='center', fontsize=30)
        
        # Plot the data on the next row (i*2 + 1) for the current task and index
        for j, task in enumerate(task_names):
            data = np.random.rand(len(shape_names), len(color_names))  # Replace this with real data later
            
            ax = axes[i * 2 + 1, j]
            heatmap = ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            
            # Set the labels for x and y axis
            ax.set_xticks(np.arange(len(color_names)))
            ax.set_xticklabels(color_names, rotation=90)
            ax.set_yticks(np.arange(len(shape_names)))
            ax.set_yticklabels(shape_names)
            ax.set_title(task)

    plt.tight_layout()
    plt.savefig(f"thesis_pics/behavior/{plot_dict['args'].arg_name}.png")
    


plot_dicts, min_max_dict, complete_order = load_dicts(args)
for plot_dict in plot_dicts:
    plot_behaviors(plot_dict)
print(f"\nDuration: {duration()}. Done!")