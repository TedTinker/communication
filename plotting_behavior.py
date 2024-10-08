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
    start_index = 0 
    stop_index = start_index + 100 
    while(stop_index < len(all_behaviors[0])):
        start_stop_indexes.append((start_index, stop_index))
        start_index = stop_index 
        stop_index += 100
        

    
    fig, axes = plt.subplots(nrows=len(start_stop_indexes), ncols=len(task_names), figsize=(15, 9))
    for i, (start, stop) in enumerate(start_stop_indexes):
        for j, task in enumerate(task_names):
            data = np.random.rand(len(shape_names), len(color_names)) # Random data, fix it with this.
            #all_these_behaviors = [behavior[start_index:stop_index] for behavior in all_behaviors]
            #all_these_behaviors = sum(all_these_behaviors, [])
            ax = axes[i, j]
            heatmap = ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(color_names)))
            ax.set_xticklabels(color_names, rotation=90)
            ax.set_yticks(np.arange(len(shape_names)))
            ax.set_yticklabels(shape_names)
            if i == 0:  # Set titles only for the top row
                ax.set_title(task)
            if j == 0:
                ax.set_ylabel(f'{start}-{stop}', rotation=0, labelpad=50, va='center')

    plt.tight_layout()
    plt.figsave(f"behavior/{plot_dict["args"].arg_name}.png")
    


plot_dicts, min_max_dict, complete_order = load_dicts(args)
plot_behaviors(plot_dicts)
print(f"\nDuration: {duration()}. Done!")