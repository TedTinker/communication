#%%

import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats
import random
from itertools import product

from utils import args, duration, load_dicts, task_map, color_map, shape_map

print("name:\n{}".format(args.arg_name))

try:
    os.chdir(f"saved_{args.comp}")
except:
    pass
try: os.mkdir("thesis_pics/behavior")
except: pass



def dict_to_ordered_list(d):
    return [d[key] for key in sorted(d.keys())]

tasks = dict_to_ordered_list(task_map)
colors = dict_to_ordered_list(color_map)
shapes = dict_to_ordered_list(shape_map)


def generate_random_string():
    return random.choice([t.char for t in tasks[:2]]) + random.choice([c.char for c in colors[:2]]) + random.choice([s.char for s in shapes[:2]])

example_behaviors = [
    {i: [random.choice([generate_random_string(), generate_random_string(), "AAA"]) for _ in range(random.randint(3, 10))]
     for i in range(10001)} for _ in range(3)]



def behaviors_to_data(behaviors, start_epoch, finish_epoch):
    all_strings = ['AAA'] + [''.join((t.char, c.char, s.char)) for (t, c, s) in product(tasks, colors, shapes)]
    range_keys = range(start_epoch, finish_epoch)  
    string_counts = {string: 0 for string in all_strings}
    total_count = 0
    for i in range_keys:
        for mother_voice in behaviors[i]:
            if(mother_voice.char_text[0] == "A"):
                pass
            else:
                string_counts[mother_voice.char_text] += 1
                total_count += 1
    string_percentages = {key: (count) for key, count in string_counts.items()}
    return(string_percentages)



def create_ranges(start, end, step):
    ranges = []
    for i in range(start, end, step):
        ranges.append((i, min(i + step, end+1)))
    return ranges



def plot_behaviors(plot_dict):
    all_behaviors = plot_dict["behavior"][0]
        
    epoch_ranges = create_ranges(1, 45000, 2000)
    data = {epoch: behaviors_to_data(all_behaviors, start, end) for epoch, (start, end) in enumerate(epoch_ranges)}
    vmax = 0 
    for epoch, d in data.items():
        for key, percentage in d.items():
            if(percentage > vmax):
                vmax = percentage
    
    fig, axes = plt.subplots(len(epoch_ranges), len(tasks), figsize=(3 * len(tasks), 3 * len(epoch_ranges)))

    for row, (start, end) in enumerate(epoch_ranges):
        percentages = data[row]
        for col, task in enumerate(tasks):
            # Create a heatmap grid for each task
            heatmap = np.zeros((len(shapes), len(colors)))
            
            for i, shape in enumerate(shapes):
                for j, color in enumerate(colors):
                    key = task.char + color.char + shape.char  # First letters to match the task_chars, color_chars, shape_chars
                    if(task.name == "SILENCE"):
                        key = "AAA"
                    heatmap[i, j] = percentages.get(key, 0)  # Get the percentage for this combination

            ax = axes[row, col]
            im = ax.imshow(heatmap, vmin=0, vmax=100, cmap='gray_r')  # Darker colors for higher percentages

            if col == 0:
                ax.set_ylabel(f'Epoch {start}-{end}', fontsize=12)

            # Set axis labels for each task in each row (color on x-axis, shape on y-axis)
            if task.name == "SILENCE":
                ax.set_title("NO ACTION", pad=10)
                ax.set_xticks([])  # Remove x-ticks
                ax.set_yticks([])  # Remove y-ticks
                ax.set_xticklabels([])  # Remove x-tick labels
                ax.set_yticklabels([])  # Remove y-tick labels
            else:
                ax.set_title(task.name, pad=10)
                ax.set_xticks(np.arange(len(colors)))
                ax.set_yticks(np.arange(len(shapes)))
                ax.set_xticklabels([c.name for c in colors], rotation=90, fontsize=10)  # Smaller font-size for x labels
                ax.set_yticklabels([s.name for s in shapes], rotation=45, fontsize=10)  # Smaller font-size for y labels

    # Add an overall title
    fig.suptitle('Behavior Analysis', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.savefig(f"thesis_pics/behavior/example.png")
    plt.savefig(f"thesis_pics/behavior/{plot_dict['args'].arg_name}.png")
    


plot_dicts, min_max_dict, complete_order = load_dicts(args)
for plot_dict in plot_dicts:
    plot_behaviors(plot_dict)
print(f"\nDuration: {duration()}. Done!")
# %%