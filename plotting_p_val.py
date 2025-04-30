import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

from utils import args, duration, load_dicts

print("name:\n{}".format(args.arg_name))

os.chdir(f"saved_{args.comp}")
try: os.mkdir("thesis_pics/p_values")
except: pass

plot_dicts, min_max_dict, complete_order = load_dicts(args)



arg_names = []
values_to_plot = {}
for plot_dict in plot_dicts:
    arg_name = plot_dict["args"].arg_name
    arg_names.append(arg_name)
    values_to_plot[arg_name] = {}
    
task_names = []
for key in plot_dicts[0].keys():
    if(key.startswith("wins_")):
        if(key[5:] != "free_play"):
            task_names.append(key[5:])
            
for plot_dict in plot_dicts:
    args = plot_dict["args"]

    # Final win-rates                 
    for task_name in task_names:
        wins = plot_dict["wins_" + task_name]
        wins = np.array(wins)
        wins = wins[:,-1]
        values_to_plot[args.arg_name]["wins_" + task_name] = wins
        
        gen_wins = plot_dict["gen_wins_" + task_name]
        gen_wins = np.array(gen_wins)  
        gen_wins = gen_wins[:,-1]
        values_to_plot[args.arg_name]["gen_wins_" + task_name] = wins
        
    # reward
    reward = plot_dict["reward"]
    reward = np.array(reward)
    reward = reward[:,-1]
    values_to_plot[args.arg_name]["reward"] = reward
    
    gen_reward = plot_dict["gen_reward"]
    gen_reward= np.array(gen_reward)   
    gen_reward = gen_reward[:,-1]
    values_to_plot[args.arg_name]["gen_reward"] = gen_reward



num_args = len(values_to_plot.keys())
value_names = []
for value_name in list(values_to_plot.values())[0].keys():
    value_names.append(value_name)
    
    

def compare_and_plot(values_1, values_2, args_name_1, args_name_2, here, data_type='boolean', confidence=0.9):
    n1, n2 = len(values_1), len(values_2)
    alpha = 1 - confidence
    z_value = stats.norm.ppf(1 - alpha / 2)  # Z-value for the confidence interval (two-tailed)

    if data_type == 'boolean':
        mean_1, mean_2 = np.mean(values_1), np.mean(values_2)
        ci_1 = z_value * np.sqrt((mean_1 * (1 - mean_1)) / n1)
        ci_2 = z_value * np.sqrt((mean_2 * (1 - mean_2)) / n2)
        count = np.array([np.sum(values_1), np.sum(values_2)])
        nobs = np.array([n1, n2])
        stat, p_value = proportions_ztest(count, nobs)

    elif data_type == 'numeric':
        mean_1, mean_2 = np.mean(values_1), np.mean(values_2)
        se_1, se_2 = stats.sem(values_1), stats.sem(values_2)
        t_value = stats.t.ppf(1 - alpha / 2, df=min(n1, n2) - 1)
        ci_1 = t_value * se_1
        ci_2 = t_value * se_2
        stat, p_value = stats.ttest_ind(values_1, values_2)

    if p_value < alpha:  # Significant difference
        if mean_1 > mean_2:
            color_1, color_2 = 'green', 'red'  # values_1 is better
        else:
            color_1, color_2 = 'red', 'green'  # values_2 is better
    else:  # No significant difference
        color_1, color_2 = 'white', 'white'

    print(f"\t\tMeans: {mean_1}, {mean_2}")
    print(f"\t\tConf: {ci_1}, {ci_2}")
    print(f"\t\tColor: {color_1}, {color_2}")

    here.bar(
        [f'{args_name_1}', f'{args_name_2}'],
        [mean_1, mean_2],
        yerr=[ci_1, ci_2],
        color=[color_1, color_2],
        edgecolor='black',
        capsize=10
    )

    if data_type == 'boolean':
        here.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
        here.set_yticklabels([f'{int(tick*100)}%' for tick in here.get_yticks()])  # Format y-ticks as percentages

    
    

for value_name in value_names:
    print(f"\n{value_name}")
    #fig, axes = plt.subplots(num_args, num_args, figsize = (10, 10))
    fig, axes = plt.subplots(num_args-1, num_args-1, figsize = (10, 10))
    fig.suptitle(f'{value_name}')
    for (row, args_name_1), (column, args_name_2) in product(enumerate(values_to_plot.keys()), enumerate(values_to_plot.keys())):
        print(f"row {row} column {column}: {args_name_1}, {args_name_2}")
        if(row == len(values_to_plot)-1 or column == 0):
            print("passing")
        else:
            print("in plot")
            ax = axes[row, column-1]
            if(row >= column):
                print(f"\t row, column {row}, {column}: REMOVING AXIS")
                ax.axis("off")
            else:
                print(f"\t row, column {row}, {column}: {args_name_1} vs {args_name_2}")
                value_1 = values_to_plot[args_name_1][value_name]
                value_2 = values_to_plot[args_name_2][value_name]
                value_1 = np.array([v for v in values_to_plot[args_name_1][value_name] if v is not None])
                value_2 = np.array([v for v in values_to_plot[args_name_2][value_name] if v is not None])
                compare_and_plot(value_1, value_2, args_name_1, args_name_2, ax, data_type = "boolean" if "win" in value_name else "numeric")            
    fig.tight_layout()
    plt.savefig(f"thesis_pics/p_values/{value_name}_p_values.png")
    plt.close()



print(f"\nDuration: {duration()}. Done!")
