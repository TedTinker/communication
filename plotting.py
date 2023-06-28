import os
import matplotlib.pyplot as plt 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

import numpy as np
from math import log
from itertools import accumulate

from utils import args, duration, load_dicts, print

print("name:\n{}\n".format(args.arg_name),)



def get_quantiles(plot_dict, name, adjust_xs = True):
    xs = [i for i, x in enumerate(plot_dict[name][0]) if x != None]
    #print("\n\n", name)
    #for agent_record in plot_dict[name]: print(len(agent_record))
    lists = np.array(plot_dict[name], dtype=float)    
    lists = lists[:,xs]
    quantile_dict = {"xs" : [x * plot_dict["args"].keep_data for x in xs] if adjust_xs else xs}
    quantile_dict["min"] = np.min(lists, 0)
    quantile_dict["q10"] = np.quantile(lists, .1, 0)
    quantile_dict["q20"] = np.quantile(lists, .2, 0)
    quantile_dict["q30"] = np.quantile(lists, .3, 0)
    quantile_dict["q40"] = np.quantile(lists, .4, 0)
    quantile_dict["med"] = np.quantile(lists, .5, 0)
    quantile_dict["q60"] = np.quantile(lists, .6, 0)
    quantile_dict["q70"] = np.quantile(lists, .7, 0)
    quantile_dict["q80"] = np.quantile(lists, .8, 0)
    quantile_dict["q90"] = np.quantile(lists, .9, 0)
    quantile_dict["max"] = np.max(lists, 0)
    return(quantile_dict)

def get_logs(quantile_dict):
    for key in quantile_dict.keys():
        if(key != "xs"): quantile_dict[key] = np.log(quantile_dict[key])
    return(quantile_dict)
    


def awesome_plot(here, quantile_dict, color, label, min_max = None, line_transparency = .9, fill_transparency = .1):
    here.fill_between(quantile_dict["xs"], quantile_dict["min"], quantile_dict["max"], color = color, alpha = fill_transparency, linewidth = 0)
    here.fill_between(quantile_dict["xs"], quantile_dict["q10"], quantile_dict["q90"], color = color, alpha = fill_transparency, linewidth = 0)    
    here.fill_between(quantile_dict["xs"], quantile_dict["q20"], quantile_dict["q80"], color = color, alpha = fill_transparency, linewidth = 0)
    here.fill_between(quantile_dict["xs"], quantile_dict["q30"], quantile_dict["q70"], color = color, alpha = fill_transparency, linewidth = 0)
    here.fill_between(quantile_dict["xs"], quantile_dict["q40"], quantile_dict["q60"], color = color, alpha = fill_transparency, linewidth = 0)
    handle, = here.plot(quantile_dict["xs"], quantile_dict["med"], color = color, alpha = line_transparency, label = label)
    if(min_max != None and min_max[0] != min_max[1]): here.set_ylim([min_max[0], min_max[1]])
    return(handle)
    
    
    
def many_min_max(min_max_list):
    mins = [min_max[0] for min_max in min_max_list if min_max[0] != None]
    maxs = [min_max[1] for min_max in min_max_list if min_max[1] != None]
    return((min(mins), max(maxs)))
    


def plots(plot_dicts, min_max_dict):
    fig, axs = plt.subplots(17, len(plot_dicts), figsize = (10*len(plot_dicts), 150))
                
    for i, plot_dict in enumerate(plot_dicts):
        
        row_num = 0
        epochs = plot_dict["args"].epochs
        sums = list(accumulate(epochs))
        percentages = [s / sums[-1] for s in sums][:-1]
        
        def divide_arenas(xs, here = plt):
            if(type(xs) == dict): xs = xs["xs"]
            xs = [xs[int(round(len(xs)*p))] for p in percentages]
            for x in xs: here.axvline(x=x, color = (0,0,0,.2))
    
        # Cumulative rewards
        rew_dict = get_quantiles(plot_dict, "rewards", adjust_xs = False)
        max_reward = max([r for (w,r) in plot_dict["args"].better_reward])
        max_rewards = [max_reward*x for x in range(rew_dict["xs"][-1])]
        min_reward = min([r for (w,r) in plot_dict["args"].default_reward] + [plot_dict["args"].step_lim_punishment]) + plot_dict["args"].wall_punishment
        min_rewards = [min_reward*x for x in range(rew_dict["xs"][-1])]
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        awesome_plot(ax, rew_dict, "turquoise", "Reward")
        ax.axhline(y = 0, color = 'black', linestyle = '--', alpha = .2)
        ax.set_ylabel("Cumulative Reward")
        ax.set_xlabel("Epochs")
        ax.set_title(plot_dict["arg_title"] + "\nCumulative Rewards")
        divide_arenas(rew_dict, ax)
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        def plot_cumulative_rewards_shared_min_max(here):
            awesome_plot(here, rew_dict, "turquoise", "Reward", min_max_dict["rewards"])
            here.axhline(y = 0, color = "black", linestyle = '--', alpha = .2)
            here.plot(max_rewards, color = "black", label = "Max Reward")
            here.plot(min_rewards, color = "black", label = "Min Reward")
            here.set_ylabel("Cumulative Reward")
            here.set_xlabel("Epochs")
            here.set_title(plot_dict["arg_title"] + "\nCumulative Rewards, shared min/max")
            divide_arenas(rew_dict, here)
        plot_cumulative_rewards_shared_min_max(ax)
        fig2, ax2 = plt.subplots(figsize = (10, 10))  
        plot_cumulative_rewards_shared_min_max(ax2)  
        ax2.set_title("Cumulative Rewards")
        fig2.savefig("thesis_pics/rewards_{}.png".format(plot_dict["arg_name"]), bbox_inches = "tight", dpi=300) 
        plt.close(fig2)
            
    
    
        # Ending spot
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        spot_names = np.array([spot_names for spot_names in plot_dict["spot_names"]])
        agents = spot_names.shape[0]
        xs = [x for x in range(spot_names.shape[1])]
        if(plot_dict["args"].hard_maze): 
            kinds = ["NONE"]
            if("t" in plot_dict["args"].maze_list or "1" in plot_dict["args"].maze_list): kinds += ["L", "R"]
            if("2" in plot_dict["args"].maze_list): kinds += ["LL", "LR", "RL", "RR"]
            if("3" in plot_dict["args"].maze_list): kinds += ["LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"]
        else: kinds = ["NONE", "BAD", "GOOD"]
        
        def plot_exits(here):
            for j, kind in enumerate(kinds):
                counts = np.count_nonzero(spot_names == kind, 0)
                counts = [count + (j*agents*1.1) for count in counts]
                here.fill_between(xs, [j*agents*1.1 for _ in xs], counts, color = "black", linewidth = 0)
                if(j != len(kinds)-1):
                    here.plot(xs, [agents*1.05 + j*agents*1.1 for _ in xs], color = "black", linestyle = "--")
            here.set_yticks([(2*j+1)*agents*1.1/2 for j in range(len(kinds))], kinds, rotation='vertical')
            here.tick_params(left = False)
            here.set_ylim([-1, len(kinds)*agents*1.1])
            here.set_ylabel("Chosen Exit")
            here.set_xlabel("Epochs")
            here.set_title(plot_dict["arg_title"] + "\nChosen Exits")
            divide_arenas(xs, here)
        plot_exits(ax)
        fig2, ax2 = plt.subplots(figsize = (10, 10))  
        plot_exits(ax2)  
        ax2.set_title("Chosen Exits")
        fig2.savefig("thesis_pics/exits_{}.png".format(plot_dict["arg_name"]), bbox_inches = "tight", dpi=300) 
        plt.close(fig2)
        
        
        
        # Forward Losses
        accuracy_dict = get_quantiles(plot_dict, "accuracy")
        comp_dict = get_quantiles(plot_dict, "complexity")
        min_max = many_min_max([min_max_dict["accuracy"], min_max_dict["complexity"]])
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        h1 = awesome_plot(ax, accuracy_dict, "green", "Accuracy")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        h2 = awesome_plot(ax, comp_dict, "red",  "Complexity")
        ax.legend(handles = [h1, h2])
        ax.set_title(plot_dict["arg_title"] + "\nForward Losses")
        divide_arenas(accuracy_dict, ax)
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        h1 = awesome_plot(ax, accuracy_dict, "green", "Accuracy", min_max)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        h2 = awesome_plot(ax, comp_dict, "red",  "Complexity", min_max)
        ax.legend(handles = [h1, h2])
        ax.set_title(plot_dict["arg_title"] + "\nForward Losses, shared min/max")
        divide_arenas(accuracy_dict, ax)
        
        
        
        # Log Forward Losses
        log_accuracy_dict = get_logs(accuracy_dict)
        log_comp_dict = get_logs(comp_dict)
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        h1 = awesome_plot(ax, log_accuracy_dict, "green", "log Accuracy")
        ax.set_ylabel("log Loss")
        ax.set_xlabel("Epochs")
        h2 = awesome_plot(ax, log_comp_dict, "red",  "log Complexity")
        ax.legend(handles = [h1, h2])
        ax.set_title(plot_dict["arg_title"] + "\nlog Forward Losses")
        divide_arenas(accuracy_dict, ax)
        
        try:
            min_max = (log(min_max[0]), log(min_max[1]))
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            h1 = awesome_plot(ax, log_accuracy_dict, "green", "log Accuracy", min_max)
            ax.set_ylabel("log Loss")
            ax.set_xlabel("Epochs")
            h2 = awesome_plot(ax, log_comp_dict, "red",  "log Complexity", min_max)
            ax.legend(handles = [h1, h2])
            ax.set_title(plot_dict["arg_title"] + "\nlog Forward Losses, shared min/max")
            divide_arenas(accuracy_dict, ax)
        except: pass
        
        
        
        # Other Losses
        alpha_dict = get_quantiles(plot_dict, "alpha")
        actor_dict = get_quantiles(plot_dict, "actor")
        crit1_dict = get_quantiles(plot_dict, "critic_1")
        crit2_dict = get_quantiles(plot_dict, "critic_2")
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        h1 = awesome_plot(ax, actor_dict, "red", "Actor")
        ax.set_ylabel("Actor Loss")
        ax2 = ax.twinx()
        h2 = awesome_plot(ax2, crit1_dict, "blue", "log Critic")
        awesome_plot(ax2, crit2_dict, "blue", "log Critic")
        ax2.set_ylabel("log Critic Losses")
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.08))
        h3 = awesome_plot(ax3, alpha_dict, "black", "Alpha")
        ax3.set_ylabel("Alpha Loss")
        ax.set_xlabel("Epochs")
        ax.legend(handles = [h1, h2, h3])
        ax.set_title(plot_dict["arg_title"] + "\nOther Losses")
        divide_arenas(crit1_dict, ax)
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        min_max = many_min_max([min_max_dict["critic_1"], min_max_dict["critic_2"]])
        h1 = awesome_plot(ax, actor_dict, "red", "Actor", min_max_dict["actor"])
        ax.set_ylabel("Actor Loss")
        ax.set_xlabel("Epochs")
        ax2 = ax.twinx()
        h2 = awesome_plot(ax2, crit1_dict, "blue", "log Critic", min_max)
        awesome_plot(ax2, crit2_dict, "blue", "log Critic", min_max)
        ax2.set_ylabel("log Critic Losses")
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.08))
        h3 = awesome_plot(ax3, alpha_dict, "black", "Alpha", min_max_dict["alpha"])
        ax3.set_ylabel("Alpha Loss")
        ax.legend(handles = [h1, h2, h3])
        ax.set_title(plot_dict["arg_title"] + "\nOther Losses, shared min/max")
        divide_arenas(crit1_dict, ax)
        
        
        
        # Extrinsic and Intrinsic rewards
        ext_dict = get_quantiles(plot_dict, "extrinsic")
        cur_dict = get_quantiles(plot_dict, "intrinsic_curiosity")
        ent_dict = get_quantiles(plot_dict, "intrinsic_entropy")
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        handles = []
        handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic"))
        ax.set_ylabel("Extrinsic")
        ax.set_xlabel("Epochs")
        if((cur_dict["min"] != cur_dict["max"]).all()):
            ax2 = ax.twinx()
            handles.append(awesome_plot(ax2, cur_dict, "green", "Curiosity"))
            ax2.set_ylabel("Curiosity")
        if((ent_dict["min"] != ent_dict["max"]).all()):
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, ent_dict, "black", "Entropy"))
            ax3.set_ylabel("Entropy")
        ax.legend(handles = handles)
        ax.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards")
        divide_arenas(ext_dict, ax)
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        handles = []
        handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic", min_max_dict["extrinsic"]))
        ax.set_ylabel("Extrinsic")
        ax.set_xlabel("Epochs")
        if((cur_dict["min"] != cur_dict["max"]).all()):
            ax2 = ax.twinx()
            handles.append(awesome_plot(ax2, cur_dict, "green", "Curiosity", min_max_dict["intrinsic_curiosity"]))
            ax2.set_ylabel("Curiosity")
        if((ent_dict["min"] != ent_dict["max"]).all()):
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, ent_dict, "black", "Entropy", min_max_dict["intrinsic_entropy"]))
            ax3.set_ylabel("Entropy")
        ax.legend(handles = handles)
        ax.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared min/max")
        divide_arenas(ext_dict, ax)        
        
        
        # Extrinsic and Intrinsic rewards with same dims
        ext_dict = get_quantiles(plot_dict, "extrinsic")
        cur_dict = get_quantiles(plot_dict, "intrinsic_curiosity")
        ent_dict = get_quantiles(plot_dict, "intrinsic_entropy")
        min_max = many_min_max([min_max_dict["extrinsic"], min_max_dict["intrinsic_curiosity"], min_max_dict["intrinsic_entropy"]])
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        handles = []
        handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic"))
        ax.set_ylabel("Rewards")
        ax.set_xlabel("Epochs")
        if((cur_dict["min"] != cur_dict["max"]).all()):
            handles.append(awesome_plot(ax, cur_dict, "green", "Curiosity"))
        if((ent_dict["min"] != ent_dict["max"]).all()):
            handles.append(awesome_plot(ax, ent_dict, "black", "Entropy"))
        ax.legend(handles = handles)
        ax.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared dims)")
        divide_arenas(ext_dict, ax)
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        handles = []
        handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic", min_max))
        ax.set_ylabel("Rewards")
        ax.set_xlabel("Epochs")
        if((cur_dict["min"] != cur_dict["max"]).all()):
            handles.append(awesome_plot(ax, cur_dict, "green", "Curiosity", min_max))
        if((ent_dict["min"] != ent_dict["max"]).all()):
            handles.append(awesome_plot(ax, ent_dict, "black", "Entropy", min_max))
        ax.legend(handles = handles)
        ax.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared min/max and dim")
        divide_arenas(ext_dict, ax)
        
        
        
        # Curiosities
        naive_dict = get_quantiles(plot_dict, "naive")
        free_dict = get_quantiles(plot_dict, "free")
        min_max = many_min_max([min_max_dict["naive"], min_max_dict["free"]])
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        awesome_plot(ax, naive_dict, "green", "Naive")
        awesome_plot(ax, free_dict, "red", "Free")
        ax.set_ylabel("Curiosity")
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.set_title(plot_dict["arg_title"] + "\nPossible Curiosities")
        divide_arenas(naive_dict, ax)
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        awesome_plot(ax, naive_dict, "green", "Naive", min_max)
        awesome_plot(ax, free_dict, "red", "Free", min_max)
        ax.set_ylabel("Curiosity")
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.set_title(plot_dict["arg_title"] + "\nPossible Curiosities, shared min/max")
        divide_arenas(naive_dict, ax)
        
        
        
        # Log Curiosities
        log_naive_dict = get_logs(naive_dict)
        log_free_dict = get_logs(free_dict)
        
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        awesome_plot(ax, log_naive_dict, "green", "log Naive")
        awesome_plot(ax, log_free_dict, "red", "log Free")
        ax.set_ylabel("log Curiosity")
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.set_title(plot_dict["arg_title"] + "\nlog Possible Curiosities")
        divide_arenas(naive_dict, ax)
        
        try:
            min_max = (log(min_max[0]), log(min_max[1]))
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            awesome_plot(ax, log_naive_dict, "green", "log Naive", min_max)
            awesome_plot(ax, log_free_dict, "red", "log Free", min_max)
            ax.set_ylabel("log Curiosity")
            ax.set_xlabel("Epochs")
            ax.legend()
            ax.set_title(plot_dict["arg_title"] + "\nlog Possible Curiosities, shared min/max")
            divide_arenas(naive_dict, ax)
        except: pass
        
        
        
        print("{}:\t{}.".format(duration(), plot_dict["arg_name"]))

    
    
    # Done!
    fig.tight_layout(pad=1.0)
    plt.savefig("plot.png", bbox_inches = "tight")
    plt.close(fig)
    
    

plot_dicts, min_max_dict, _, _ = load_dicts(args)
plots(plot_dicts, min_max_dict)
print("\nDuration: {}. Done!".format(duration()))