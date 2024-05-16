import os
import matplotlib.pyplot as plt 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

import numpy as np
from math import log
from itertools import accumulate

from utils import args, duration, load_dicts, print, real_names

print("name:\n{}\n".format(args.arg_name),)



def get_quantiles(plot_dict, name, precision = 5, adjust_xs=True, remove_none = True):
    # Convert all None to np.nan in the dataset for uniformity
    lists = np.array([[np.nan if x is None else x for x in agent] for agent in plot_dict[name]], dtype=float)
    
    # Adjust xs based on whether to adjust them by 'keep_data' multiplier or not
    xs = np.arange(lists.shape[1])
    if adjust_xs and "args" in plot_dict and hasattr(plot_dict["args"], "keep_data"):
        xs = xs * plot_dict["args"].keep_data

    # Prepare quantile_dict with adjusted or original xs
    quantile_dict = {"xs": xs}
    
    # Calculate quantiles and other statistics only on non-nan values
    if(precision >= 5):
        quantile_dict["min"] = np.nanmin(lists, axis=0)
        quantile_dict["max"] = np.nanmax(lists, axis=0)
    if(precision >= 4):
        quantile_dict["q10"] = np.nanquantile(lists, 0.1, axis=0)
        quantile_dict["q90"] = np.nanquantile(lists, 0.9, axis=0)
    if(precision >= 3):
        quantile_dict["q20"] = np.nanquantile(lists, 0.2, axis=0)
        quantile_dict["q80"] = np.nanquantile(lists, 0.8, axis=0)
    if(precision >= 2):
        quantile_dict["q30"] = np.nanquantile(lists, 0.3, axis=0)
        quantile_dict["q70"] = np.nanquantile(lists, 0.7, axis=0)
    if(precision >= 1):
        quantile_dict["q40"] = np.nanquantile(lists, 0.4, axis=0)
        quantile_dict["q60"] = np.nanquantile(lists, 0.6, axis=0)
    quantile_dict["med"] = np.nanquantile(lists, 0.5, axis=0)

    
    
    
    
    return quantile_dict

def get_list_quantiles(list_of_lists, plot_dict, precision = 5, remove_none = True):
    quantile_dicts = []
    for layer in range(len(list_of_lists[0])):
        l = [l[layer] for l in list_of_lists]
        if(remove_none):
            xs = [i for i, x in enumerate(l[0]) if x != None]
        else:
            xs = [i for i, x in enumerate(l[0])]
        lists = np.array(l, dtype=float)    
        lists = lists[:,xs]
        quantile_dict = {"xs" : [x * plot_dict["args"].keep_data for x in xs]}
        if(precision >= 5):
            quantile_dict["min"] = np.min(lists, axis=0)
            quantile_dict["max"] = np.max(lists, axis=0)
        if(precision >= 4):
            quantile_dict["q10"] = np.quantile(lists, 0.1, axis=0)
            quantile_dict["q90"] = np.quantile(lists, 0.9, axis=0)
        if(precision >= 3):
            quantile_dict["q20"] = np.quantile(lists, 0.2, axis=0)
            quantile_dict["q80"] = np.quantile(lists, 0.8, axis=0)
        if(precision >= 2):
            quantile_dict["q30"] = np.quantile(lists, 0.3, axis=0)
            quantile_dict["q70"] = np.quantile(lists, 0.7, axis=0)
        if(precision >= 1):
            quantile_dict["q40"] = np.quantile(lists, 0.4, axis=0)
            quantile_dict["q60"] = np.quantile(lists, 0.6, axis=0)
        quantile_dict["med"] = np.quantile(lists, 0.5, axis=0)
        quantile_dicts.append(quantile_dict)
    return(quantile_dicts)

def get_logs(quantile_dict):
    for key in quantile_dict.keys():
        if(key != "xs"): quantile_dict[key] = np.log(quantile_dict[key])
    return(quantile_dict)

def get_rolling_average(quantile_dict, epochs = 100):
    for key, value in quantile_dict.items():
        if(key == "xs"):
            pass 
        else:
            l = quantile_dict[key]
            rolled_l = []
            for i in range(len(l)):
                if(i < epochs):
                    rolled_l.append(sum(l[:i+1])/(i+1))
                else:
                    rolled_l.append(sum(l[i-epochs+1:i+1])/epochs)
            quantile_dict[key] = rolled_l
    return(quantile_dict)
    


def awesome_plot(here, quantile_dict, color, label, min_max = None, line_transparency = .9, fill_transparency = .1):
    keys = list(quantile_dict.keys())
    if("min" in keys and "max" in keys):
        here.fill_between(quantile_dict["xs"], quantile_dict["min"], quantile_dict["max"], color = color, alpha = fill_transparency, linewidth = 0)
    if("q10" in keys and "q90" in keys):
        here.fill_between(quantile_dict["xs"], quantile_dict["q10"], quantile_dict["q90"], color = color, alpha = fill_transparency, linewidth = 0)    
    if("q20" in keys and "q80" in keys):
        here.fill_between(quantile_dict["xs"], quantile_dict["q20"], quantile_dict["q80"], color = color, alpha = fill_transparency, linewidth = 0)
    if("q30" in keys and "q70" in keys):
        here.fill_between(quantile_dict["xs"], quantile_dict["q30"], quantile_dict["q70"], color = color, alpha = fill_transparency, linewidth = 0)
    if("q40" in keys and "q60" in keys):
        here.fill_between(quantile_dict["xs"], quantile_dict["q40"], quantile_dict["q60"], color = color, alpha = fill_transparency, linewidth = 0)
    if("med" in keys):
        handle, = here.plot(quantile_dict["xs"], quantile_dict["med"], color = color, alpha = line_transparency, label = label)
    if(min_max != None and min_max[0] != min_max[1]): here.set_ylim([min_max[0], min_max[1]])
    return(handle)
    
    
    
def many_min_max(min_max_list):
    mins = [min_max[0] for min_max in min_max_list if min_max[0] != None]
    maxs = [min_max[1] for min_max in min_max_list if min_max[1] != None]
    return((min(mins), max(maxs)))
    


def plots(plot_dicts, min_max_dict):
    too_many_plot_dicts = len(plot_dicts) > 25
    figsize = (10, 10)
    if(not too_many_plot_dicts):
        fig, axs = plt.subplots(30, len(plot_dicts), figsize = (20*len(plot_dicts), 300))
                
    for i, plot_dict in enumerate(plot_dicts):
        
        row_num = 0
        args = plot_dict["args"]
        epochs = args.epochs
        sums = list(accumulate(epochs))
        percentages = [s / sums[-1] for s in sums][:-1]
        
        def divide_arenas(xs, here = plt):
            if(type(xs) == dict): xs = xs["xs"]
            xs = [xs[int(round(len(xs)*p))] for p in percentages]
            for x in xs: here.axvline(x=x, color = (0,0,0,.2))
    
        # Rolling win-rate
        action_name_list = []
        for key in plot_dict.keys():
            if(key.startswith("wins_")):
                action_name_list.append(key[5:])
        try:
            action_name_list.remove("none")
        except:
            pass 
        for action_name in action_name_list:
            win_dict = get_quantiles(plot_dict, "wins_" + action_name.lower(), precision = 1, adjust_xs = False, remove_none = False)
            gen_win_dict = get_quantiles(plot_dict, "gen_wins_" + action_name.lower(), precision = 1, adjust_xs = False, remove_none = False)
            
            win_dict = get_rolling_average(win_dict)
            gen_win_dict = get_rolling_average(gen_win_dict)
                
            def plot_rolling_average_wins_shared_min_max(here, gen = False):
                awesome_plot(here, gen_win_dict if gen else win_dict, "pink" if gen else "turquoise", "WinRate", (0,1))
                here.set_ylabel((f"Rolling-Average Gen-Win-Rate" if gen else f"Rolling-Average Win-Rate"))
                here.set_xlabel("Epochs")
                here.set_title(plot_dict["arg_title"] + (f"\nRolling-Average Gen-Win-Rate ({action_name})" if gen else f"\nRolling-Average Win-Rate ({action_name})"))
                divide_arenas([x for x in range(sum(epochs))], here)
                    
            if(not too_many_plot_dicts): 
                ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
                plot_rolling_average_wins_shared_min_max(ax)
                ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
                plot_rolling_average_wins_shared_min_max(ax, gen = True)
                
            fig2, ax2 = plt.subplots(figsize = figsize)  
            plot_rolling_average_wins_shared_min_max(ax2)  
            ax2.set_title("Rolling-Average Win-Rate")
            fig2.savefig("thesis_pics/{}_wins_{}.png".format(action_name.lower(), plot_dict["arg_name"]), bbox_inches = "tight", dpi=300) 
            plt.close(fig2)
            
            fig2, ax2 = plt.subplots(figsize = figsize)  
            plot_rolling_average_wins_shared_min_max(ax2)  
            ax2.set_title("Rolling-Average Gen-Win-Rate")
            fig2.savefig("thesis_pics/{}_gen_wins_{}.png".format(action_name.lower(), plot_dict["arg_name"]), bbox_inches = "tight", dpi=300) 
            plt.close(fig2)
    
        # Cumulative rewards
        rew_dict = get_quantiles(plot_dict, "rewards", precision = 1, adjust_xs = False)
        max_reward = args.reward
        max_rewards = [max_reward*x for x in range(rew_dict["xs"][-1])]
        min_reward = args.step_lim_punishment
        min_rewards = [min_reward*x for x in range(rew_dict["xs"][-1])]
        if(not too_many_plot_dicts):
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
        
        if(not too_many_plot_dicts): plot_cumulative_rewards_shared_min_max(ax)
        fig2, ax2 = plt.subplots(figsize = figsize)  
        plot_cumulative_rewards_shared_min_max(ax2)  
        ax2.set_title("Cumulative Rewards")
        fig2.savefig("thesis_pics/{}_rewards.png".format(plot_dict["arg_name"]), bbox_inches = "tight", dpi=300) 
        plt.close(fig2)
        
        
        
        # Cumulative generalization-test rewards
        gen_rew_dict = get_quantiles(plot_dict, "gen_rewards", precision = 1, adjust_xs = False)
        max_reward = args.reward
        max_rewards = [max_reward*x for x in range(gen_rew_dict["xs"][-1])]
        min_reward = args.step_lim_punishment
        min_rewards = [min_reward*x for x in range(gen_rew_dict["xs"][-1])]
        if(not too_many_plot_dicts):
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            awesome_plot(ax, gen_rew_dict, "pink", "Reward")
            ax.axhline(y = 0, color = 'black', linestyle = '--', alpha = .2)
            ax.set_ylabel("Cumulative Reward")
            ax.set_xlabel("Epochs")
            ax.set_title(plot_dict["arg_title"] + "\nCumulative Rewards\nin Generalization-Tests")
            divide_arenas(gen_rew_dict, ax)
            
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            
        def plot_cumulative_gen_rewards_shared_min_max(here):
            awesome_plot(here, gen_rew_dict, "pink", "Reward", min_max_dict["gen_rewards"])
            here.axhline(y = 0, color = "black", linestyle = '--', alpha = .2)
            here.plot(max_rewards, color = "black", label = "Max Reward")
            here.plot(min_rewards, color = "black", label = "Min Reward")
            here.set_ylabel("Cumulative Reward")
            here.set_xlabel("Epochs")
            here.set_title(plot_dict["arg_title"] + "\nCumulative Rewards\nin Generalization-Tests, shared min/max")
            divide_arenas(gen_rew_dict, here)
        
        if(not too_many_plot_dicts): plot_cumulative_gen_rewards_shared_min_max(ax)
        fig2, ax2 = plt.subplots(figsize = figsize)  
        plot_cumulative_gen_rewards_shared_min_max(ax2)  
        ax2.set_title("Cumulative Rewards\nin Generalization-Tests")
        fig2.savefig("thesis_pics/gen_rewards_{}.png".format(plot_dict["arg_name"]), bbox_inches = "tight", dpi=300) 
        plt.close(fig2)
            
        
        
        if(not too_many_plot_dicts): 
            # Forward Losses
            rgbd_dict = get_quantiles(plot_dict, "rgbd_loss", precision = 1)
            comm_dict = get_quantiles(plot_dict, "comm_loss", precision = 1)
            sensors_dict = get_quantiles(plot_dict, "sensors_loss", precision = 1)
            accuracy_dict = get_quantiles(plot_dict, "accuracy", precision = 1)
            comp_dict = get_quantiles(plot_dict, "complexity", precision = 1)
            min_max = many_min_max([min_max_dict["accuracy"], min_max_dict["complexity"]])
            
            handles = []
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            handles.append(awesome_plot(ax, rgbd_dict, "blue", "RGBD-Loss"))
            handles.append(awesome_plot(ax, comm_dict, "red", "Comm-Loss"))
            handles.append(awesome_plot(ax, sensors_dict, "orange", "Sensors-Loss"))
            handles.append(awesome_plot(ax, accuracy_dict, "purple", "Accuracy"))
            handles.append(awesome_plot(ax, comp_dict, "green",  "Complexity"))
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epochs")
            ax.legend(handles = handles)
            ax.set_title(plot_dict["arg_title"] + "\nForward Losses")
            divide_arenas(accuracy_dict, ax)
            
            handles = []
            min_max = many_min_max([min_max_dict["rgbd_loss"], min_max_dict["comm_loss"], min_max_dict["sensors_loss"], min_max_dict["accuracy"], min_max_dict["complexity"]])
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            handles.append(awesome_plot(ax, rgbd_dict, "blue", "RGBD-Loss", min_max))
            handles.append(awesome_plot(ax, comm_dict, "red", "Comm-Loss", min_max))
            handles.append(awesome_plot(ax, sensors_dict, "orange", "Sensors-Loss", min_max))
            handles.append(awesome_plot(ax, accuracy_dict, "purple", "Accuracy", min_max))
            handles.append(awesome_plot(ax, comp_dict, "green",  "Complexity", min_max))
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epochs")
            ax.legend(handles = handles)
            ax.set_title(plot_dict["arg_title"] + "\nForward Losses, shared min/max")
            divide_arenas(accuracy_dict, ax)
            
            
            
            # Log Forward Losses
            log_rgbd_dict = get_logs(rgbd_dict)
            log_comm_dict = get_logs(comm_dict)
            log_sensors_dict = get_logs(sensors_dict)
            log_accuracy_dict = get_logs(accuracy_dict)
            log_comp_dict = get_logs(comp_dict)
            handles = []
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            handles.append(awesome_plot(ax, log_rgbd_dict, "blue", "RGBD-Loss"))
            handles.append(awesome_plot(ax, log_comm_dict, "red", "Comm-Loss"))
            handles.append(awesome_plot(ax, log_sensors_dict, "orange", "Sensors-Loss"))
            handles.append(awesome_plot(ax, log_accuracy_dict, "purple", "Accuracy"))
            handles.append(awesome_plot(ax, log_comp_dict, "green",  "Complexity"))
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epochs")
            ax.legend(handles = handles)
            ax.set_title(plot_dict["arg_title"] + "\nlog Forward Losses")
            divide_arenas(accuracy_dict, ax)
            
            try:
                handles = []
                min_max = (log(min_max[0]), log(min_max[1]))
                ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
                handles.append(awesome_plot(ax, log_rgbd_dict, "blue", "RGBD-Loss", min_max))
                handles.append(awesome_plot(ax, log_comm_dict, "red", "Comm-Loss", min_max))
                handles.append(awesome_plot(ax, log_sensors_dict, "orange", "Sensors-Loss", min_max))
                handles.append(awesome_plot(ax, log_accuracy_dict, "purple", "Accuracy", min_max))
                handles.append(awesome_plot(ax, log_comp_dict, "green",  "Complexity", min_max))
                ax.set_ylabel("Loss")
                ax.set_xlabel("Epochs")
                ax.legend(handles = handles)
                ax.set_title(plot_dict["arg_title"] + "\nlog Forward Losses, shared min/max")
                divide_arenas(accuracy_dict, ax)
            except: pass
            
            
            
            # Other Losses
            alpha_dict = get_quantiles(plot_dict, "alpha", precision = 1)
            actor_dict = get_quantiles(plot_dict, "actor", precision = 1)
            crit_dicts = get_list_quantiles(plot_dict["critics"], plot_dict, precision = 1)
            
            handles = []
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            handles.append(awesome_plot(ax, actor_dict, "red", "Actor"))
            ax.set_ylabel("Actor Loss")
            ax2 = ax.twinx()
            handles.append(awesome_plot(ax2, crit_dicts[0], "blue", "log Critic 1"))
            ax2.set_ylabel("log Critic Losses")
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, alpha_dict, "black", "Alpha"))
            ax3.set_ylabel("Alpha Loss")
            ax.set_xlabel("Epochs")
            ax.legend(handles = handles)
            ax.set_title(plot_dict["arg_title"] + "\nSensors Losses")
            divide_arenas(actor_dict, ax)
            
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            crit_min_max = many_min_max([crit_min_max for crit_min_max in min_max_dict["critics"]])
            
            handles = []
            handles.append(awesome_plot(ax, actor_dict, "red", "Actor", min_max_dict["actor"]))
            ax.set_ylabel("Actor Loss")
            ax.set_xlabel("Epochs")
            ax2 = ax.twinx()
            #handles.append(awesome_plot(ax2, crit_dicts[0], "blue", "log Critic", crit_min_max))
            ax2.set_ylabel("log Critic Losses")
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, alpha_dict, "black", "Alpha", min_max_dict["alpha"]))
            ax3.set_ylabel("Alpha Loss")
            ax.legend(handles = handles)
            ax.set_title(plot_dict["arg_title"] + "\nSensors Losses, shared min/max")
            divide_arenas(actor_dict, ax)
            
            
            
            # Extrinsic and Intrinsic rewards
            this_precision = 1
            keys = \
                    ["q40", "q60"] if this_precision == 1 else \
                    ["q30", "q70"] if this_precision == 2 else \
                    ["q20", "q80"] if this_precision == 3 else \
                    ["q10", "q90"] if this_precision == 4 else \
                    ["min", "max"]
                    
            ext_dict = get_quantiles(plot_dict, "extrinsic", precision = this_precision)
            ent_dict = get_quantiles(plot_dict, "intrinsic_entropy", precision = this_precision)
            cur_dict = get_quantiles(plot_dict, "intrinsic_curiosity", precision = this_precision)
            imi_dict = get_quantiles(plot_dict, "intrinsic_imitation", precision = this_precision)
            
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            handles = []
            handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic"))
            ax.set_ylabel("Extrinsic")
            ax.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).all()):
                ax2 = ax.twinx()
                handles.append(awesome_plot(ax2, ent_dict, "black", "Entropy"))
                ax2.set_ylabel("Entropy")
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).all()):
                ax3 = ax.twinx()
                ax3.spines["right"].set_position(("axes", 1.08))
                handles.append(awesome_plot(ax3, cur_dict, "green", "Curiosity"))
                ax3.set_ylabel("Curiosity")
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).all()):
                ax4 = ax.twinx()
                ax4.spines["right"].set_position(("axes", 1.16))
                handles.append(awesome_plot(ax4, imi_dict, "blue", "Imitation"))
                ax4.set_ylabel("Imitation")
            ax.legend(handles = handles)
            ax.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards")
            divide_arenas(ext_dict, ax)
            
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            handles = []
            handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic", min_max_dict["extrinsic"]))
            ax.set_ylabel("Extrinsic")
            ax.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).all()):
                ax2 = ax.twinx()
                handles.append(awesome_plot(ax2, ent_dict, "black", "Entropy", min_max_dict["intrinsic_entropy"]))
                ax2.set_ylabel("Entropy")
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).all()):
                ax3 = ax.twinx()
                ax3.spines["right"].set_position(("axes", 1.08))
                handles.append(awesome_plot(ax3, cur_dict, "green", "Curiosity", min_max_dict["intrinsic_curiosity"]))
                ax3.set_ylabel("Curiosity")
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).all()):
                ax4 = ax.twinx()
                ax4.spines["right"].set_position(("axes", 1.16))
                handles.append(awesome_plot(ax4, imi_dict, "blue", "Imitation", min_max_dict["intrinsic_imitation"]))
                ax3.set_ylabel("Imitation")
            ax.legend(handles = handles)
            ax.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared min/max")
            divide_arenas(ext_dict, ax)        
            
            
            # Extrinsic and Intrinsic rewards with same dims
            this_precision = 1
            keys = \
                    ["q40", "q60"] if this_precision == 1 else \
                    ["q30", "q70"] if this_precision == 2 else \
                    ["q20", "q80"] if this_precision == 3 else \
                    ["q10", "q90"] if this_precision == 4 else \
                    ["min", "max"]
            ext_dict = get_quantiles(plot_dict, "extrinsic", precision = this_precision)
            ent_dict = get_quantiles(plot_dict, "intrinsic_entropy", precision = this_precision)
            cur_dict = get_quantiles(plot_dict, "intrinsic_curiosity", precision = this_precision)
            imi_dict = get_quantiles(plot_dict, "intrinsic_imitation", precision = this_precision)
            min_max = many_min_max([min_max_dict["extrinsic"], min_max_dict["intrinsic_entropy"], min_max_dict["intrinsic_curiosity"]])#, min_max_dict["intrinsic_imitation"]])
            
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            handles = []
            handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic"))
            ax.set_ylabel("Rewards")
            ax.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).all()):
                handles.append(awesome_plot(ax, ent_dict, "black", "Entropy"))
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).all()):
                handles.append(awesome_plot(ax, cur_dict, "green", "Curiosity"))
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).all()):
                handles.append(awesome_plot(ax, imi_dict, "blue", "Imitation"))
            ax.legend(handles = handles)
            ax.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared dims")
            divide_arenas(ext_dict, ax)
            
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            handles = []
            handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic", min_max))
            ax.set_ylabel("Rewards")
            ax.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).all()):
                handles.append(awesome_plot(ax, ent_dict, "black", "Entropy", min_max))
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).all()):
                handles.append(awesome_plot(ax, cur_dict, "green", "Curiosity", min_max))
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).all()):
                handles.append(awesome_plot(ax, imi_dict, "blue", "Imitation", min_max))
            ax.legend(handles = handles)
            ax.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared min/max and dim")
            divide_arenas(ext_dict, ax)
            
            
            
            # Curiosities
            prediction_error_dict = get_quantiles(plot_dict, "prediction_error", precision = 1)
            hidden_state_dicts = get_list_quantiles(plot_dict["hidden_state"], plot_dict, precision = 1)
            min_max = many_min_max([min_max_dict["prediction_error"]] + [hidden_state_min_max for hidden_state_min_max in min_max_dict["hidden_state"]])
            
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            awesome_plot(ax, prediction_error_dict, "green", "prediction_error")
            for layer, hidden_state_dict in enumerate(hidden_state_dicts):
                awesome_plot(ax, hidden_state_dict, (1, layer/len(hidden_state_dicts), 0), "hidden_state {}".format(layer+1))
            ax.set_ylabel("Curiosity")
            ax.set_xlabel("Epochs")
            ax.legend()
            ax.set_title(plot_dict["arg_title"] + "\nPossible Curiosities")
            divide_arenas(prediction_error_dict, ax)
            
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            awesome_plot(ax, prediction_error_dict, "green", "prediction_error", min_max)
            for layer, hidden_state_dict in enumerate(hidden_state_dicts):
                awesome_plot(ax, hidden_state_dict, (1, layer/len(hidden_state_dicts), 0), "hidden_state {}".format(layer+1), min_max)
            ax.set_ylabel("Curiosity")
            ax.set_xlabel("Epochs")
            ax.legend()
            ax.set_title(plot_dict["arg_title"] + "\nPossible Curiosities, shared min/max")
            divide_arenas(prediction_error_dict, ax)
            
            
            
            # Log Curiosities
            log_prediction_error_dict = get_logs(prediction_error_dict)
            log_hidden_state_dicts = [get_logs(hidden_state_dict) for hidden_state_dict in hidden_state_dicts]
            
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            awesome_plot(ax, log_prediction_error_dict, "green", "log prediction_error")
            for layer, log_hidden_state_dict in enumerate(log_hidden_state_dicts):
                awesome_plot(ax, log_hidden_state_dict, (1, layer/len(hidden_state_dicts), 0), "log hidden_state {}".format(layer+1))
            ax.set_ylabel("log Curiosity")
            ax.set_xlabel("Epochs")
            ax.legend()
            ax.set_title(plot_dict["arg_title"] + "\nlog Possible Curiosities")
            divide_arenas(prediction_error_dict, ax)
            
            try:
                min_max = (log(min_max[0]), log(min_max[1]))
                ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
                awesome_plot(ax, log_prediction_error_dict, "green", "log prediction_error", min_max)
                for layer, log_hidden_state_dict in enumerate(log_hidden_state_dicts):
                    awesome_plot(ax, log_hidden_state_dict, (1, layer/len(hidden_state_dicts), 0), "log hidden_state {}".format(layer+1), min_max)
                ax.set_ylabel("log Curiosity")
                ax.set_xlabel("Epochs")
                ax.legend()
                ax.set_title(plot_dict["arg_title"] + "\nlog Possible Curiosities, shared min/max")
                divide_arenas(prediction_error_dict, ax)
            except: pass
        
        
        
        print("{}:\t{}.".format(duration(), plot_dict["arg_name"]))

    
    
    # Done!
    if(not too_many_plot_dicts):
        fig.tight_layout(pad=1.0)
        plt.savefig("plot.png", bbox_inches = "tight")
        plt.close(fig)
    
    

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plots(plot_dicts, min_max_dict)
print("\nDuration: {}. Done!".format(duration()))