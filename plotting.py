import os
import matplotlib.pyplot as plt 
custom_ls = (0, (3, 5, 1, 5))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

import numpy as np
from math import log
from scipy import interpolate
from itertools import accumulate
from statistics import mode

from utils import args, duration, load_dicts, print, real_names

print("name:\n{}\n".format(args.arg_name),)

dpi = 50



def clean_and_interpolate_nan(arr, xs, non_nan_mask):
    arr = np.array(arr, dtype=np.float64)
    if np.isnan(arr[0]):
        arr[0] = 0.0
    if np.isnan(arr[-1]):
        last_valid = arr[~np.isnan(arr)][-1]
        arr[-1] = last_valid
    nan_mask = np.isnan(arr)
    interpolator = interpolate.interp1d(xs[~nan_mask], arr[~nan_mask], bounds_error=False, fill_value="extrapolate")
    arr[nan_mask] = interpolator(xs[nan_mask])
    return arr



def get_quantiles(plot_dict, name, levels = [1, 2, 3, 4, 5], adjust_xs=True):
    # Convert all None to np.nan in the dataset for uniformity
    max_len = mode([len(agent) for agent in plot_dict[name]])
    lists = np.array([[np.nan if x in [None, "not_used"] else x for x in agent] for agent in plot_dict[name] if len(agent) == max_len], dtype=float)
    non_nan_mask = ~np.isnan(lists).all(axis=0)
    
    # Adjust xs based on whether to adjust them by 'keep_data' multiplier or not
    xs = np.arange(lists.shape[1])
    if adjust_xs and "args" in plot_dict and hasattr(plot_dict["args"], "keep_data"):
        xs = xs * plot_dict["args"].keep_data

    # Prepare quantile_dict with adjusted or original xs
    quantile_dict = {"xs": xs}    
    
    # Calculate quantiles and other statistics only on non-nan values
    quantile_dict["med"] = np.nanquantile(lists, 0.5, axis=0)
    if(1 in levels):
        quantile_dict["q40"] = np.nanquantile(lists, 0.4, axis=0)
        quantile_dict["q60"] = np.nanquantile(lists, 0.6, axis=0)
    if(2 in levels):
        quantile_dict["q30"] = np.nanquantile(lists, 0.3, axis=0)
        quantile_dict["q70"] = np.nanquantile(lists, 0.7, axis=0)
    if(3 in levels):
        quantile_dict["q20"] = np.nanquantile(lists, 0.2, axis=0)
        quantile_dict["q80"] = np.nanquantile(lists, 0.8, axis=0)
    if(4 in levels):
        quantile_dict["q10"] = np.nanquantile(lists, 0.1, axis=0)
        quantile_dict["q90"] = np.nanquantile(lists, 0.9, axis=0)
    if(5 in levels):
        quantile_dict["min"] = np.nanmin(lists, axis=0)
        quantile_dict["max"] = np.nanmax(lists, axis=0)
        
    for key, values in quantile_dict.items():
        if key != "xs":
            nan_mask = np.isnan(values)
            if np.any(nan_mask):
                quantile_dict[key] = clean_and_interpolate_nan(values, xs, non_nan_mask) 

    return quantile_dict



def get_list_quantiles(list_of_lists, plot_dict, levels = [1, 2, 3, 4, 5]):
    quantile_dicts = []
    for layer in range(len(list_of_lists[0])):
        l = [l[layer] for l in list_of_lists]
        max_len = mode(len(sublist) for sublist in l)
        lists = np.array([l[i] for i in range(len(l)) if len(l[i]) == max_len], dtype=float)   
        xs = [i for i, x in enumerate(lists[0])] 
        lists = lists[:,xs]
        quantile_dict = {"xs" : [x * plot_dict["args"].keep_data for x in xs]}
        quantile_dict["med"] = np.quantile(lists, 0.5, axis=0)
        if(1 in levels):
            quantile_dict["q40"] = np.quantile(lists, 0.4, axis=0)
            quantile_dict["q60"] = np.quantile(lists, 0.6, axis=0)
        if(2 in levels):
            quantile_dict["q30"] = np.quantile(lists, 0.3, axis=0)
            quantile_dict["q70"] = np.quantile(lists, 0.7, axis=0)
        if(3 in levels):
            quantile_dict["q20"] = np.quantile(lists, 0.2, axis=0)
            quantile_dict["q80"] = np.quantile(lists, 0.8, axis=0)
        if(4 in levels):
            quantile_dict["q10"] = np.quantile(lists, 0.1, axis=0)
            quantile_dict["q90"] = np.quantile(lists, 0.9, axis=0)
        if(5 in levels):
            quantile_dict["min"] = np.min(lists, axis=0)
            quantile_dict["max"] = np.max(lists, axis=0)
        quantile_dicts.append(quantile_dict)
        
    return(quantile_dicts)



def get_logs(quantile_dict):
    log_quantile_dict = {"xs" : quantile_dict["xs"]}
    for key in quantile_dict.keys():
        if(key != "xs"): log_quantile_dict[key] = np.log(quantile_dict[key])
    return(log_quantile_dict)

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
    


def awesome_plot(here, quantile_dict, color, label, min_max = None, line_transparency = .9, fill_transparency = .1, linestyle = "solid"):
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
        handle, = here.plot(quantile_dict["xs"], quantile_dict["med"], color = color, alpha = line_transparency, label = label, linestyle = linestyle)
    if(min_max != None and min_max[0] != min_max[1]): here.set_ylim([min_max[0], min_max[1]])
    return(handle)
    
    
    
def many_min_max(min_max_list):
    mins = [min_max[0] for min_max in min_max_list if min_max[0] != None]
    maxs = [min_max[1] for min_max in min_max_list if min_max[1] != None]
    return((min(mins), max(maxs)))
    


def plots(plot_dicts, min_max_dict):
    too_many_plot_dicts = len(plot_dicts) > 16
    levels = [1, 2, 3, 4, 5]
    if(not too_many_plot_dicts):
        fig, axs = plt.subplots(34, len(plot_dicts), figsize = (20*len(plot_dicts), 300))                
                
    for i, plot_dict in enumerate(plot_dicts):
        row_num = 0
        if(not too_many_plot_dicts):
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        args = plot_dict["args"]
        print(f"\nStarting {plot_dict['arg_name']}.")
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
                if(key[5:] != "free_play"):
                    action_name_list.append(key[5:])
                    
        fig2, ax2 = plt.subplots(len(action_name_list), 2, figsize = (20, 30))
        fig2.suptitle(plot_dict["arg_title"])  
        fig2_row_num = 0
                    
        for action_name in action_name_list:
            win_dict = get_quantiles(plot_dict, "wins_" + action_name.lower(), levels = levels, adjust_xs = False)
            gen_win_dict = get_quantiles(plot_dict, "gen_wins_" + action_name.lower(), levels = levels, adjust_xs = False)
            for key in gen_win_dict:
                if key not in ["xs"]:
                    gen_win_dict[key] = gen_win_dict[key] 
            
            win_dict = get_rolling_average(win_dict)
            gen_win_dict = get_rolling_average(gen_win_dict)
                
            def plot_rolling_average_wins(here, gen = False):
                awesome_plot(here, gen_win_dict if gen else win_dict, "pink" if gen else "turquoise", "WinRate", (0,1))
                here.set_ylabel((f"Rolling-Average Gen-Win-Rate" if gen else f"Rolling-Average Win-Rate"))
                here.set_xlabel("Epochs")
                here.set_title(plot_dict["arg_title"] + (f"\nRolling-Average Gen-Win-Rate ({action_name})" if gen else f"\nRolling-Average Win-Rate ({action_name})"))
                divide_arenas([x for x in range(sum(epochs))], here)
                    
            if(not too_many_plot_dicts): 
                plot_rolling_average_wins(ax)
                ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
                plot_rolling_average_wins(ax, gen = True)
                ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
                
            plot_rolling_average_wins(ax2[fig2_row_num, 0])  
            ax2[fig2_row_num, 0].set_title(f"Rolling-Average Win-Rate ({action_name})")
            plot_rolling_average_wins(ax2[fig2_row_num, 1], gen = True)  
            ax2[fig2_row_num, 1].set_title(f"Rolling-Average Gen-Win-Rate ({action_name})")
            
            fig2_row_num += 1
            print(f"\tFinished win-rates ({action_name}).")
            
        fig2.savefig(f"thesis_pics/win_rates_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=dpi) 
        plt.close(fig2)
        print(f"\t\tFinished win-rates.")
            
                
                
        # Cumulative rewards
        rew_dict = get_quantiles(plot_dict, "rewards", levels = levels, adjust_xs = False)
        gen_rew_dict = get_quantiles(plot_dict, "gen_rewards", levels = [1], adjust_xs = False)
        
        def plot_cumulative_rewards(here, gen = False, min_max = None):
            awesome_plot(here, gen_rew_dict if gen else rew_dict, "pink" if gen else "turquoise", "Reward", min_max)
            here.axhline(y = 0, color = 'black', linestyle = '--', alpha = .2)
            here.set_ylabel("Cumulative Reward")
            here.set_xlabel("Epochs")
            here.set_title(plot_dict["arg_title"] + "\nCumulative Rewards" + ("\nin Generalization-Tests" if gen else "") + ("" if min_max == None else ", shared min/max"))
            divide_arenas(gen_rew_dict if gen else rew_dict, here)
        
        if(not too_many_plot_dicts): 
            plot_cumulative_rewards(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_cumulative_rewards(ax, min_max = min_max_dict["rewards"])
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_cumulative_rewards(ax, gen = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_cumulative_rewards(ax, gen = True, min_max = min_max_dict["gen_rewards"])
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            
        fig2, ax2 = plt.subplots(4, 1, figsize = (20, 60)) 
        fig2.suptitle(plot_dict["arg_title"])  
        plot_cumulative_rewards(ax2[0])  
        ax2[0].set_title("Cumulative Rewards")
        plot_cumulative_rewards(ax2[1], min_max = min_max_dict["rewards"])  
        ax2[1].set_title("Cumulative Rewards, shared_min/max")
        plot_cumulative_rewards(ax2[2], gen = True)  
        ax2[2].set_title("Cumulative Rewards\nin Generalization-Tests")
        plot_cumulative_rewards(ax2[3], gen = True, min_max = min_max_dict["gen_rewards"])  
        ax2[3].set_title("Cumulative Rewards\nin Generalization-Tests, shared min/max")
        fig2.savefig(f"thesis_pics/rewards_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=dpi) 
        plt.close(fig2)
            
        print(f"\tFinished cumulative rewards.")
        
        
        
        # Forward Losses
        rgbd_dict = get_quantiles(plot_dict, "rgbd_loss", levels = levels)
        comm_dict = get_quantiles(plot_dict, "comm_loss", levels = levels)
        sensors_dict = get_quantiles(plot_dict, "sensors_loss", levels = levels)
        accuracy_dict = get_quantiles(plot_dict, "accuracy", levels = levels)
        comp_dict = get_quantiles(plot_dict, "complexity", levels = levels)
        forward_losses_min_max = many_min_max([min_max_dict["rgbd_loss"], min_max_dict["comm_loss"], min_max_dict["sensors_loss"], min_max_dict["accuracy"]])
        
        log_rgbd_dict = get_logs(rgbd_dict)
        log_comm_dict = get_logs(comm_dict)
        log_sensors_dict = get_logs(sensors_dict)
        log_accuracy_dict = get_logs(accuracy_dict)
        log_comp_dict = get_logs(comp_dict)
        if(forward_losses_min_max[0] == 0):
            log_forward_losses_min_max = (.01, forward_losses_min_max[1])
        else:
            log_forward_losses_min_max = forward_losses_min_max
        log_forward_losses_min_max = (log(log_forward_losses_min_max[0]), log(log_forward_losses_min_max[1]))
            
        def plot_forward_losses(here, log = False, min_max = None):
            handles = []
            handles.append(awesome_plot(here, log_rgbd_dict if log else rgbd_dict, "blue", "RGBD-Loss", min_max))
            handles.append(awesome_plot(here, log_comm_dict if log else comm_dict, "red", "Comm-Loss", min_max))
            handles.append(awesome_plot(here, log_sensors_dict if log else sensors_dict, "orange", "Sensors-Loss", min_max))
            handles.append(awesome_plot(here, log_accuracy_dict if log else accuracy_dict, "purple", "Accuracy", min_max))
            handles.append(awesome_plot(here, log_comp_dict if log else comp_dict, "green",  "Complexity", min_max))
            here.set_ylabel("Loss")
            here.set_xlabel("Epochs")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\n" + ("log " if log else "") + "Forward Losses" + ("" if min_max == None else ", shared min/max"))
            divide_arenas(accuracy_dict, here)
            
        if(not too_many_plot_dicts): 
            plot_forward_losses(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_forward_losses(ax, min_max = forward_losses_min_max)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_forward_losses(ax, log = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_forward_losses(ax, log = True, min_max = log_forward_losses_min_max)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        
        fig2, ax2 = plt.subplots(4, 1, figsize = (20, 60)) 
        fig2.suptitle(plot_dict["arg_title"])  
        plot_forward_losses(ax2[0])  
        ax2[0].set_title("Forward Losses")
        plot_forward_losses(ax2[1], min_max = forward_losses_min_max)  
        ax2[1].set_title("Forward Losses, shared min/max")
        plot_forward_losses(ax2[2], log = True)  
        ax2[2].set_title("log Forward Losses")
        plot_forward_losses(ax2[3], log = True, min_max = log_forward_losses_min_max)  
        ax2[3].set_title("log Forward Losses, shared min/max")
        fig2.savefig(f"thesis_pics/forward_losses_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=dpi) 
        plt.close(fig2)
        
        print(f"\tFinished forward losses.")
            
        
            
        # Other Losses
        alpha_dict = get_quantiles(plot_dict, "alpha", levels = levels)
        actor_dict = get_quantiles(plot_dict, "actor", levels = levels)
        crit_dicts = get_list_quantiles(plot_dict["critics"], plot_dict, levels = levels)
        crit_min_max = many_min_max([crit_min_max for crit_min_max in min_max_dict["critics"]])
        
        def plot_other_losses(here, min_max = False):
            handles = []
            handles.append(awesome_plot(here, actor_dict, "red", "Actor", min_max_dict["actor"] if min_max else None))
            here.set_ylabel("Actor Loss")
            ax2 = here.twinx()
            handles.append(awesome_plot(ax2, crit_dicts[0], "blue", "log Critic 1", crit_min_max if min_max else None ))
            ax2.set_ylabel("log Critic Losses")
            ax3 = here.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, alpha_dict, "black", "Alpha", min_max_dict["alpha"] if min_max else None))
            ax3.set_ylabel("Alpha Loss")
            here.set_xlabel("Epochs")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nActor, Critic Losses" + (", shared min/max" if min_max else ""))
            divide_arenas(actor_dict, here)
            
        if(not too_many_plot_dicts): 
            plot_other_losses(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_other_losses(ax, min_max = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            
        fig2, ax2 = plt.subplots(2, 1, figsize = (20, 30))
        fig2.suptitle(plot_dict["arg_title"])  
        plot_other_losses(ax2[0])  
        ax2[0].set_title("Actor, Critic Losses")
        plot_other_losses(ax2[1], min_max = True)  
        ax2[1].set_title("Actor, Critic Losses, shared min/max")
        fig2.savefig(f"thesis_pics/actor_critic_losses_{action_name.lower()}_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=dpi) 
        plt.close(fig2)
        
        print(f"\tFinished other losses.")
            
        
            
        # Extrinsic and Intrinsic rewards
        these_levels = levels
        keys = \
                ["min", "max"] if 5 in these_levels else \
                ["q10", "q90"] if 4 in these_levels else \
                ["q20", "q80"] if 3 in these_levels else \
                ["q30", "q70"] if 2 in these_levels else \
                ["q40", "q60"]
                
        ext_dict = get_quantiles(plot_dict, "extrinsic", levels = these_levels)
        ent_dict = get_quantiles(plot_dict, "intrinsic_entropy", levels = these_levels)
        cur_dict = get_quantiles(plot_dict, "intrinsic_curiosity", levels = these_levels)
        imi_dict = get_quantiles(plot_dict, "intrinsic_imitation", levels = these_levels)
        rewards_min_max = many_min_max([min_max_dict["extrinsic"], min_max_dict["intrinsic_entropy"], min_max_dict["intrinsic_curiosity"]])
        
        def plot_extrinsic_and_intrinsic_rewards(here, min_max = False):
            handles = []
            handles.append(awesome_plot(here, ext_dict, "red", "Extrinsic", min_max_dict["extrinsic"] if min_max else None))
            here.set_ylabel("Extrinsic")
            here.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).any()):
                ax2 = here.twinx()
                handles.append(awesome_plot(ax2, ent_dict, "black", "Entropy", min_max_dict["intrinsic_entropy"] if min_max else None))
                ax2.set_ylabel("Entropy")
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).any()):
                ax3 = here.twinx()
                ax3.spines["right"].set_position(("axes", 1.08))
                handles.append(awesome_plot(ax3, cur_dict, "green", "Curiosity", min_max_dict["intrinsic_curiosity"] if min_max else None))
                ax3.set_ylabel("Curiosity")
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).any()):
                ax4 = here.twinx()
                ax4.spines["right"].set_position(("axes", 1.16))
                handles.append(awesome_plot(ax4, imi_dict, "blue", "Imitation", min_max_dict["intrinsic_imitation"] if min_max else None))
                ax4.set_ylabel("Imitation")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards" + (", shared min/max" if min_max else ""))
            divide_arenas(ext_dict, here)      
        
        def plot_extrinsic_and_intrinsic_rewards_shared_dim(here, min_max = False):
            handles = []
            handles.append(awesome_plot(here, ext_dict, "red", "Extrinsic", rewards_min_max if min_max else None))
            here.set_ylabel("Rewards")
            here.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).any()):
                handles.append(awesome_plot(here, ent_dict, "black", "Entropy", rewards_min_max if min_max else None))
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).any()):
                handles.append(awesome_plot(here, cur_dict, "green", "Curiosity", rewards_min_max if min_max else None))
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).any()):
                handles.append(awesome_plot(here, imi_dict, "blue", "Imitation", rewards_min_max if min_max else None))
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared dims" + (", shared min/max" if min_max else ""))
            divide_arenas(ext_dict, here)
    
        if(not too_many_plot_dicts): 
            plot_extrinsic_and_intrinsic_rewards(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_extrinsic_and_intrinsic_rewards(ax, min_max = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_extrinsic_and_intrinsic_rewards_shared_dim(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_extrinsic_and_intrinsic_rewards_shared_dim(ax, min_max = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            
        fig2, ax2 = plt.subplots(4, 1, figsize = (20, 60)) 
        fig2.suptitle(plot_dict["arg_title"])  
        plot_extrinsic_and_intrinsic_rewards(ax2[0])  
        ax2[0].set_title("Extrinsic and Intrinsic Rewards")
        plot_extrinsic_and_intrinsic_rewards(ax2[1], min_max = True)  
        ax2[1].set_title("Extrinsic and Intrinsic Rewards, shared min/max")
        plot_extrinsic_and_intrinsic_rewards_shared_dim(ax2[2])  
        ax2[2].set_title("Extrinsic and Intrinsic Rewards, shared dim")
        plot_extrinsic_and_intrinsic_rewards_shared_dim(ax2[3], min_max = True)  
        ax2[3].set_title("Extrinsic and Intrinsic Rewards, shared dim, shared min/max")
        fig2.savefig(f"thesis_pics/extrinsic_and_intrinsic_rewards_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=dpi) 
        plt.close(fig2)
        
        print(f"\tFinished extrinsic and intrinsic rewards.")

        
            
        # Curiosities
        rgbd_prediction_error_dict = get_quantiles(plot_dict, "rgbd_prediction_error_curiosity", levels = [])
        comm_prediction_error_dict = get_quantiles(plot_dict, "comm_prediction_error_curiosity", levels = [])
        sensors_prediction_error_dict = get_quantiles(plot_dict, "sensors_prediction_error_curiosity", levels = [])
        prediction_error_dict = get_quantiles(plot_dict, "prediction_error_curiosity", levels = [])
        
        rgbd_hidden_state_dict = get_quantiles(plot_dict, "rgbd_hidden_state_curiosity", levels = [])
        comm_hidden_state_dict = get_quantiles(plot_dict, "comm_hidden_state_curiosity", levels = [])
        sensors_hidden_state_dict = get_quantiles(plot_dict, "sensors_hidden_state_curiosity", levels = [])
        hidden_state_dict = get_quantiles(plot_dict, "hidden_state_curiosity", levels = [])
        
        curiosity_min_max = many_min_max(
            [min_max_dict["rgbd_prediction_error_curiosity"], 
            min_max_dict["comm_prediction_error_curiosity"], 
            min_max_dict["sensors_prediction_error_curiosity"],
            min_max_dict["prediction_error_curiosity"],
            min_max_dict["rgbd_hidden_state_curiosity"],
            min_max_dict["comm_hidden_state_curiosity"],
            min_max_dict["sensors_hidden_state_curiosity"],
            min_max_dict["hidden_state_curiosity"]])
        
        log_rgbd_prediction_error_dict = get_logs(rgbd_prediction_error_dict)
        log_comm_prediction_error_dict = get_logs(comm_prediction_error_dict)
        log_sensors_prediction_error_dict = get_logs(sensors_prediction_error_dict)
        log_prediction_error_dict = get_logs(prediction_error_dict)
        
        log_rgbd_hidden_state_dict = get_logs(rgbd_hidden_state_dict)
        log_comm_hidden_state_dict = get_logs(comm_hidden_state_dict)
        log_sensors_hidden_state_dict = get_logs(sensors_hidden_state_dict)
        log_hidden_state_dict = get_logs(hidden_state_dict)
        
        if(curiosity_min_max[0] == 0):
            log_curiosity_min_max = (.01, curiosity_min_max[1])
        else:
            log_curiosity_min_max = curiosity_min_max
        log_curiosity_min_max = (log(log_curiosity_min_max[0]), log(log_curiosity_min_max[1]))
        
        def plot_prediction_error_curiosities(here, log = False, min_max = False):
            this_min_max = log_curiosity_min_max if (log and min_max) else curiosity_min_max if min_max else None
            awesome_plot(here, log_prediction_error_dict if log else prediction_error_dict, "green", "Total " + ("log " if log else "") + "Prediction Error", min_max = this_min_max, linestyle = "solid")
            awesome_plot(here, log_rgbd_prediction_error_dict if log else rgbd_prediction_error_dict, "green", "RGBD", min_max = this_min_max, linestyle = "dotted")
            awesome_plot(here, log_comm_prediction_error_dict if log else comm_prediction_error_dict, "green", "Comm", min_max = this_min_max, linestyle = "dashed")
            awesome_plot(here, log_sensors_prediction_error_dict if log else sensors_prediction_error_dict, "green", "Sensors", min_max = this_min_max, linestyle = custom_ls)
            here.set_ylabel("Prediction Error Curiosity")
            here.set_xlabel("Epochs")
            here.legend()
            here.set_title(plot_dict["arg_title"] + "\n" + ("log " if log else "") + "Possible Prediciton Error Curiosities" + (", shaped min_max" if min_max else ""))
            divide_arenas(prediction_error_dict, here)
            
        def plot_hidden_state_curiosities(here, log = False, min_max = False):
            this_min_max = log_curiosity_min_max if (log and min_max) else curiosity_min_max if min_max else None
            awesome_plot(here, log_hidden_state_dict if log else hidden_state_dict, "red", "Total Hidden State", min_max = this_min_max, linestyle = "solid")
            awesome_plot(here, log_rgbd_hidden_state_dict if log else rgbd_hidden_state_dict, "red", "RGBD", min_max = this_min_max, linestyle = "dotted")
            awesome_plot(here, log_comm_hidden_state_dict if log else comm_hidden_state_dict, "red", "Comm", min_max = this_min_max, linestyle = "dashed")
            awesome_plot(here, log_sensors_hidden_state_dict if log else sensors_hidden_state_dict, "red", "Sensors", min_max = this_min_max, linestyle = custom_ls)
            here.set_ylabel("Hidden State Curiosity")
            here.set_xlabel("Epochs")
            here.legend()
            here.set_title(plot_dict["arg_title"] + "\n" + ("log " if log else "") + "Possible Hidden State Curiosities" + (", shaped min_max" if min_max else ""))
            divide_arenas(prediction_error_dict, here)
        
        if(not too_many_plot_dicts): 
            plot_prediction_error_curiosities(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_hidden_state_curiosities(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_prediction_error_curiosities(ax, log = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_hidden_state_curiosities(ax, log = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_prediction_error_curiosities(ax, min_max = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_hidden_state_curiosities(ax, min_max = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_prediction_error_curiosities(ax, log = True, min_max = True)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_hidden_state_curiosities(ax, log = True, min_max = True)
            
        fig2, ax2 = plt.subplots(4, 2, figsize = (40, 60)) 
        fig2.suptitle(plot_dict["arg_title"])  
        plot_prediction_error_curiosities(ax2[0,0])  
        ax2[0,0].set_title("Possible Prediction Error Curiosities")
        plot_hidden_state_curiosities(ax2[1,0])  
        ax2[1,0].set_title("Possible Hidden State Curiosities")
        
        plot_prediction_error_curiosities(ax2[0,1], log = True)  
        ax2[0,1].set_title("Possible log Prediction Error Curiosities")
        plot_hidden_state_curiosities(ax2[1,1], log = True)  
        ax2[1,1].set_title("Possible log Hidden State Curiosities")
        
        plot_prediction_error_curiosities(ax2[2,0], min_max = True)  
        ax2[2,0].set_title("Possible Prediction Error Curiosities, shared min/max")
        plot_hidden_state_curiosities(ax2[3,0], min_max = True)  
        ax2[3,0].set_title("Possible Hidden State Curiosities, shared min/max")
        
        plot_prediction_error_curiosities(ax2[2,1], log = True, min_max = True)  
        ax2[2,1].set_title("Possible log Prediction Error Curiosities, shared min/max")
        plot_hidden_state_curiosities(ax2[3,1], log = True, min_max = True)  
        ax2[3,1].set_title("Possible log Hidden State Curiosities, shared min/max")
        
        fig2.savefig(f"thesis_pics/curiosities_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=dpi) 
        plt.close(fig2)
        
        print(f"\tFinished curiosities.")
        print(f"Finished {plot_dict['arg_name']}.")
        print(f"Duration: {duration()}")

    
    
    # Finish!
    if(not too_many_plot_dicts):
        print("\nSaving full plot...")
        fig.tight_layout(pad=1.0)
        plt.savefig("plot.png", bbox_inches = "tight")
        plt.close(fig)
    
    

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plots(plot_dicts, min_max_dict)
print(f"\nDuration: {duration()}. Done!")