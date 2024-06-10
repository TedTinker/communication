import os
import matplotlib.pyplot as plt 
custom_ls = (0, (3, 5, 1, 5))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

import numpy as np
from math import log
from itertools import accumulate
from statistics import mode

from utils import args, duration, load_dicts, print, real_names

print("name:\n{}\n".format(args.arg_name),)



def get_quantiles(plot_dict, name, levels = [1, 2, 3, 4, 5], adjust_xs=True, remove_none = True):
    # Convert all None to np.nan in the dataset for uniformity
    max_len = mode([len(agent) for agent in plot_dict[name]])
    lists = np.array([[0 if x is None else x for x in agent] for agent in plot_dict[name] if len(agent) == max_len], dtype=float)
    
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

    return quantile_dict

def get_list_quantiles(list_of_lists, plot_dict, levels = [1, 2, 3, 4, 5], remove_none = True):
    quantile_dicts = []
    for layer in range(len(list_of_lists[0])):
        l = [l[layer] for l in list_of_lists]
        max_len = mode(len(sublist) for sublist in l)
        lists = np.array([l[i] for i in range(len(l)) if len(l[i]) == max_len], dtype=float)   
        if(remove_none):
            xs = [i for i, x in enumerate(lists[0]) if x != None]
        else:
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
    too_many_plot_dicts = len(plot_dicts) > 25
    figsize = (10, 10)
    if(not too_many_plot_dicts):
        fig, axs = plt.subplots(30, len(plot_dicts), figsize = (20*len(plot_dicts), 300))
                
    for i, plot_dict in enumerate(plot_dicts):
        
        row_num = 0
        ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
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
                if(key[5:] != "free_play"):
                    action_name_list.append(key[5:])
        for action_name in action_name_list:
            win_dict = get_quantiles(plot_dict, "wins_" + action_name.lower(), levels = [1], adjust_xs = False, remove_none = False)
            gen_win_dict = get_quantiles(plot_dict, "gen_wins_" + action_name.lower(), levels = [1], adjust_xs = False, remove_none = False)
            
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
                
            fig2, ax2 = plt.subplots(2, 1, figsize = (20, 30))
            plot_rolling_average_wins(ax2[0])  
            ax2[0].set_title("Rolling-Average Win-Rate")
            plot_rolling_average_wins(ax2[1], gen = True)  
            ax2[1].set_title("Rolling-Average Gen-Win-Rate")
            fig2.savefig(f"thesis_pics/win_rates_{action_name.lower()}_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=300) 
            plt.close(fig2)
            
            
    
        # Cumulative rewards
        rew_dict = get_quantiles(plot_dict, "rewards", levels = [1], adjust_xs = False)
        max_reward = args.reward
        max_rewards = [max_reward*x for x in range(rew_dict["xs"][-1])]
        min_reward = args.step_lim_punishment
        min_rewards = [min_reward*x for x in range(rew_dict["xs"][-1])]
        
        def plot_cumulative_rewards(here):
            awesome_plot(here, rew_dict, "turquoise", "Reward")
            here.axhline(y = 0, color = 'black', linestyle = '--', alpha = .2)
            here.set_ylabel("Cumulative Reward")
            here.set_xlabel("Epochs")
            here.set_title(plot_dict["arg_title"] + "\nCumulative Rewards")
            divide_arenas(rew_dict, here)
                        
        def plot_cumulative_rewards_shared_min_max(here):
            awesome_plot(here, rew_dict, "turquoise", "Reward", min_max_dict["rewards"])
            here.axhline(y = 0, color = "black", linestyle = '--', alpha = .2)
            here.set_ylabel("Cumulative Reward")
            here.set_xlabel("Epochs")
            here.set_title(plot_dict["arg_title"] + "\nCumulative Rewards, shared min/max")
            divide_arenas(rew_dict, here)
        
        if(not too_many_plot_dicts): 
            plot_cumulative_rewards(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_cumulative_rewards_shared_min_max(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        
        
        
        # Cumulative generalization-test rewards
        gen_rew_dict = get_quantiles(plot_dict, "gen_rewards", levels = [1], adjust_xs = False)
        max_reward = args.reward
        max_rewards = [max_reward*x for x in range(gen_rew_dict["xs"][-1])]
        min_reward = args.step_lim_punishment
        min_rewards = [min_reward*x for x in range(gen_rew_dict["xs"][-1])]
        
        def plot_cumulative_gen_rewards(here):
            awesome_plot(here, gen_rew_dict, "pink", "Reward")
            ax.axhline(y = 0, color = 'black', linestyle = '--', alpha = .2)
            ax.set_ylabel("Cumulative Reward")
            ax.set_xlabel("Epochs")
            ax.set_title(plot_dict["arg_title"] + "\nCumulative Rewards\nin Generalization-Tests")
            divide_arenas(gen_rew_dict, here)
                        
        def plot_cumulative_gen_rewards_shared_min_max(here):
            awesome_plot(here, gen_rew_dict, "pink", "Reward", min_max_dict["gen_rewards"])
            here.axhline(y = 0, color = "black", linestyle = '--', alpha = .2)
            here.set_ylabel("Cumulative Reward")
            here.set_xlabel("Epochs")
            here.set_title(plot_dict["arg_title"] + "\nCumulative Rewards\nin Generalization-Tests, shared min/max")
            divide_arenas(gen_rew_dict, here)
        
        if(not too_many_plot_dicts): 
            plot_cumulative_gen_rewards(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_cumulative_gen_rewards_shared_min_max(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            
        fig2, ax2 = plt.subplots(4, 1, figsize = (20, 60)) 
        plot_cumulative_rewards(ax2[0])  
        ax2[0].set_title("Cumulative Rewards")
        plot_cumulative_rewards_shared_min_max(ax2[1])  
        ax2[1].set_title("Cumulative Rewards, shared_min/max")
        plot_cumulative_gen_rewards(ax2[2])  
        ax2[2].set_title("Cumulative Rewards\nin Generalization-Tests")
        plot_cumulative_gen_rewards_shared_min_max(ax2[3])  
        ax2[3].set_title("Cumulative Rewards\nin Generalization-Tests, shared min/max")
        fig2.savefig(f"thesis_pics/cumulative_rewards_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=300) 
        plt.close(fig2)
            
        
        
        # Forward Losses
        rgbd_dict = get_quantiles(plot_dict, "rgbd_loss", levels = [1])
        comm_dict = get_quantiles(plot_dict, "comm_loss", levels = [1])
        sensors_dict = get_quantiles(plot_dict, "sensors_loss", levels = [1])
        accuracy_dict = get_quantiles(plot_dict, "accuracy", levels = [1])
        comp_dict = get_quantiles(plot_dict, "complexity", levels = [1])
        min_max = many_min_max([min_max_dict["rgbd_loss"], min_max_dict["comm_loss"], min_max_dict["sensors_loss"], min_max_dict["accuracy"]])
            
        def plot_forward_losses(here):
            handles = []
            handles.append(awesome_plot(here, rgbd_dict, "blue", "RGBD-Loss"))
            handles.append(awesome_plot(here, comm_dict, "red", "Comm-Loss"))
            handles.append(awesome_plot(here, sensors_dict, "orange", "Sensors-Loss"))
            handles.append(awesome_plot(here, accuracy_dict, "purple", "Accuracy"))
            handles.append(awesome_plot(here, comp_dict, "green",  "Complexity"))
            here.set_ylabel("Loss")
            here.set_xlabel("Epochs")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nForward Losses")
            divide_arenas(accuracy_dict, here)
            
        def plot_forward_losses_shared_min_max(here):
            handles = []
            handles.append(awesome_plot(here, rgbd_dict, "blue", "RGBD-Loss", min_max))
            handles.append(awesome_plot(here, comm_dict, "red", "Comm-Loss", min_max))
            handles.append(awesome_plot(here, sensors_dict, "orange", "Sensors-Loss", min_max))
            handles.append(awesome_plot(here, accuracy_dict, "purple", "Accuracy", min_max))
            handles.append(awesome_plot(here, comp_dict, "green",  "Complexity", min_max))
            here.set_ylabel("Loss")
            here.set_xlabel("Epochs")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nForward Losses, shared min/max")
            divide_arenas(accuracy_dict, here)
            
        if(not too_many_plot_dicts): 
            plot_forward_losses(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_forward_losses_shared_min_max(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        
        
        
        # Log Forward Losses
        log_rgbd_dict = get_logs(rgbd_dict)
        log_comm_dict = get_logs(comm_dict)
        log_sensors_dict = get_logs(sensors_dict)
        log_accuracy_dict = get_logs(accuracy_dict)
        log_comp_dict = get_logs(comp_dict)
        if(min_max[0] == 0):
            min_max = (.01, min_max[1])
        min_max = (log(min_max[0]), log(min_max[1]))
        
        def plot_log_forward_losses(here):
            handles = []
            handles.append(awesome_plot(here, log_rgbd_dict, "blue", "RGBD-Loss"))
            handles.append(awesome_plot(here, log_comm_dict, "red", "Comm-Loss"))
            handles.append(awesome_plot(here, log_sensors_dict, "orange", "Sensors-Loss"))
            handles.append(awesome_plot(here, log_accuracy_dict, "purple", "Accuracy"))
            handles.append(awesome_plot(here, log_comp_dict, "green",  "Complexity"))
            here.set_ylabel("Loss")
            here.set_xlabel("Epochs")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nlog Forward Losses")
            divide_arenas(accuracy_dict, here)
        
        def plot_log_forward_losses_shared_min_max(here):
            handles = []
            handles.append(awesome_plot(here, log_rgbd_dict, "blue", "RGBD-Loss", min_max))
            handles.append(awesome_plot(here, log_comm_dict, "red", "Comm-Loss", min_max))
            handles.append(awesome_plot(here, log_sensors_dict, "orange", "Sensors-Loss", min_max))
            handles.append(awesome_plot(here, log_accuracy_dict, "purple", "Accuracy", min_max))
            handles.append(awesome_plot(here, log_comp_dict, "green",  "Complexity", min_max))
            here.set_ylabel("Loss")
            here.set_xlabel("Epochs")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nlog Forward Losses, shared min/max")
            divide_arenas(accuracy_dict, here)
        
        if(not too_many_plot_dicts): 
            plot_log_forward_losses(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_log_forward_losses_shared_min_max(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        
        fig2, ax2 = plt.subplots(4, 1, figsize = (20, 60)) 
        plot_forward_losses(ax2[0])  
        ax2[0].set_title("Forward Losses")
        plot_forward_losses_shared_min_max(ax2[1])  
        ax2[1].set_title("Forward Losses, shared min/max")
        plot_log_forward_losses(ax2[2])  
        ax2[2].set_title("log Forward Losses")
        plot_log_forward_losses_shared_min_max(ax2[3])  
        ax2[3].set_title("log Forward Losses, shared min/max")
        fig2.savefig(f"thesis_pics/forward_losses_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=300) 
        plt.close(fig2)
            
        
            
        # Other Losses
        alpha_dict = get_quantiles(plot_dict, "alpha", levels = [1])
        actor_dict = get_quantiles(plot_dict, "actor", levels = [1])
        crit_dicts = get_list_quantiles(plot_dict["critics"], plot_dict, levels = [1])
        crit_min_max = many_min_max([crit_min_max for crit_min_max in min_max_dict["critics"]])
        
        def plot_other_losses(here):
            handles = []
            handles.append(awesome_plot(here, actor_dict, "red", "Actor"))
            here.set_ylabel("Actor Loss")
            ax2 = here.twinx()
            handles.append(awesome_plot(ax2, crit_dicts[0], "blue", "log Critic 1"))
            ax2.set_ylabel("log Critic Losses")
            ax3 = here.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, alpha_dict, "black", "Alpha"))
            ax3.set_ylabel("Alpha Loss")
            here.set_xlabel("Epochs")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nActor, Critic Losses")
            divide_arenas(actor_dict, here)
        
        
        def plot_other_losses_shared_min_max(here):
            handles = []
            handles.append(awesome_plot(here, actor_dict, "red", "Actor", min_max_dict["actor"]))
            here.set_ylabel("Actor Loss")
            here.set_xlabel("Epochs")
            ax2 = here.twinx()
            handles.append(awesome_plot(ax2, crit_dicts[0], "blue", "log Critic 1", crit_min_max))
            ax2.set_ylabel("log Critic Losses")
            ax3 = here.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, alpha_dict, "black", "Alpha", min_max_dict["alpha"]))
            ax3.set_ylabel("Alpha Loss")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nActor, Critic Losses, shared min/max")
            divide_arenas(actor_dict, here)
            
        if(not too_many_plot_dicts): 
            plot_other_losses(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_other_losses_shared_min_max(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            
        fig2, ax2 = plt.subplots(2, 1, figsize = (20, 30))
        plot_other_losses(ax2[0])  
        ax2[0].set_title("Actor, Critic Losses")
        plot_other_losses_shared_min_max(ax2[1])  
        ax2[1].set_title("Actor, Critic Losses, shared min/max")
        fig2.savefig(f"thesis_pics/actor_critic_losses_{action_name.lower()}_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=300) 
        plt.close(fig2)
            
            
            
        # Extrinsic and Intrinsic rewards
        these_levels = [1]
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
        
        def plot_extrinsic_and_intrinsic_rewards(here):
            handles = []
            handles.append(awesome_plot(here, ext_dict, "red", "Extrinsic"))
            here.set_ylabel("Extrinsic")
            here.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).all()):
                ax2 = here.twinx()
                handles.append(awesome_plot(ax2, ent_dict, "black", "Entropy"))
                ax2.set_ylabel("Entropy")
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).all()):
                ax3 = here.twinx()
                ax3.spines["right"].set_position(("axes", 1.08))
                handles.append(awesome_plot(ax3, cur_dict, "green", "Curiosity"))
                ax3.set_ylabel("Curiosity")
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).all()):
                ax4 = here.twinx()
                ax4.spines["right"].set_position(("axes", 1.16))
                handles.append(awesome_plot(ax4, imi_dict, "blue", "Imitation"))
                ax4.set_ylabel("Imitation")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards")
            divide_arenas(ext_dict, here)
            
        def plot_extrinsic_and_intrinsic_rewards_shared_min_max(here):
            handles = []
            handles.append(awesome_plot(here, ext_dict, "red", "Extrinsic", min_max_dict["extrinsic"]))
            here.set_ylabel("Extrinsic")
            here.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).all()):
                ax2 = here.twinx()
                handles.append(awesome_plot(ax2, ent_dict, "black", "Entropy", min_max_dict["intrinsic_entropy"]))
                ax2.set_ylabel("Entropy")
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).all()):
                ax3 = here.twinx()
                ax3.spines["right"].set_position(("axes", 1.08))
                handles.append(awesome_plot(ax3, cur_dict, "green", "Curiosity", min_max_dict["intrinsic_curiosity"]))
                ax3.set_ylabel("Curiosity")
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).all()):
                ax4 = here.twinx()
                ax4.spines["right"].set_position(("axes", 1.16))
                handles.append(awesome_plot(ax4, imi_dict, "blue", "Imitation", min_max_dict["intrinsic_imitation"]))
                ax3.set_ylabel("Imitation")
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared min/max")
            divide_arenas(ext_dict, here)        
        
        if(not too_many_plot_dicts): 
            plot_extrinsic_and_intrinsic_rewards(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_extrinsic_and_intrinsic_rewards_shared_min_max(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        
        
        
        # Extrinsic and Intrinsic rewards with same dims
        these_levels = [1]
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
        min_max = many_min_max([min_max_dict["extrinsic"], min_max_dict["intrinsic_entropy"], min_max_dict["intrinsic_curiosity"]])#, min_max_dict["intrinsic_imitation"]])
        
        def plot_extrinsic_and_intrinsic_rewards_shared_dim(here):
            handles = []
            handles.append(awesome_plot(here, ext_dict, "red", "Extrinsic"))
            here.set_ylabel("Rewards")
            here.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).all()):
                handles.append(awesome_plot(here, ent_dict, "black", "Entropy"))
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).all()):
                handles.append(awesome_plot(here, cur_dict, "green", "Curiosity"))
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).all()):
                handles.append(awesome_plot(here, imi_dict, "blue", "Imitation"))
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared dims")
            divide_arenas(ext_dict, here)
        
        def plot_extrinsic_and_intrinsic_rewards_shared_dim_shared_min_max(here):
            handles = []
            handles.append(awesome_plot(here, ext_dict, "red", "Extrinsic", min_max))
            here.set_ylabel("Rewards")
            here.set_xlabel("Epochs")
            if((ent_dict[keys[0]] != ent_dict[keys[1]]).all()):
                handles.append(awesome_plot(here, ent_dict, "black", "Entropy", min_max))
            if((cur_dict[keys[0]] != cur_dict[keys[1]]).all()):
                handles.append(awesome_plot(here, cur_dict, "green", "Curiosity", min_max))
            if((imi_dict[keys[0]] != imi_dict[keys[1]]).all()):
                handles.append(awesome_plot(here, imi_dict, "blue", "Imitation", min_max))
            here.legend(handles = handles)
            here.set_title(plot_dict["arg_title"] + "\nExtrinsic and Intrinsic Rewards, shared min/max and dim")
            divide_arenas(ext_dict, here)
            
        if(not too_many_plot_dicts): 
            plot_extrinsic_and_intrinsic_rewards_shared_dim(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_extrinsic_and_intrinsic_rewards_shared_dim_shared_min_max(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            
        fig2, ax2 = plt.subplots(4, 1, figsize = (20, 60)) 
        plot_extrinsic_and_intrinsic_rewards(ax2[0])  
        ax2[0].set_title("Extrinsic and Intrinsic Rewards")
        plot_extrinsic_and_intrinsic_rewards_shared_min_max(ax2[1])  
        ax2[1].set_title("Extrinsic and Intrinsic Rewards, shared min/max")
        plot_extrinsic_and_intrinsic_rewards_shared_dim(ax2[2])  
        ax2[2].set_title("Extrinsic and Intrinsic Rewards, shared dim")
        plot_extrinsic_and_intrinsic_rewards_shared_dim_shared_min_max(ax2[3])  
        ax2[3].set_title("Extrinsic and Intrinsic Rewards, shared min/max and dim")
        fig2.savefig(f"thesis_pics/extrinsic_and_intrinsic_rewards_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=300) 
        plt.close(fig2)


            
        # Curiosities
        rgbd_prediction_error_curiosity_dict = get_quantiles(plot_dict, "rgbd_prediction_error_curiosity", levels = [])
        comm_prediction_error_curiosity_dict = get_quantiles(plot_dict, "comm_prediction_error_curiosity", levels = [])
        sensors_prediction_error_curiosity_dict = get_quantiles(plot_dict, "sensors_prediction_error_curiosity", levels = [])
        prediction_error_curiosity_dict = get_quantiles(plot_dict, "prediction_error_curiosity", levels = [])
        
        rgbd_hidden_state_curiosity_dict = get_quantiles(plot_dict, "rgbd_hidden_state_curiosity", levels = [])
        comm_hidden_state_curiosity_dict = get_quantiles(plot_dict, "comm_hidden_state_curiosity", levels = [])
        sensors_hidden_state_curiosity_dict = get_quantiles(plot_dict, "sensors_hidden_state_curiosity", levels = [])
        hidden_state_curiosity_dict = get_quantiles(plot_dict, "hidden_state_curiosity", levels = [])
        
        min_max = many_min_max(
            [min_max_dict["rgbd_prediction_error_curiosity"]] + 
            [min_max_dict["comm_prediction_error_curiosity"]] + 
            [min_max_dict["sensors_prediction_error_curiosity"]] + 
            [min_max_dict["prediction_error_curiosity"]] + 
            [min_max_dict["rgbd_hidden_state_curiosity"]] + 
            [min_max_dict["comm_hidden_state_curiosity"]] +
            [min_max_dict["sensors_hidden_state_curiosity"]] +
            [min_max_dict["hidden_state_curiosity"]])
        
        def plot_curiosities(here):
            awesome_plot(here, prediction_error_curiosity_dict, "green", "prediction_error", linestyle = "solid")
            awesome_plot(here, rgbd_prediction_error_curiosity_dict, "green", "rgbd_prediction_error", linestyle = "dotted")
            awesome_plot(here, comm_prediction_error_curiosity_dict, "green", "comm_prediction_error", linestyle = "dashed")
            awesome_plot(here, sensors_prediction_error_curiosity_dict, "green", "sensors_prediction_error", linestyle = custom_ls)
            awesome_plot(here, hidden_state_curiosity_dict, "red", "hidden_state", linestyle = "solid")
            awesome_plot(here, rgbd_hidden_state_curiosity_dict, "red", "rgbd_hidden_state", linestyle = "dotted")
            awesome_plot(here, comm_hidden_state_curiosity_dict, "red", "comm_hidden_state", linestyle = "dashed")
            awesome_plot(here, sensors_hidden_state_curiosity_dict, "red", "sensors_hidden_state", linestyle = custom_ls)
            here.set_ylabel("Curiosity")
            here.set_xlabel("Epochs")
            here.legend()
            here.set_title(plot_dict["arg_title"] + "\nPossible Curiosities")
            divide_arenas(prediction_error_curiosity_dict, here)
            
        
        def plot_curiosities_shared_min_max(here):
            awesome_plot(here, prediction_error_curiosity_dict, "green", "prediction_error", min_max = min_max, linestyle = "solid")
            awesome_plot(here, rgbd_prediction_error_curiosity_dict, "green", "rgbd_prediction_error", min_max = min_max, linestyle = "dotted")
            awesome_plot(here, comm_prediction_error_curiosity_dict, "green", "comm_prediction_error", min_max = min_max, linestyle = "dashed")
            awesome_plot(here, sensors_prediction_error_curiosity_dict, "green", "sensors_prediction_error", min_max = min_max, linestyle = custom_ls)
            awesome_plot(here, hidden_state_curiosity_dict, "red", "hidden_state", min_max = min_max, linestyle = "solid")
            awesome_plot(here, rgbd_hidden_state_curiosity_dict, "red", "rgbd_hidden_state", min_max = min_max, linestyle = "dotted")
            awesome_plot(here, comm_hidden_state_curiosity_dict, "red", "comm_hidden_state", min_max = min_max, linestyle = "dashed")
            awesome_plot(here, sensors_hidden_state_curiosity_dict, "red", "sensors_hidden_state", min_max = min_max, linestyle = custom_ls)
            here.set_ylabel("Curiosity")
            here.set_xlabel("Epochs")
            here.legend()
            here.set_title(plot_dict["arg_title"] + "\nPossible Curiosities, shared min/max")
            divide_arenas(prediction_error_curiosity_dict, here)
        
        if(not too_many_plot_dicts): 
            plot_curiosities(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_curiosities_shared_min_max(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
        
        
        
        # Log Curiosities
        log_rgbd_prediction_error_dict = get_logs(rgbd_prediction_error_curiosity_dict)
        log_comm_prediction_error_dict = get_logs(comm_prediction_error_curiosity_dict)
        log_sensors_prediction_error_dict = get_logs(sensors_prediction_error_curiosity_dict)
        log_prediction_error_dict = get_logs(prediction_error_curiosity_dict)
        
        log_rgbd_hidden_state_dict = get_logs(rgbd_hidden_state_curiosity_dict)
        log_comm_hidden_state_dict = get_logs(comm_hidden_state_curiosity_dict)
        log_sensors_hidden_state_dict = get_logs(sensors_hidden_state_curiosity_dict)
        log_hidden_state_dict = get_logs(hidden_state_curiosity_dict)
        
        if(min_max[0] == 0):
            min_max = (.01, min_max[1])
        min_max = (log(min_max[0]), log(min_max[1]))
        
        def plot_log_curiosities(here):
            awesome_plot(here, log_prediction_error_dict, "green", "log prediction_error", linestyle = "solid")
            awesome_plot(here, log_rgbd_prediction_error_dict, "green", "log rgbd prediction_error", linestyle = "dotted")
            awesome_plot(here, log_comm_prediction_error_dict, "green", "log comm prediction_error", linestyle = "dashed")
            awesome_plot(here, log_sensors_prediction_error_dict, "green", "log sensors prediction_error", linestyle = custom_ls)
            awesome_plot(here, log_hidden_state_dict, "red", "log hidden_state", linestyle = "solid")
            awesome_plot(here, log_rgbd_hidden_state_dict, "red", "log rgbd hidden_state", linestyle = "dotted")
            awesome_plot(here, log_comm_hidden_state_dict, "red", "log comm hidden_state", linestyle = "dashed")
            awesome_plot(here, log_sensors_hidden_state_dict, "red", "log sensors hidden_state", linestyle = custom_ls)
            here.set_ylabel("log Curiosity")
            here.set_xlabel("Epochs")
            here.legend()
            here.set_title(plot_dict["arg_title"] + "\nlog Possible Curiosities")
            divide_arenas(prediction_error_curiosity_dict, here)
        
        def plot_log_curiosities_shared_min_max(here):
            awesome_plot(here, log_prediction_error_dict, "green", "log prediction_error", min_max = min_max, linestyle = "solid")
            awesome_plot(here, log_rgbd_prediction_error_dict, "green", "log rgbd prediction_error", min_max = min_max, linestyle = "dotted")
            awesome_plot(here, log_comm_prediction_error_dict, "green", "log comm prediction_error", min_max = min_max, linestyle = "dashed")
            awesome_plot(here, log_sensors_prediction_error_dict, "green", "log sensors prediction_error", min_max = min_max, linestyle = custom_ls)
            awesome_plot(here, log_hidden_state_dict, "red", "log hidden_state", min_max = min_max, linestyle = "solid")
            awesome_plot(here, log_rgbd_hidden_state_dict, "red", "log rgbd hidden_state", min_max = min_max, linestyle = "dotted")
            awesome_plot(here, log_comm_hidden_state_dict, "red", "log comm hidden_state", min_max = min_max, linestyle = "dashed")
            awesome_plot(here, log_sensors_hidden_state_dict, "red", "log sensors hidden_state", min_max = min_max, linestyle = custom_ls)
            here.set_ylabel("log Curiosity")
            here.set_xlabel("Epochs")
            here.legend()
            here.set_title(plot_dict["arg_title"] + "\nlog Possible Curiosities, shared min/max")
            divide_arenas(prediction_error_curiosity_dict, here)
        
    
        if(not too_many_plot_dicts): 
            plot_log_curiosities(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            plot_log_curiosities_shared_min_max(ax)
            ax = axs[row_num,i] if len(plot_dicts) > 1 else axs[row_num] ; row_num += 1
            
        fig2, ax2 = plt.subplots(4, 1, figsize = (20, 60)) 
        plot_curiosities(ax2[0])  
        ax2[0].set_title("Possible Curiosities")
        plot_curiosities_shared_min_max(ax2[1])  
        ax2[1].set_title("Possible Curiosities, shared min/max")
        plot_log_curiosities(ax2[2])  
        ax2[2].set_title("log Possible Curiosities")
        plot_log_curiosities_shared_min_max(ax2[3])  
        ax2[3].set_title("log Possible Curiosities, shared min/max")
        fig2.savefig(f"thesis_pics/curiosities_{plot_dict['arg_name']}.png", bbox_inches = "tight", dpi=300) 
        plt.close(fig2)
        
    print("{}:\t{}.".format(duration(), plot_dict["arg_name"]))

    
    
    # Done!
    if(not too_many_plot_dicts):
        fig.tight_layout(pad=1.0)
        plt.savefig("plot.png", bbox_inches = "tight")
        plt.close(fig)
    
    

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plots(plot_dicts, min_max_dict)
print("\nDuration: {}. Done!".format(duration()))