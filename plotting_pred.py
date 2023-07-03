import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import torch
from torch.distributions import Normal
import torch.nn.functional as F

from utils import args, duration, load_dicts, print

print("name:\n{}".format(args.arg_name))



def easy_plotting_pred(complete_order, plot_dicts):
    epochs_maze_names = list(set(["_".join(key.split("_")[1:]) for key in plot_dicts[0]["pos_lists"].keys()]))
    epochs_maze_names.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pred_lists"].keys()])) ; agents.sort()
    first_arena_name = plot_dicts[0]["args"].maze_list[0] 
    episodes = len(plot_dicts[0]["pred_lists"]["1_0_{}".format(first_arena_name)])
    
    cmap = plt.cm.get_cmap("gray_r")
    norm = Normalize(vmin = -1, vmax = 1)
    handles = []
    for c in [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]:
        handle = plt.scatter(0, 0, marker = "s", s = 250, facecolor = cmap(norm(c)))
        handles.append(handle)
    plt.close()
        
    for epoch_maze_name in epochs_maze_names:
        epoch, maze_name = epoch_maze_name.split("_")
        for agent in agents:
            for arg_name in complete_order:
                if(arg_name in ["break", "empty_space"]): pass 
                else:
                    for plot_dict in plot_dicts:
                        if(plot_dict["arg_name"] == arg_name): pred_lists = plot_dict["pred_lists"]["{}_{}_{}".format(agent, epoch, maze_name)] ; break 
                    obs_size = 12 + plot_dict["args"].randomness
                    for episode in range(episodes):
                        pred_list = pred_lists[episode]
                        rows = len(pred_list) ; columns = 3 + 2 * plot_dict["args"].samples_per_pred
                        fig, axs = plt.subplots(rows, columns, figsize = (columns * 3, rows * 1.5))
                        title = "Agent {}: Epoch {} (Maze {}), Episode {}".format(agent, epoch, maze_name, episode)
                        fig.suptitle(plot_dict["arg_title"] + "\n" + title, y = 1.1)
                        for row, (action_name, obs, zp_mu_pred, zp_preds, zq_mu_pred, zq_preds) in enumerate(pred_list):
                            for column in range(columns):
                                ax = axs[row, column] ; ax.axis("off")
                                if(row == 0 and column > 0): pass
                                else:                
                                    # Actual obs
                                    if(column == 0):   
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = obs, norm = norm)
                                        ax.set_title("Step {}\nAction: {}".format(row, action_name))
                                    # ZP Mean
                                    elif(column == 1): 
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = torch.tanh(zp_mu_pred), norm = norm)
                                        ax.set_title("ZP Mean")
                                    # ZP Samples
                                    elif(column in [i+2 for i in range(plot_dict["args"].samples_per_pred)]):
                                        pred_num = column - 2
                                        pred = zp_preds[pred_num]
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = torch.tanh(pred), norm = norm)
                                        ax.set_title("ZP Sample {}".format(pred_num+1))
                                    # ZQ Mean
                                    elif(column == 2 + plot_dict["args"].samples_per_pred):
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = torch.tanh(zq_mu_pred), norm = norm)
                                        ax.set_title("ZQ Mean")
                                    # ZQ Samples
                                    else:
                                        pred_num = column - 3 - plot_dict["args"].samples_per_pred
                                        pred = zq_preds[pred_num]
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = torch.tanh(pred), norm = norm)
                                        ax.set_title("ZQ Sample {}".format(pred_num+1))
                        plt.savefig("{}/{}.png".format(arg_name, title), format = "png", bbox_inches = "tight")
                        plt.close()
                        
        print("{}:\tDone with easy epoch {}.".format(duration(), epoch))
                                
                                                
        
def hard_plotting_pred(complete_order, plot_dicts):
    epochs_maze_names = list(set(["_".join(key.split("_")[1:]) for key in plot_dicts[0]["pos_lists"].keys()]))
    epochs_maze_names.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pred_lists"].keys()])) ; agents.sort()
    first_arena_name = plot_dicts[0]["args"].maze_list[0] 
    episodes = len(plot_dicts[0]["pred_lists"]["1_0_{}".format(first_arena_name)])
        
    for epoch_maze_name in epochs_maze_names:
        epoch, maze_name = epoch_maze_name.split("_")
        for agent in agents:
            for arg_name in complete_order:
                if(arg_name in ["break", "empty_space"]): pass 
                else:
                    for plot_dict in plot_dicts:
                        if(plot_dict["arg_name"] == arg_name): pred_lists = plot_dict["pred_lists"]["{}_{}_{}".format(agent, epoch, maze_name)] ; break 
                    for episode in range(episodes):
                        pred_list = pred_lists[episode]
                        rows = len(pred_list) ; columns = 3 + 2 * plot_dict["args"].samples_per_pred
                        fig, axs = plt.subplots(rows, columns, figsize = (columns * 2, rows * 2.5))
                        title = "Agent {}: Epoch {} (Maze {}), Episode {}".format( agent, epoch, maze_name, episode)
                        fig.suptitle(plot_dict["arg_title"] + "\n" + title, y = 1.1)
                        for row, (action_name, (rgbd, spe), ((rgbd_mu_pred_p, pred_rgbd_p), (spe_mu_pred_p, pred_spe_p)), ((rgbd_mu_pred_q, pred_rgbd_q), (spe_mu_pred_q, pred_spe_q))) in enumerate(pred_list):
                            for column in range(columns):
                                ax = axs[row, column] ; ax.axis("off")
                                if(row == 0 and column > 0): pass
                                else:                
                                    steps_per_step = plot_dict["args"].steps_per_step
                                    # Actual obs
                                    if(column == 0):   
                                        ax.imshow(rgbd[:,:,0:3])
                                        ax.set_title("Step {}\nAction: {}\nSpeed {}".format(row, action_name, steps_per_step*round(spe.item())))
                                    # ZP Mean
                                    elif(column == 1): 
                                        ax.imshow(torch.sigmoid(rgbd_mu_pred_p[:,:,0:3])) 
                                        ax.set_title("ZP Mean\nSpeed {}".format(steps_per_step*round(spe_mu_pred_p.item())))
                                    # ZP Samples
                                    elif(column in [i+2 for i in range(plot_dict["args"].samples_per_pred)]):
                                        pred_num = column - 2
                                        pred = pred_rgbd_p[pred_num]
                                        ax.imshow(torch.sigmoid(pred[:,:,0:3]))
                                        ax.set_title("ZP Sample {}\nSpeed {}".format(pred_num+1, steps_per_step*round(pred_spe_p[pred_num].item())))
                                    # ZQ Mean
                                    elif(column == 2 + plot_dict["args"].samples_per_pred):
                                        ax.imshow(torch.sigmoid(rgbd_mu_pred_q[:,:,0:3]))
                                        ax.set_title("ZQ Mean\nSpeed {}".format(steps_per_step*round(spe_mu_pred_q.item())))
                                    # ZQ Samples
                                    else:
                                        pred_num = column - 3 - plot_dict["args"].samples_per_pred
                                        pred = pred_rgbd_q[pred_num]
                                        ax.imshow(torch.sigmoid(pred[:,:,0:3]))
                                        ax.set_title("ZQ Sample {}\nSpeed {}".format(pred_num+1, steps_per_step*round(pred_spe_q[pred_num].item())))
                        plt.savefig("{}/{}.png".format(arg_name, title), format = "png", bbox_inches = "tight")
                        plt.close()
                        
        print("{}:\tDone with hard epoch {}.".format(duration(), epoch))
    


plot_dicts, min_max_dict, complete_order, plot_dicts = load_dicts(args)    
print("\nPlotting predictions.\n")    ; easy_plotting_pred(complete_order, plot_dicts)
print("\nDuration: {}. Done!".format(duration()))