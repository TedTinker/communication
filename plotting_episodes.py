#%% 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
import re
import imageio
import numpy as np

from utils import print, args, duration, load_dicts
from pybullet_data.robot_maker import how_to_plot_sensors



def plot_episodes(complete_order, plot_dicts):
    global args
    for arg_name in complete_order:
        if(arg_name in ["break", "empty_space"]): 
            pass 
        else:
            for plot_dict in plot_dicts:
                if(plot_dict["arg_name"] == arg_name):
                    episode_dicts = plot_dict["episode_dicts"]
                    args = plot_dict["args"]
                    for key, episode_dict in episode_dicts.items():
                        plot_episode(key, episode_dict, arg_name)
                        
                        
                        
def plot_episode(key, episode_dict, arg_name, saving = True):
    if(saving):
        agent_num, epoch, episode_num, swapping = key.split("_")
        try:
            os.mkdir(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}_swapping_{swapping}")
        except: 
            pass
        os.chdir(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}_swapping_{swapping}")
        print("Saving {}: agent {}, epoch {}, episode {}.{}".format(arg_name, agent_num, epoch, episode_num, " Swapping!" if swapping == 1 else ""))
    steps = len(episode_dict["rgbds_1"])
    for step in range(steps):
        plot_step(step, episode_dict, last_step = step + 1 == steps, saving = saving)
        if(episode_dict["task"]).parenting: pass 
        else: plot_step(step, episode_dict, agent_1 = False, last_step = step + 1 == steps, saving = saving)
    if(saving):
        print("SAVED PLOTS")
        os.chdir('..')
        os.chdir('..')
    
    
    
def plot_step(step, episode_dict, agent_1 = True, last_step = False, saving = True):
    agent_num = 1 if agent_1 else 2
    
    data = []
    
    data.append(["Goal", [episode_dict["goal"]], .1])
    if not step == 0:
        data.append(["Mother Comm", [episode_dict[f"mother_comm_{agent_num}"][step-1]], .1])
        
    data.append(["Bird's Eye View", [episode_dict[f"birds_eye_{agent_num}"][step], "image"], 1])
    
    if(not step == 0):
        data.append(["", ["Real"], ["Prior"], ["Posterior"], .1])
    
    if(step == 0):
        data.append([f"RGBD ({agent_num})", [episode_dict[f"rgbds_{agent_num}"][step], "image"], 1])
        data.append([f"Comm_In ({agent_num})", [episode_dict[f"comms_in_{agent_num}"][step]], 1])
        data.append([f"Sensors ({agent_num})", [episode_dict[f"sensors_{agent_num}"][step], "sensors"], 1])
    else:
        data.append(
            [f"RGBD ({agent_num})", 
            [episode_dict[f"rgbds_{agent_num}"][step], "image"],
            [episode_dict[f"prior_predicted_rgbds_{agent_num}"][step-1], "image"],
            [episode_dict[f"posterior_predicted_rgbds_{agent_num}"][step-1], "image"], 1])
        data.append(
            [f"father_comm ({agent_num})",
            [episode_dict[f"comms_in_{agent_num}"][step]],
            ["\n\n" + episode_dict[f"prior_predicted_comms_in_{agent_num}"][step-1]],
            ["\n\n\n\n" + episode_dict[f"posterior_predicted_comms_in_{agent_num}"][step-1]], .3])
        data.append(
            [f"Sensors ({agent_num})", 
            [episode_dict[f"sensors_{agent_num}"][step], "sensors"],
            [episode_dict[f"prior_predicted_sensors_{agent_num}"][step-1], "sensors"],
            [episode_dict[f"posterior_predicted_sensors_{agent_num}"][step-1], "sensors"], 1])
        
        data.append([f"Action ({agent_num})", [episode_dict[f"action_texts_{agent_num}"][step-1]], .1])
        data.append([f"Comms Out ({agent_num})", [episode_dict[f"comms_out_{agent_num}"][step-1]], .1])
        
        data.append([f"RGBD DKL ({agent_num})", [episode_dict[f"rgbd_dkls_{agent_num}"][:step], "plot"], .5])
        data.append([f"Comm DKL ({agent_num})", [episode_dict[f"comm_dkls_{agent_num}"][:step], "plot"], .5])
        data.append([f"Sensors DKL ({agent_num})", [episode_dict[f"sensors_dkls_{agent_num}"][:step], "plot"], .5])
        
    max_sublist_len = 0
    for sublist in data:
        if(len(sublist) > max_sublist_len):
            max_sublist_len = len(sublist)
        


    def plot_text(ax, value):
        ax.text(0.1, 0.5, f"{value}", fontsize=12, verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')

    def plot_image(ax, image):
        ax.text(0.1, 0.9, "", fontsize=12, verticalalignment='center', transform=ax.transAxes)
        ax.imshow(image, cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

    def plot_sensors(ax, sensors_data):
        ax.text(0.1, 0.9, "", fontsize=12, verticalalignment='center', transform=ax.transAxes)
        sensors_image = how_to_plot_sensors(sensors_data)
        sensors_image = sensors_image[80:-70, 125:-100]
        ax.imshow(sensors_image)
        ax.axis('off')

    def plot_line_plot(ax, plot_data):
        ax.text(0.1, 0.9, "", fontsize=12, verticalalignment='center', transform=ax.transAxes)
        ax.plot(plot_data)
        ax.axis('on')
        
        
        
    def plot_sublist(fig, gs, sublist, row):
        ax = fig.add_subplot(gs[row, 0])
        plot_text(ax, sublist[0] + (":" if sublist[0] != "" else ""))
        for column, subsublist in enumerate(sublist[1:-1]):
            ax = fig.add_subplot(gs[row, column+1])
            if isinstance(subsublist[0], str):
                plot_text(ax, subsublist[0])
            elif subsublist[-1] == "image":
                plot_image(ax, subsublist[0])
            elif subsublist[-1] == "plot":
                ax = fig.add_subplot(gs[row, 1:])
                plot_line_plot(ax, subsublist[0])          
            elif subsublist[-1] == "sensors":
                plot_sensors(ax, subsublist[0])
        return(1)
        
        

    def create_plot(data):
        fig = plt.figure(figsize=(20, 25))
        height_ratios = [sublist[-1] for sublist in data]
        gs = gridspec.GridSpec(len(data), max_sublist_len, figure=fig, height_ratios=height_ratios)
        for row, sublist in enumerate(data):
            plot_sublist(fig, gs, sublist, row)
        #plt.tight_layout()
        if(saving):
            plt.savefig(f"Step {step} Agent {agent_num}.png")
        else:
            plt.show()
        plt.close()
        
    create_plot(data)
    
    

if __name__ == "__main__":
    print("name:\n{}\n".format(args.arg_name),)
    plot_dicts, min_max_dict, complete_order = load_dicts(args)
    plot_episodes(complete_order, plot_dicts)
    print("\nDuration: {}. Done!".format(duration()))
    
# %%