#%% 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
import re
import imageio
import numpy as np

from utils import print, args, duration, load_dicts, wheels_joints_to_string, get_goal_from_one_hots, plot_number_bars
from pybullet_data.robots.robot_maker import robot_dict



def human_friendly_text(goal):
    return(f"{goal.human_text} ({goal.char_text})")


    
def plot_step(step, episode_dict, agent_1 = True, last_step = False, saving = True, dreaming = False, args = args):
    sensor_plotter, sensor_values = robot_dict[args.robot_name]

    agent_num = 1 if agent_1 else 2
    
    obs = episode_dict[f"obs_{agent_num}"][step]
    vision = obs.vision[0,:,:,:-1]
    touch = obs.touch.tolist()[0]
    command_voice = obs.command_voice
    report_voice = obs.report_voice
    if(step != 0):
        prior = episode_dict[f"prior_predictions_{agent_num}"][step-1]
        prior_vision = prior.vision[0,0,:,:,:-1]
        prior_touch = prior.touch.tolist()[0][0]
        prior_command_voice = prior.command_voice 
        prior_report_voice = prior.report_voice 
        posterior = episode_dict[f"posterior_predictions_{agent_num}"][step-1]
        posterior_vision = posterior.vision[0,0,:,:,:-1]
        posterior_touch = posterior.touch.tolist()[0][0]
        posterior_command_voice = posterior.command_voice 
        posterior_report_voice = posterior.report_voice 
        action = episode_dict[f"action_{agent_num}"][step-1]
            
    data = []
    
    data.append(["Goal", [human_friendly_text(episode_dict["goal"])], .1])
    if not step == 0:
        data.append(["Acheived Goal", [human_friendly_text(report_voice)], .1])
        
    data.append(["Bird's Eye View", [episode_dict[f"birds_eye_{agent_num}"][step], "image"], 1])
    
    if(not step == 0):
        data.append(["", ["Real (not seen in dream; \nagent sees posterior)" + "" if dreaming and step != 0 else ""], ["Prior"], ["Posterior"], .1])
    

    
    if(step == 0):
        data.append([f"Vision ({agent_num})", [vision, "image"], 1])
        data.append([f"Touch ({agent_num})", [touch, "touch"], 1])
        data.append([f"Command voice ({agent_num})", [human_friendly_text(command_voice)], 1])
        data.append([f"Report voice ({agent_num})", [human_friendly_text(report_voice)], 1])
    else:
        data.append(
            [f"Vision ({agent_num})", 
            [vision, "image"],
            [prior_vision, "image"],
            [posterior_vision, "image"], 1])
        data.append(
            [f"Touch ({agent_num})", 
            [touch, "touch"],
            [prior_touch, "touch"],
            [posterior_touch, "touch"], 1])
        data.append(
            [f"Command voice ({agent_num})",
            [human_friendly_text(command_voice)],
            [human_friendly_text(prior_command_voice)],
            [human_friendly_text(posterior_command_voice)], .3])
        data.append(
            [f"Report voice ({agent_num})",
            [human_friendly_text(report_voice)],
            [human_friendly_text(prior_report_voice)],
            [human_friendly_text(posterior_report_voice)], .3])
        
        data.append([f"Wheels, Joints ({agent_num})", [action.wheels_joints, "bar_plot"], .5])
        data.append([f"Voice Out ({agent_num})", [human_friendly_text(get_goal_from_one_hots(action.voice_out))], .3])
        
        data.append([f"Vision DKL ({agent_num})", [episode_dict[f"vision_dkl_{agent_num}"][:step], "line_plot"], .5])
        data.append([f"Touch DKL ({agent_num})", [episode_dict[f"touch_dkl_{agent_num}"][:step], "line_plot"], .5])
        data.append([f"Command voice DKL ({agent_num})", [episode_dict[f"command_voice_dkl_{agent_num}"][:step], "line_plot"], .5])
        data.append([f"Report voice DKL ({agent_num})", [episode_dict[f"report_voice_dkl_{agent_num}"][:step], "line_plot"], .5])
        
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

    def plot_touch(ax, touch_data):
        ax.text(0.1, 0.9, "", fontsize=12, verticalalignment='center', transform=ax.transAxes)
        touch_image = sensor_plotter(touch_data)
        touch_image = touch_image[80:-70, 125:-100]
        ax.imshow(touch_image)
        ax.axis('off')
        
    def plot_bar_plot(ax, plot_data):
        numbers = plot_data.flatten().tolist()
        fontsize = 12
        ax.bar(range(len(numbers)), numbers, color=['red' if x < 0 else 'blue' for x in numbers])
        
        ax.axhline(0, color='black', linewidth=1)
        #ax.set_xlabel("Index", fontsize = fontsize)
        ax.set_ylabel("Value", fontsize = fontsize)
        #ax.title("Bar Plot of Actions", fontsize = fontsize)
        ax.set_ylim(-1, 1) 
        xticks = ["left wheel", "right wheel"]
        i = 1
        while(len(xticks) < len(numbers)):
            xticks.append(f"joint {i}")
            i += 1
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=0, ha='right', fontsize=fontsize)
        ax.tick_params(axis='x', which='both', bottom=True, top=False)
        #ax.set_yticks(fontsize = fontsize)
        #ax.axis('on')

    def plot_line_plot(ax, plot_data):
        ax.text(0.1, 0.9, "", fontsize=12, verticalalignment='center', transform=ax.transAxes)
        ax.plot(plot_data)
        ax.axis('on')
        
        
        
    def plot_sublist(fig, gs, sublist, row):
        ax = fig.add_subplot(gs[row, 0])
        plot_text(ax, sublist[0] + (":" if sublist[0] != "" else ""))
        
        for column, subsublist in enumerate(sublist[1:-1]):
            if isinstance(subsublist[0], str):
                ax = fig.add_subplot(gs[row, column+1])
                plot_text(ax, subsublist[0])
            elif subsublist[-1] == "image":
                ax = fig.add_subplot(gs[row, column+1])
                plot_image(ax, subsublist[0])
            elif subsublist[-1] == "bar_plot":
                ax = fig.add_subplot(gs[row, 1:])
                plot_bar_plot(ax, subsublist[0])  
            elif subsublist[-1] == "line_plot":
                ax = fig.add_subplot(gs[row, 1:])
                plot_line_plot(ax, subsublist[0])          
            elif subsublist[-1] == "touch":
                ax = fig.add_subplot(gs[row, column+1])
                plot_touch(ax, subsublist[0])
        return(1)
        
        

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
            
    

if __name__ == "__main__":
    print("name:\n{}\n".format(args.arg_name),)
    plot_dicts, min_max_dict, complete_order = load_dicts(args)
    plot_episodes(complete_order, plot_dicts)
    print("\nDuration: {}. Done!".format(duration()))
    
# %%
