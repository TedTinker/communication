#%% 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
import re
import imageio
import numpy as np

from utils import print, args, duration, load_dicts, wheels_joints_to_string, get_goal_from_one_hots, plot_number_bars
if(args.robot_name == "two_side_arm"):
    from pybullet_data.robot_maker_two_side_arm import how_to_plot_sensors
if(args.robot_name == "one_head_arm"):
    from pybullet_data.robot_maker_one_head_arm import how_to_plot_sensors
if(args.robot_name == "two_head_arm"):
    from pybullet_data.robot_maker_two_head_arm import how_to_plot_sensors



def human_friendly_text(goal):
    return(f"{goal.human_text} ({goal.char_text})")



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
        print("Saving {}: agent {}, epoch {}, episode {}.{}".format(arg_name, agent_num, epoch, episode_num, " Swapping!" if swapping == 1 else ""), end = "... ")
    steps = len(episode_dict["obs_1"])
    for step in range(steps):
        plot_step(step, episode_dict, last_step = step + 1 == steps, saving = saving)
        if(episode_dict["processor"]).parenting: pass 
        else: plot_step(step, episode_dict, agent_1 = False, last_step = step + 1 == steps, saving = saving)
    if(saving):
        print("SAVED PLOTS")
        os.chdir('..')
        os.chdir('..')
    
    
    
def plot_step(step, episode_dict, agent_1 = True, last_step = False, saving = True):
    agent_num = 1 if agent_1 else 2
    
    obs = episode_dict[f"obs_{agent_num}"][step]
    rgbd = obs.rgbd[0,:,:,:-1]
    sensors = obs.sensors.tolist()[0]
    father_voice = obs.father_voice
    mother_voice = obs.mother_voice
    if(step != 0):
        prior = episode_dict[f"prior_predictions_{agent_num}"][step-1]
        prior_rgbd = prior.rgbd[0,0,:,:,:-1]
        prior_sensors = prior.sensors.tolist()[0][0]
        prior_father_voice = prior.father_voice 
        prior_mother_voice = prior.mother_voice 
        posterior = episode_dict[f"posterior_predictions_{agent_num}"][step-1]
        posterior_rgbd = posterior.rgbd[0,0,:,:,:-1]
        posterior_sensors = posterior.sensors.tolist()[0][0]
        posterior_father_voice = posterior.father_voice 
        posterior_mother_voice = posterior.mother_voice 
        action = episode_dict[f"action_{agent_num}"][step-1]
            
    data = []
    
    data.append(["Goal", [human_friendly_text(episode_dict["goal"])], .1])
    if not step == 0:
        data.append(["Acheived Goal", [human_friendly_text(mother_voice)], .1])
        
    data.append(["Bird's Eye View", [episode_dict[f"birds_eye_{agent_num}"][step], "image"], 1])
    
    if(not step == 0):
        data.append(["", ["Real"], ["Prior"], ["Posterior"], .1])
    

    
    if(step == 0):
        data.append([f"RGBD ({agent_num})", [rgbd, "image"], 1])
        data.append([f"Sensors ({agent_num})", [sensors, "sensors"], 1])
        data.append([f"Father voice ({agent_num})", [human_friendly_text(father_voice)], 1])
        data.append([f"Mother voice ({agent_num})", [human_friendly_text(mother_voice)], 1])
    else:
        data.append(
            [f"RGBD ({agent_num})", 
            [rgbd, "image"],
            [prior_rgbd, "image"],
            [posterior_rgbd, "image"], 1])
        data.append(
            [f"Sensors ({agent_num})", 
            [sensors, "sensors"],
            [prior_sensors, "sensors"],
            [posterior_sensors, "sensors"], 1])
        data.append(
            [f"Father voice ({agent_num})",
            [human_friendly_text(father_voice)],
            ["\n\n" + human_friendly_text(prior_father_voice)],
            ["\n\n\n\n" + human_friendly_text(posterior_father_voice)], .3])
        data.append(
            [f"Mother voice ({agent_num})",
            [human_friendly_text(mother_voice)],
            ["\n\n" + human_friendly_text(prior_mother_voice)],
            ["\n\n\n\n" + human_friendly_text(posterior_mother_voice)], .3])
        
        data.append([f"Wheels, Joints ({agent_num})", [action.wheels_joints, "bar_plot"], .5])
        data.append([f"Voice Out ({agent_num})", [human_friendly_text(get_goal_from_one_hots(action.voice_out))], .1])
        
        data.append([f"RGBD DKL ({agent_num})", [episode_dict[f"rgbd_dkl_{agent_num}"][:step], "line_plot"], .5])
        data.append([f"Sensors DKL ({agent_num})", [episode_dict[f"sensors_dkl_{agent_num}"][:step], "line_plot"], .5])
        data.append([f"Father voice DKL ({agent_num})", [episode_dict[f"father_voice_dkl_{agent_num}"][:step], "line_plot"], .5])
        data.append([f"Mother voice DKL ({agent_num})", [episode_dict[f"mother_voice_dkl_{agent_num}"][:step], "line_plot"], .5])
        
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
        
    def plot_bar_plot(ax, plot_data):
        numbers = plot_data.flatten().tolist()
        fontsize = 7
        ax.bar(range(len(numbers)), numbers, color=['red' if x < 0 else 'blue' for x in numbers])
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel("Index", fontsize = fontsize)
        ax.set_ylabel("Value", fontsize = fontsize)
        #ax.title("Bar Plot of Actions", fontsize = fontsize)
        ax.set_ylim(-1, 1) 
        if(len(numbers) == 3):
            ax.set_xticks(range(3), ["left wheel speed", "right wheel speed", "joint 1 speed"], rotation=45, ha='right', fontsize = fontsize)
        if(len(numbers) == 4):
            ax.set_xticks(range(4), ["left wheel speed", "right wheel speed", "joint 1 speed", "joint 2 speed"], rotation=45, ha='right', fontsize = fontsize)
        #ax.set_yticks(fontsize = fontsize)
        ax.axis('on')

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
            elif subsublist[-1] == "bar_plot":
                ax = fig.add_subplot(gs[row, 1:])
                plot_bar_plot(ax, subsublist[0])  
            elif subsublist[-1] == "line_plot":
                ax = fig.add_subplot(gs[row, 1:])
                plot_line_plot(ax, subsublist[0])          
            elif subsublist[-1] == "sensors":
                plot_sensors(ax, subsublist[0])
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
