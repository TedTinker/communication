import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import torch
import numpy as np

from utils import print, args, duration, load_dicts
print("name:\n{}".format(args.arg_name))

def plot_episodes(complete_order, plot_dicts):
    for arg_name in complete_order:
        if(arg_name in ["break", "empty_space"]): pass 
        else:
            for plot_dict in plot_dicts:
                if(plot_dict["arg_name"] == arg_name):
                    episode_dicts = plot_dict["episode_dicts"]
                    for key, episode_dict in episode_dicts.items():
                        plot_episode(key, episode_dict, arg_name)
                        
                        
                        
def plot_episode(key, episode_dict, arg_name):
    agent_num, epoch, episode_num, swapping = key.split("_")
    os.mkdir(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}_swapping_{swapping}")
    os.chdir(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}_swapping_{swapping}")
    print("{}: agent {}, epoch {}, episode {}.{}".format(arg_name, agent_num, epoch, episode_num, " Swapping!" if swapping == 1 else ""))
    steps = len(episode_dict["rgbds_1"])
    for step in range(steps):
        plot_step(step, episode_dict, last_step = step + 1 == steps)
        if(episode_dict["task"]).parent: pass 
        else: plot_step(step, episode_dict, agent_1 = False, last_step = step + 1 == steps)
    os.chdir('..')
    os.chdir('..')
    
    
    
def plot_step(step, episode_dict, agent_1 = True, last_step = False):
    agent_num = 1 if agent_1 else 2
    
    text_list = []
    image_list = []
    label_list = []
    
    if(step == 0):
        goal = episode_dict["task"].goal_text
        text_list.append(goal)
        label_list.append("Goal:")
        
    birds_eye = episode_dict[f"birds_eye_{agent_num}"][step]
    text_list.append(None)
    image_list.append(birds_eye)
    label_list.append(f"View Above ({agent_num}):")

    rgbd = episode_dict[f"rgbds_{agent_num}"][step]
    text_list.append(None)
    image_list.append(rgbd)
    label_list.append(f"RGBD ({agent_num}):")
    
    speed = episode_dict[f"speeds_{agent_num}"][step]
    text_list.append(speed)
    label_list.append(f"Speed ({agent_num}):")

    comms_in = episode_dict[f"comms_in_{agent_num}"][step]
    text_list.append(comms_in)
    label_list.append(f"Comms In ({agent_num}):")
    
    if not last_step:
        
        recommended_actions = episode_dict[f"recommended_{agent_num}"][step]
        text_list.append(recommended_actions)
        label_list.append(f"Recommendation ({agent_num}):")

        actions = episode_dict[f"actions_{agent_num}"][step]
        text_list.append(actions)
        label_list.append(f"Actions ({agent_num}):")
        
        comms_out = episode_dict[f"comms_out_{agent_num}"][step]
        text_list.append(comms_out)
        label_list.append(f"Comms Out ({agent_num}):")

        rewards = episode_dict["rewards"][step]
        text_list.append(rewards)
        label_list.append("Reward:")
        
        values = episode_dict[f"critic_predictions_{agent_num}"][step]
        values_text = ""
        for i, value in enumerate(values):
            values_text += "{}".format(value) + ("." if i+1 == len(values) else ", ")
        text_list.append(values_text)
        label_list.append(f"Predicted Values ({agent_num}):")

        rgbd_p = episode_dict[f"prior_predicted_rgbds_{agent_num}"][step]
        text_list.append(None)
        image_list.append(rgbd_p)
        label_list.append(f"Predicted RGBD (Prior) ({agent_num}):")
        
        speed_p = episode_dict[f"prior_predicted_speeds_{agent_num}"][step]
        text_list.append(speed_p)
        label_list.append(f"Predicted Speed (Prior) ({agent_num}):")

        comms_in_p = episode_dict[f"prior_predicted_comms_in_{agent_num}"][step]
        text_list.append(comms_in_p)
        label_list.append(f"Predicted Comms (Prior) ({agent_num}):")

        rgbd_q = episode_dict[f"posterior_predicted_rgbds_{agent_num}"][step]
        text_list.append(None)
        image_list.append(rgbd_q)
        label_list.append(f"Predicted RGBD (Posterior) ({agent_num}):")
        
        speed_q = episode_dict[f"posterior_predicted_speeds_{agent_num}"][step]
        text_list.append(speed_q)
        label_list.append(f"Predicted Speed (Posterior) ({agent_num}):")

        comms_in_q = episode_dict[f"posterior_predicted_comms_in_{agent_num}"][step]
        text_list.append(comms_in_q)
        label_list.append(f"Predicted Comms (Posterior) ({agent_num}):")
        
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(len(label_list), 2, height_ratios=[5 if text == None else 4 if text.startswith("Yaw:") else 1 for text in text_list], width_ratios=[1, 4])
    images_plotted = 0
    for i, (text, label) in enumerate(zip(text_list, label_list)):
        ax_text = fig.add_subplot(gs[i, 0])
        ax_text.axis('off')
        ax_img = fig.add_subplot(gs[i, 1])
        ax_img.axis('off')
        if(i == 0): 
            ax_text.text(0.0, 1, "Step {}".format(step+1) if not last_step else "Done", 
                    va='center', ha='left', fontsize=20, fontweight='bold')
        ax_text.text(0.1, 0, label, va='center', ha='left', fontsize=12, fontweight='bold')
        if(text) == None:
            ax_img.imshow(image_list[images_plotted])
            images_plotted += 1
        else:
            text = text.replace('\t', ' ').replace('(', ' ').replace(')', ' ')
            ax_img.text(0.2, 0, text, va='center', ha='left', fontsize=12)
            
    plt.savefig(f"Step {step+1} Agent {agent_num}")
    plt.close()

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plot_episodes(complete_order, plot_dicts)
print("\nDuration: {}. Done!".format(duration()))