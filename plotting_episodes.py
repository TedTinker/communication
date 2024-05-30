import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
import re
import imageio
import numpy as np

from utils import print, args, duration, load_dicts



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
                        
                        
                        
def plot_episode(key, episode_dict, arg_name):
    agent_num, epoch, episode_num, swapping = key.split("_")
    try:
        os.mkdir(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}_swapping_{swapping}")
    except: 
        pass
    os.chdir(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}_swapping_{swapping}")
    print("Plotting {}: agent {}, epoch {}, episode {}.{}".format(arg_name, agent_num, epoch, episode_num, " Swapping!" if swapping == 1 else ""))
    steps = len(episode_dict["rgbds_1"])
    for step in range(steps):
        plot_step(step, episode_dict, last_step = step + 1 == steps)
        if(episode_dict["task"]).parenting: pass 
        else: plot_step(step, episode_dict, agent_1 = False, last_step = step + 1 == steps)
    print("SAVED PLOTS")
    os.chdir('..')
    os.chdir('..')
    
    
    
def plot_step(step, episode_dict, agent_1 = True, last_step = False):
    agent_num = 1 if agent_1 else 2
    
    text_list = []
    image_list = []
    plot_list = []
    label_list = []
    
    goal = episode_dict["goal"]
    text_list.append(goal)
    label_list.append("Goal:")
    
    if not step == 0:
        which_goal_message = episode_dict[f"which_goal_message_{agent_num}"][step-1]
        text_list.append(which_goal_message)
        label_list.append("Achieved goal:")
        
    birds_eye = episode_dict[f"birds_eye_{agent_num}"][step]
    text_list.append("image")
    image_list.append(birds_eye)
    label_list.append(f"View Above ({agent_num}):")

    rgbd = episode_dict[f"rgbds_{agent_num}"][step]
    text_list.append("image")
    image_list.append(rgbd)
    label_list.append(f"RGBD ({agent_num}):")
    
    if not step == 0:

        rgbd_p = episode_dict[f"prior_predicted_rgbds_{agent_num}"][step-1]
        text_list.append("image")
        image_list.append(rgbd_p)
        label_list.append(f"Predicted RGBD (Prior) ({agent_num}):")
        
        rgbd_q = episode_dict[f"posterior_predicted_rgbds_{agent_num}"][step-1]
        text_list.append("image")
        image_list.append(rgbd_q)
        label_list.append(f"Predicted RGBD (Posterior) ({agent_num}):")

    comms_in = episode_dict[f"comms_in_{agent_num}"][step]
    text_list.append(comms_in)
    label_list.append(f"Comms In ({agent_num}):")
    
    if not step == 0:

        comms_in_p = episode_dict[f"prior_predicted_comms_in_{agent_num}"][step-1]
        text_list.append(comms_in_p)
        label_list.append(f"Predicted Comms (Prior) ({agent_num}):")
        
        comms_in_q = episode_dict[f"posterior_predicted_comms_in_{agent_num}"][step-1]
        text_list.append(comms_in_q)
        label_list.append(f"Predicted Comms (Posterior) ({agent_num}):")
    
    # How to make this use names of any none-zero sensors?
    sensors = episode_dict[f"sensors_{agent_num}"][step][0]
    sensor_names = [args.sensor_names[i] for i in range(len(sensors)) if sensors[i] > 0]    
    text_list.append(str(sensor_names))
    label_list.append(f"Sensors ({agent_num}):")
    
    if not step == 0:
        
        sensors_p = episode_dict[f"prior_predicted_sensors_{agent_num}"][step-1]
        sensor_names_p = [args.sensor_names[i] for i in range(len(sensors_p)) if sensors_p[i] > .25]    
        text_list.append(str(sensor_names_p))
        label_list.append(f"Predicted Sensors (Prior) ({agent_num}):")
        
        sensors_q = episode_dict[f"posterior_predicted_sensors_{agent_num}"][step-1]
        sensor_names_q = [args.sensor_names[i] for i in range(len(sensors_q)) if sensors_q[i] > .25]    
        text_list.append(str(sensor_names_q))
        label_list.append(f"Predicted Sensors (Posterior) ({agent_num}):")
        
        raw_rewards = episode_dict["raw_rewards"][step-1]
        text_list.append(raw_rewards)
        label_list.append("Raw reward:")
        
        distance_rewards = episode_dict[f"distance_rewards_{agent_num}"][step-1]
        text_list.append(distance_rewards)
        label_list.append("Distance reward:")
        
        angle_rewards = episode_dict[f"angle_rewards_{agent_num}"][step-1]
        text_list.append(angle_rewards)
        label_list.append("Angle reward:")
        
        total_rewards = episode_dict[f"total_rewards_{agent_num}"][step-1]
        text_list.append(total_rewards)
        label_list.append("Total reward:")
        
        values = episode_dict[f"critic_predictions_{agent_num}"][step-1]
        values_text = ""
        for i, value in enumerate(values):
            values_text += "{}".format(value) + ("." if i+1 == len(values) else ", ")
        text_list.append(values_text)
        label_list.append(f"Predicted Values ({agent_num}):")
    
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
        
    if not step == 0:
        rgbd_dkls = episode_dict[f"rgbd_dkls_{agent_num}"][:step]
        text_list.append("plot")
        plot_list.append(rgbd_dkls)
        label_list.append(f"RGBD DKL ({agent_num}):")
            
        comm_dkls = episode_dict[f"comm_dkls_{agent_num}"][:step]
        text_list.append("plot")
        plot_list.append(comm_dkls)
        label_list.append(f"Comm DKL ({agent_num}):")
            
        sensors_dkls = episode_dict[f"sensors_dkls_{agent_num}"][:step]
        text_list.append("plot")
        plot_list.append(sensors_dkls)
        label_list.append(f"Sensors DKL ({agent_num}):")
        
    fig = plt.figure(figsize=(15, 20 if last_step else 20))
    gs = gridspec.GridSpec(len(label_list), 2, height_ratios=[20 if text == "image" else 10 if text == "plot" else 1 if text.startswith("Yaw:") else 1 for text in text_list], width_ratios=[1, 4])
    images_plotted = 0
    plots_plotted = 0
    for i, (text, label) in enumerate(zip(text_list, label_list)):
        ax_text = fig.add_subplot(gs[i, 0])
        ax_text.axis('off')
        ax_img = fig.add_subplot(gs[i, 1])
        if(i == 0): 
            ax_text.text(0.0, 1, "Step {}".format(step) if not last_step else "Step {} (Done)".format(step), 
                    va='center', ha='left', fontsize=20, fontweight='bold')
        ax_text.text(0.1, 0, label, va='center', ha='left', fontsize=12, fontweight='bold')
        if(text) == "image":
            image = image_list[images_plotted]
            ax_img.imshow(image)
            rect = patches.Rectangle((-.5, -.5), image.shape[1], image.shape[0], linewidth=4, edgecolor='black', facecolor='none')
            ax_img.add_patch(rect)
            ax_img.axis('off')
            images_plotted += 1
        elif(text) == "plot":
            ax_img.plot(plot_list[plots_plotted])
            ax_img.set_ylim(bottom=0)
            plots_plotted += 1
        else:
            text = text.replace('\t', ' ') # .replace('(', ' ').replace(')', ' ')
            ax_img.text(0.2, 0, text, va='center', ha='left', fontsize=12)
            ax_img.axis('off')
            
    plt.savefig(f"Step {step} Agent {agent_num}.png")
    plt.close()
    
    
  
plot_dicts, min_max_dict, complete_order = load_dicts(args)
plot_episodes(complete_order, plot_dicts)
print("\nDuration: {}. Done!".format(duration()))