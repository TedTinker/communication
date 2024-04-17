import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
import re
import imageio
import numpy as np

from utils import print, args, duration, load_dicts
print("name:\n{}".format(args.arg_name))

def plot_episodes(complete_order, plot_dicts):
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
                        animate_episode(key, episode_dict, arg_name)
                        
                        
                        
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
        if(episode_dict["task"]).parent: pass 
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
    
    sensors = episode_dict[f"sensors_{agent_num}"][step]
    text_list.append(str(sensors))
    label_list.append(f"Sensors ({agent_num}):")
    
    if not step == 0:
        
        sensors_p = episode_dict[f"prior_predicted_sensors_{agent_num}"][step-1]
        text_list.append(str(sensors_p))
        label_list.append(f"Predicted Sensors (Prior) ({agent_num}):")
        
        sensors_q = episode_dict[f"posterior_predicted_sensors_{agent_num}"][step-1]
        text_list.append(str(sensors_q))
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
        dkls = episode_dict[f"dkls_{agent_num}"][step-1]
        for layer in range(len(dkls)):
            dkl_list = [episode_dict[f"dkls_{agent_num}"][s][layer] for s in range(step)]
            text_list.append("plot")
            plot_list.append(dkl_list)
            label_list.append(f"Layer {layer} DKL ({agent_num}):")
        
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
    
    
    
def animate_episode(key, episode_dict, arg_name):
    agent_num, epoch, episode_num, swapping = key.split("_")
    try:
        os.mkdir(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}_swapping_{swapping}/animation")
    except: 
        pass
    os.chdir(f"{arg_name}/epoch_{epoch}_episode_{episode_num}_agent_{agent_num}_swapping_{swapping}/animation")
    print("Animating {}: agent {}, epoch {}, episode {}.{}".format(arg_name, agent_num, epoch, episode_num, " Swapping!" if swapping == 1 else ""))
    agent_1_frames = []
    agent_2_frames = []
    steps = len(episode_dict["rgbds_1"])
    for step in range(steps):
        animate_step(step, episode_dict, last_step = step + 1 == steps)
        if(episode_dict["task"]).parent: pass 
        else: animate_step(step, episode_dict, agent_1 = False, last_step = step + 1 == steps)
    files = [f for f in os.listdir("./") if f.endswith('.png')]
    
    def extract_numbers(filename):
        match = re.search(r"Step (\d+) Agent (\d+).png", filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 0
    
    sorted_files = sorted(files, key=lambda x: extract_numbers(x))
    with imageio.get_writer('animation.gif', mode='I', duration=1.0) as writer:
        for filename in sorted_files:
            image = imageio.imread(os.path.join("./", filename))
            writer.append_data(image)
    
    print("SAVED ANIMATION")
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')
    
    
    
def animate_step(step, episode_dict, agent_1 = True, last_step = False):
    agent_num = 1 if agent_1 else 2
    
    text_list = []
    image_list = []
    label_list = []
    plot_list = []
    
    goal = episode_dict["goal"]
    text_list.append(goal)
    label_list.append("Goal:")
        
    birds_eye = episode_dict[f"birds_eye_{agent_num}"][step]
    text_list.append("image")
    image_list.append(birds_eye)
    label_list.append(f"View Above ({agent_num}):")

    rgbd = episode_dict[f"rgbds_{agent_num}"][step]
    text_list.append("image")
    image_list.append(rgbd)
    label_list.append(f"RGBD ({agent_num}):")
    
    if step == 0:
        rgbd_q_shape = episode_dict[f"posterior_predicted_rgbds_{agent_num}"][step-1].shape
        rgbd_q = np.ones(rgbd_q_shape)
        text_list.append("image")
        image_list.append(rgbd_q)
        label_list.append(f"Predicted RGBD (Posterior) ({agent_num}):")
    else:
        rgbd_q = episode_dict[f"posterior_predicted_rgbds_{agent_num}"][step-1]
        text_list.append("image")
        image_list.append(rgbd_q)
        label_list.append(f"Predicted RGBD (Posterior) ({agent_num}):")

    comms_in = episode_dict[f"comms_in_{agent_num}"][step]
    text_list.append(comms_in)
    label_list.append(f"Comms In ({agent_num}):")
    
    if step == 0:
        comms_in_q = ""
        text_list.append(comms_in_q)
        label_list.append(f"Predicted Comms (Posterior) ({agent_num}):")
    else:
        comms_in_q = episode_dict[f"posterior_predicted_comms_in_{agent_num}"][step-1]
        text_list.append(comms_in_q)
        label_list.append(f"Predicted Comms (Posterior) ({agent_num}):")
    
    sensors = episode_dict[f"sensors_{agent_num}"][step]
    text_list.append(str(sensors))
    label_list.append(f"Sensors ({agent_num}):")
    if step == 0:
        sensors_q_shape = len(episode_dict[f"posterior_predicted_sensors_{agent_num}"][step-1])
        text_list.append(str(1) * sensors_q_shape)
        label_list.append(f"Predicted Sensors (Posterior) ({agent_num}):")
        
        comms_out = ""
        text_list.append(comms_out)
        label_list.append(f"Comms Out ({agent_num}):")
    else:
        sensors_q = episode_dict[f"posterior_predicted_sensors_{agent_num}"][step-1]
        text_list.append(str(sensors_q))
        label_list.append(f"Predicted Sensors (Posterior) ({agent_num}):")
        
        comms_out = episode_dict[f"comms_out_{agent_num}"][step-1]
        text_list.append(comms_out)
        label_list.append(f"Comms Out ({agent_num}):")
        
    if not step == 0:
        dkls = episode_dict[f"dkls_{agent_num}"][step-1]
        for layer in range(len(dkls)):
            dkl_list = [episode_dict[f"dkls_{agent_num}"][s][layer] for s in range (step)]
            text_list.append("plot")
            plot_list.append(dkl_list)
            label_list.append(f"Layer {layer} DKL ({agent_num}):")
    else:
        dkls = episode_dict[f"dkls_{agent_num}"][step-1]
        for layer in range(len(dkls)):
            text_list.append("plot")
            plot_list.append([0])
            label_list.append(f"Layer {layer} DKL ({agent_num}):")
        
    fig = plt.figure(figsize=(15, 15))
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
        if text == "image":
            image = image_list[images_plotted]
            ax_img.imshow(image)
            pos = ax_img.get_position()
            rect = patches.Rectangle((-.5, -.5), image.shape[1], image.shape[0], linewidth=4, edgecolor='black', facecolor='none')
            ax_img.add_patch(rect)
            ax_img.axis('off')
            images_plotted += 1
        elif text == "plot":
            ax_img.plot(plot_list[plots_plotted])
            ax_img.set_ylim(bottom=0)
            plots_plotted += 1
        else:
            text = text.replace('\t', ' ').replace('(', ' ').replace(')', ' ')
            ax_img.text(0.2, 0, text, va='center', ha='left', fontsize=12)
            ax_img.axis('off')
            
    plt.savefig(f"Step {step} Agent {agent_num}.png")
    plt.close()
    
    
  
plot_dicts, min_max_dict, complete_order = load_dicts(args)
plot_episodes(complete_order, plot_dicts)
print("\nDuration: {}. Done!".format(duration()))