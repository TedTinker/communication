#%% 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
import re
import imageio
import numpy as np
import threading
import time

from utils import print, args, duration, load_dicts, wheels_joints_to_string, get_goal_from_one_hots, plot_number_bars
#from plotting_for_video_window import window
from pybullet_data.robots.robot_maker import robot_dict


    
def plot_video_step(step, episode_dict, agent_1=True, last_step=False, saving=True, dreaming=False, args=args):
    """ 
    window.step = step
    window.episode_dict = episode_dict
    window.agent_1 = agent_1 
    window.last_step = last_step 
    window.saving = saving 
    window.dreaming = dreaming 
    window.args = args
    window.sensor_plotter = robot_dict[self.args.robot_name]
    """

    sensor_plotter, sensor_values = robot_dict[args.robot_name]
    agent_num = 1 if agent_1 else 2

    obs = episode_dict[f"obs_{agent_num}"][step]
    vision = obs.vision[0, :, :, :-1]
    touch = obs.touch.tolist()[0]
    command_voice = obs.command_voice.human_friendly_text()
    report_voice = obs.report_voice.human_friendly_text(command = False)

    dpi = 100  
    # Create figure with no facecolor (transparent)
    fig = plt.figure(figsize=(4, 8), dpi=dpi, facecolor='none')
    fig.patch.set_alpha(0)
            
    # Main invisible axes covering the whole figure
    main_ax = fig.add_axes([0, 0, 1, 1])
    main_ax.set_axis_off()
    main_ax.patch.set_alpha(0)
    
    # Steps, upper right.
    main_ax.text(
        .94, 0.98, f"Step {step}",
        fontsize=15,
        transform=main_ax.transAxes,
        zorder=3,
        ha='right',
        va='center',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0, linewidth=2))
        
    # Vision image in the top-right region; adjust these coordinates as needed.
    vision_ax = fig.add_axes([0.05, 0.27, 0.9, 0.9])
    vision_ax.imshow(vision)
    vision_ax.set_xticks([])
    vision_ax.set_yticks([])
    vision_ax.set_xticklabels([])
    vision_ax.set_yticklabels([])
    vision_ax.patch.set_alpha(0)
    for spine in vision_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(3)  # Use a reasonable width like 3, not 50
    
    # Touch image in the mid-right region.
    touch_ax = fig.add_axes([0.05, -0.14, 0.9, 0.9])
    touch_image = sensor_plotter(touch)
    touch_image = touch_image[80:-70, 125:-120]  # Crop as needed.
    touch_ax.imshow(touch_image)
    touch_ax.patch.set_alpha(0)
    touch_ax.set_xticks([])
    touch_ax.set_yticks([])
    touch_ax.set_xticklabels([])
    touch_ax.set_yticklabels([])
    for spine in touch_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(3)
    
    # Command and report text on the main axes, aligned to the right.
    main_ax.text(
        .94, 0.10, command_voice,
        fontsize=15,
        transform=main_ax.transAxes,
        zorder=3,
        ha='right',
        va='center',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0, linewidth=2))

    main_ax.text(
        .94, 0.04, report_voice,
        fontsize=15,
        transform=main_ax.transAxes,
        zorder=3,
        ha='right',
        va='center',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0, linewidth=2))
    

    
    if saving:
        # Save with transparent=True so the background remains transparent.
        os.makedirs(f"saved_deigo/thesis_pics", exist_ok=True)
        os.makedirs(f"saved_deigo/thesis_pics/video_pics", exist_ok=True)
        plt.savefig(f"saved_deigo/thesis_pics/video_pics/Goal {command_voice} Step {step}.png",
                    transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
            
    

if __name__ == "__main__":
    print("name:\n{}\n".format(args.arg_name),)
    plot_dicts, min_max_dict, complete_order = load_dicts()
    plot_episodes(complete_order, plot_dicts)
    print("\nDuration: {}. Done!".format(duration()))
    
# %%
