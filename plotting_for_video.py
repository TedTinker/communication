#%% 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.table import Table
import tkinter as tk
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
import re
import imageio
import numpy as np
import threading
import time

from utils import print, args, duration, load_dicts, wheels_joints_to_string, get_goal_from_one_hots, plot_number_bars, empty_goal
#from plotting_for_video_window import window
from pybullet_data.robots.robot_maker import robot_dict


    
def plot_video_step(step, episode_dict, agent_1=True, last_step=False, saving=True, dreaming=False, args=args):

    sensor_plotter, sensor_values = robot_dict[args.robot_name]
    agent_num = 1 if agent_1 else 2

    obs = episode_dict[f"obs_{agent_num}"][step]
    vision = obs.vision[0, :, :, :-1]
    touch = obs.touch.tolist()[0]
    
    command_voice = obs.command_voice.human_friendly_text()
    report_voice = obs.report_voice.human_friendly_text(command = False)
        
    command_task = obs.command_voice.task.name# .replace(" ", "\n")
    command_color = obs.command_voice.color.name# .replace(" ", "\n")
    command_shape = obs.command_voice.shape.name# .replace(" ", "\n")

    report_task = obs.report_voice.task.name# .replace(" ", "\n")
    report_color = obs.report_voice.color.name# .replace(" ", "\n")
    report_shape = obs.report_voice.shape.name# .replace(" ", "\n")
    
    if(step != 0):
        posterior = episode_dict[f"posterior_predictions_{agent_num}"][step-1]
        posterior_report_voice = posterior.report_voice 
    else:
        posterior_report_voice = empty_goal
    predicted_report_task = posterior_report_voice.task.name# .replace(" ", "\n")
    predicted_report_color = posterior_report_voice.color.name# .replace(" ", "\n")
    predicted_report_shape = posterior_report_voice.shape.name# .replace(" ", "\n")
    
    cell_data = [
        ["", "Task", "Color", "Shape"],
        ["Command", command_task, command_color, command_shape],
        ["Report", report_task, report_color, report_shape],
        ["Predicted\nReport", predicted_report_task, predicted_report_color, predicted_report_shape]]
    
    visual_curiosity        = episode_dict[f"vision_dkl_{agent_num}"][:step]        
    visual_curiosity        = [0] + [c * args.hidden_state_eta_vision for c in visual_curiosity]
    touch_curiosity         = episode_dict[f"touch_dkl_{agent_num}"][:step]         
    touch_curiosity         = [0] + [c * args.hidden_state_eta_touch for c in touch_curiosity]
    report_voice_curiosity  = episode_dict[f"report_voice_dkl_{agent_num}"][:step] 
    report_voice_curiosity  = [0] + [c * args.hidden_state_eta_report_voice for c in report_voice_curiosity]

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
    touch_image = touch_image[80:-70, 80:-60]
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
    
    # Command, report, and predicted report text
    table_ax = fig.add_axes([0.05, -0.18, 0.9, 0.25])  # position: [left, bottom, width, height]
    table_ax.set_axis_off()
    """table = Table(table_ax, bbox=[0, 0, 1, 1])

    n_rows, n_cols = len(cell_data), len(cell_data[0])
    width, height = 1.0 / n_cols, 1.0 / n_rows

    for i in range(n_rows):
        for j in range(n_cols):
            text = cell_data[i][j]
            cell = table.add_cell(i, j, width, height, text=text, loc='center', facecolor='white', edgecolor='black')
            cell.get_text().set_fontsize(14)  # Set fontsize here"""
            
    # Jun wants just the text
    fontsize = 12
    table_ax.text(0, .8, s = f"Command:\n{command_task} {command_color} {command_shape}.", horizontalalignment='left', verticalalignment='center', fontsize = fontsize)
    table_ax.text(0, .45, s = f"Report:\n{report_task} {report_color} {report_shape}.", horizontalalignment='left', verticalalignment='center', fontsize = fontsize)
    table_ax.text(0, .1, s = f"Predicted Report:\n{predicted_report_task} {predicted_report_color} {predicted_report_shape}.", horizontalalignment='left', verticalalignment='center', fontsize = fontsize)
            
    # Curiosity values
    all_curiosities = visual_curiosity + touch_curiosity + report_voice_curiosity
    if(all_curiosities == []):
        all_curiosities = [0]
    min_curi = min(all_curiosities) * .9
    max_curi = max(all_curiosities) * 1.1
    
    plot_height = 0.07
    base_bottom = -0.30  
    
    curiosity_titles = ["Vision Curiosity", "Touch Curiosity", "Report Voice Curiosity"]
    curiosity_data = [visual_curiosity, touch_curiosity, report_voice_curiosity]

    for idx, (title, data) in enumerate(zip(curiosity_titles, curiosity_data)):
        bottom_pos = base_bottom - idx * (plot_height + 0.03)
        ax = fig.add_axes([0.1, bottom_pos, 0.8, plot_height])
        if len(data) > 1:
            ax.plot(data, color='black', linewidth=2)
        elif len(data) == 1:
            ax.plot([0], data, marker='o', markersize=6, color='black')
        ax.set_ylim([min_curi, max_curi])
        ax.set_xlim([0, step])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)
        ax.patch.set_alpha(0)
        ax.text(-0.02, min_curi, f"{round(min_curi)}", va='center', ha='right', fontsize=8, transform=ax.get_yaxis_transform())
        ax.text(-0.02, max_curi, f"{round(max_curi)}", va='center', ha='right', fontsize=8, transform=ax.get_yaxis_transform())
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)

    #table_ax.add_table(table)
    

    
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
    plot_dicts, min_max_dict, complete_order = load_dicts(args)
    plot_episodes(complete_order, plot_dicts)
    print("\nDuration: {}. Done!".format(duration()))
    
# %%
