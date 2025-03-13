#%% 

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

from utils import args
from pybullet_data.robots.robot_maker import robot_dict



class VideoPlotWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dynamic Video Plot Window")
        self.figure = plt.Figure(figsize=(4, 8), dpi=100, facecolor='none')
        self.figure.patch.set_alpha(0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.counter = 0
        
        self.step = None
        self.episode_dict = {}
        self.agent_1 = True 
        self.last_step = False 
        self.saving = True 
        self.dreaming = False 
        self.args = args
        self.sensor_plotter = robot_dict[self.args.robot_name]
        
        

    def update_plot(self):
        agent_num = 1 if self.agent_1 else 2
        sensor_plotter, sensor_values = robot_dict[self.args.robot_name]
        
        if(step == None):
            vision = np.random.rand(100, 100, 3)  # Replace with your actual vision image
            touch = np.random.rand(150, 150, 3)  # Replace with your processed touch image
            command_voice = "Command: Sample Command"  # Replace with obs.command_voice.human_friendly_text()
            report_voice = "Report: Sample Report"       # Replace with obs.report_voice.human_friendly_text(command=False)
        else:
            obs = self.episode_dict[f"obs_{self.agent_num}"][step]
            vision = obs.vision[0, :, :, :-1]
            touch = obs.touch.tolist()[0]
            command_voice = obs.command_voice.human_friendly_text()
            report_voice = obs.report_voice.human_friendly_text(command = False)
    
        """Clear and update the plot with new (dummy) data, similar to plot_video_step."""
        self.figure.clf()  # Clear the previous plot
        
        # Main invisible axes covering the whole figure.
        main_ax = self.figure.add_axes([0, 0, 1, 1])
        main_ax.set_axis_off()
        main_ax.patch.set_alpha(0)
        
        # Display step information at the upper right.
        main_ax.text(
            0.94, 0.98, f"Step {self.counter}",
            fontsize=15,
            transform=main_ax.transAxes,
            zorder=3,
            ha='right',
            va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0, linewidth=2))
        
        # Vision image in the top-right region (simulate with random data).
        vision_ax = self.figure.add_axes([0.05, 0.27, 0.9, 0.9])
        vision_ax.imshow(vision)
        vision_ax.set_xticks([])
        vision_ax.set_yticks([])
        vision_ax.patch.set_alpha(0)
        for spine in vision_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        
        # Touch image in the mid-right region (simulate with random grayscale data).
        touch_ax = self.figure.add_axes([0.05, -0.14, 0.9, 0.9])
        touch_ax.imshow(touch, cmap='gray')
        touch_ax.set_xticks([])
        touch_ax.set_yticks([])
        touch_ax.patch.set_alpha(0)
        for spine in touch_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        
        # Command and report text in the lower right region.
        main_ax.text(
            0.94, 0.10, command_voice,
            fontsize=15,
            transform=main_ax.transAxes,
            zorder=3,
            ha='right',
            va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0, linewidth=2))
        main_ax.text(
            0.94, 0.04, report_voice,
            fontsize=15,
            transform=main_ax.transAxes,
            zorder=3,
            ha='right',
            va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0, linewidth=2))
        
        # Redraw the canvas.
        self.canvas.draw()
        self.counter += 1
        # Schedule the next update in 1000 milliseconds (1 second).
        self.after(1000, self.update_plot)

window = VideoPlotWindow()
window.mainloop()
