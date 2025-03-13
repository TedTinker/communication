#%% 
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class VideoPlotWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dynamic Video Plot Window")
        # Create a Matplotlib Figure with a transparent background.
        self.figure = plt.Figure(figsize=(4, 8), dpi=100, facecolor='none')
        self.figure.patch.set_alpha(0)
        # Embed the figure in a Tkinter canvas.
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.counter = 0
        self.update_plot()  # Start the dynamic update

    def update_plot(self):
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
        vision = np.random.rand(100, 100, 3)  # Replace with your actual vision image
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
        touch = np.random.rand(150, 150)  # Replace with your processed touch image
        touch_ax.imshow(touch, cmap='gray')
        touch_ax.set_xticks([])
        touch_ax.set_yticks([])
        touch_ax.patch.set_alpha(0)
        for spine in touch_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        
        # Command and report text in the lower right region.
        command_voice = "Command: Sample Command"  # Replace with obs.command_voice.human_friendly_text()
        report_voice = "Report: Sample Report"       # Replace with obs.report_voice.human_friendly_text(command=False)
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

if __name__ == "__main__":
    app = VideoPlotWindow()
    app.mainloop()
