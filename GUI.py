#%% 
import os
import re
import subprocess
import threading
import tkinter as tk
from tkinter import ttk
from collections import defaultdict
from natsort import natsorted
import shutil  # Added import

num_files_to_create = 10

def parse_slurm_files():
    arg_name_to_slurm_files = defaultdict(list)
    slurm_files = [f for f in os.listdir('.') if f.startswith('slurm-') and f.endswith('.out')]
    for slurm_file in slurm_files:
        with open(slurm_file, 'r') as f:
            lines = f.readlines()
        arg_name = None
        for i, line in enumerate(lines):
            if line.strip().startswith('arg_name:'):
                j = i + 1
                while j < len(lines) and lines[j].startswith('\t'):
                    if 'This time:' in lines[j]:
                        match = re.search(r'This time:\s*(.*)', lines[j])
                        if match:
                            arg_name = match.group(1).strip()
                            break
                    j += 1
                if arg_name:
                    break
        if arg_name:
            arg_name_to_slurm_files[arg_name].append(slurm_file)
    return arg_name_to_slurm_files

class ArgNameData:
    def __init__(self, arg_name):
        self.arg_name = arg_name
        self.slurm_files = []
        self.num_files_to_create = 0
        self.files_created = False
        self.singularity_run = False
        self.files_listbox = None

    def update_slurm_files(self, slurm_files):
        self.slurm_files = slurm_files
        self.num_files_to_create = num_files_to_create
        # Do not reset files_created and singularity_run here

def create_files_for_arg_name(arg_name_data):
    arg_name = arg_name_data.arg_name
    dir_path = os.path.join('communication', 'saved_deigo', arg_name)
    os.makedirs(dir_path, exist_ok=True)
    for i in range(1, arg_name_data.num_files_to_create + 1):
        file_path = os.path.join(dir_path, f'{arg_name}_{i}')
        open(file_path, 'w').close()
    arg_name_data.files_created = True
    arg_name_data.singularity_run = False

def check_files_for_arg_name(arg_name_data):
    if not arg_name_data.files_created:
        return
    if arg_name_data.singularity_run:
        return
    arg_name = arg_name_data.arg_name
    dir_path = os.path.join('communication', 'saved_deigo', arg_name)
    existing_files = [f for f in os.listdir(dir_path) if f.startswith(f'{arg_name}_')]
    if not existing_files:
        run_python_command(arg_name_data)

def run_python_command(arg_data):
    def run_command():
        arg_name = arg_data.arg_name
        cmd = [
            'python', 'communication/finish_dicts.py',
            '--comp', 'deigo',
            '--arg_title', f'___{arg_name}___',
            '--arg_name', 'finishing_dictionaries',
            '--temp', 'True'
        ]
        subprocess.run(cmd)
        # No need to set singularity_run here
    arg_data.singularity_run = True  # Set before starting the thread
    threading.Thread(target=run_command).start()

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Arg Name Monitor')
        self.arg_name_data_dict = {}
        self.arg_name_frames = {}
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.update_interval = 2000  # milliseconds
        self.build_ui()
        self.update_data()

    def build_ui(self):
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = tk.Scrollbar(self.main_frame, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.scrollbar.pack(side="bottom", fill="x")

        # Add a frame for the "All" buttons
        all_buttons_frame = tk.Frame(self.main_frame)
        all_buttons_frame.pack(side="bottom", pady=5)

        # Add the "Create Files for All" button
        all_button = tk.Button(all_buttons_frame, text="Create Files for All", command=self.create_files_for_all)
        all_button.pack(side="left", padx=5)

        # Add the "Delete Files for All" button
        delete_all_button = tk.Button(all_buttons_frame, text="Delete Files for All", command=self.delete_files_for_all)
        delete_all_button.pack(side="left", padx=5)

    def create_files_for_all(self):
        for arg_data in self.arg_name_data_dict.values():
            self.create_files(arg_data)

    def delete_files_for_all(self):
        for arg_data in self.arg_name_data_dict.values():
            self.delete_files(arg_data)

    def update_data(self):
        arg_name_to_slurm_files = parse_slurm_files()
        # Update arg_name_data_dict
        for arg_name in natsorted(arg_name_to_slurm_files.keys()):
            slurm_files = arg_name_to_slurm_files[arg_name]
            if arg_name not in self.arg_name_data_dict:
                arg_data = ArgNameData(arg_name)
                arg_data.update_slurm_files(slurm_files)
                self.arg_name_data_dict[arg_name] = arg_data
                self.add_arg_name_frame(arg_data)
            else:
                arg_data = self.arg_name_data_dict[arg_name]
                arg_data.update_slurm_files(slurm_files)
        # Remove arg_names that no longer exist
        existing_arg_names = set(arg_name_to_slurm_files.keys())
        for arg_name in list(self.arg_name_data_dict.keys()):
            if arg_name not in existing_arg_names:
                self.remove_arg_name_frame(arg_name)
        # Update the frames
        for arg_name, arg_data in self.arg_name_data_dict.items():
            self.update_arg_name_frame(arg_data)
        # Schedule the next update
        self.root.after(self.update_interval, self.update_data)

    def add_arg_name_frame(self, arg_data):
        col_index = len(self.arg_name_frames)  # Calculate column index dynamically
        frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE)
        label = tk.Label(frame, text=f'arg_name: {arg_data.arg_name}')
        label.pack(side=tk.TOP, anchor='w')

        # Create Files button
        create_button = tk.Button(frame, text='Create Files', command=lambda arg_data=arg_data: self.create_files(arg_data))
        create_button.pack(side=tk.TOP, anchor='w')

        # Delete Files button
        delete_button = tk.Button(frame, text='Delete Files', command=lambda arg_data=arg_data: self.delete_files(arg_data))
        delete_button.pack(side=tk.TOP, anchor='w')

        files_label = tk.Label(frame, text='Files in folder:')
        files_label.pack(side=tk.TOP, anchor='w')
        files_listbox = tk.Listbox(frame)
        files_listbox.pack(side=tk.TOP, fill=tk.X, expand=True)
        arg_data.files_listbox = files_listbox
        # Place frame in the correct column and align it to the top
        frame.grid(row=0, column=col_index, padx=5, pady=5, sticky="n")  # sticky="n" aligns to the top
        self.arg_name_frames[arg_data.arg_name] = frame

    def remove_arg_name_frame(self, arg_name):
        frame = self.arg_name_frames.pop(arg_name)
        frame.destroy()
        self.arg_name_data_dict.pop(arg_name)

    def update_arg_name_frame(self, arg_data):
        # Update the files listbox
        dir_path = os.path.join('communication', 'saved_deigo', arg_data.arg_name)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            files.sort()
            arg_data.files_listbox.delete(0, tk.END)
            for f in files:
                arg_data.files_listbox.insert(tk.END, f)
            # Dynamically adjust the Listbox height
            arg_data.files_listbox.config(height=min(len(files), 10))  # Cap at 10 rows for large lists
        else:
            arg_data.files_listbox.delete(0, tk.END)
            arg_data.files_listbox.config(height=1)  # Default to 1 row when no files
        # Check if need to run the finish_dicts.py script
        check_files_for_arg_name(arg_data)

    def create_files(self, arg_data):
        create_files_for_arg_name(arg_data)

    def delete_files(self, arg_data):
        dir_path = os.path.join('communication', 'saved_deigo', arg_data.arg_name)
        if os.path.exists(dir_path):
            for item in os.listdir(dir_path):
                if item != 'agents':
                    item_path = os.path.join(dir_path, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f'Failed to delete {item_path}. Reason: {e}')
            # Update the files_listbox
            self.update_arg_name_frame(arg_data)

if __name__ == '__main__':
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()

# %%
