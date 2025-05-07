#!/usr/bin/env python

import subprocess
import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD


def clean_up_image_folder(image_folder):
	items = os.listdir(image_folder)
	if '.' in items: items.remove('.')
	if '..' in items: items.remove('..')
	if '.DS_Store' in items: items.remove('.DS_Store')
	item_numbers = [int(item.replace('.png', '')) for item in items]
	max_val = max(item_numbers)
	num_len = len(str(max_val))
	for i in range(len(items)):
		old_filepath = os.path.join(image_folder, items[i])
		new_filepath = os.path.join(image_folder, str(item_numbers[i]).zfill(num_len)+'.png')
		os.rename(old_filepath, new_filepath)

def choose_input_folder():
	# selected_folder = filedialog.askopenfilename()
	info['input_folder'] = filedialog.askdirectory()
	left_text.delete('1.0', END)
	left_text.insert(END, info['input_folder'])

def choose_input_file():
	info['input_file'] = filedialog.askopenfilename()
	left_text.delete('1.0', END)
	left_text.insert(END, info['input_file'])

def choose_output_file():
	info['output_file'] = filedialog.asksaveasfilename()
	right_text.delete('1.0', END)
	right_text.insert(END, info['output_file'])

def convert():
	input_path = left_text.get('1.0', END).replace('\n', '')
	output_path = right_text.get('1.0', END).replace('\n', '')
	framerate = framerate_text.get('1.0', END).replace('\n', '')
	try:
		framerate = int(framerate)
		if input_path.endswith('.avi'):
			# subprocess.run('ffmpeg -i {} -pix_fmt yuv420p -strict -2 {}'.format(input_path, info['output_file']), shell=True)
			subprocess.run(info['avi_cmd'].format(input_file=input_path, output_file=output_path))
		else:
			clean_up_image_folder(input_path)
			subprocess.run(info['png_cmd'].format(framerate=framerate, input_folder=input_path, output_file=output_path), shell=True)
		print('Finished creating moving!')
	except Exception as e:
		print(e)


def on_drop_left(event):
	path = event.data
	left_text.delete('1.0', END)
	left_text.insert(END, path)


def on_drop_right(event):
	path = event.data
	right_text.delete('1.0', END)
	right_text.insert(END, path)
	


info = {
	'input_folder': None,
	'output_file': None,
	'png_cmd': 'ffmpeg -framerate {framerate} -pattern_type glob -i \'{input_folder}/*.png\' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {output_file}',
	'avi_cmd': 'ffmpeg -i {input_file} -pix_fmt yuv420p -strict -2 {output_file}'
}


width = 800
height = 500

root = TkinterDnD.Tk()
root.title('Image Converter')
root.geometry('{}x{}'.format(width, height))

left_frame = ttk.Frame(root, width=int(width/2), height=height)
left_frame['padding'] = 5
left_frame['borderwidth'] = 2
left_frame['relief'] = 'groove'
left_frame.pack(side='left')
left_frame.pack_propagate(False)

right_frame = ttk.Frame(root, width=int(width/2), height=height)
right_frame['padding'] = 5
right_frame['borderwidth'] = 2
right_frame['relief'] = 'groove'
right_frame.pack(side='right')
right_frame.pack_propagate(False)

select_input_folder_button = Button(left_frame, text='Input folder', command=choose_input_folder)
select_input_folder_button.pack()
select_input_file_button = Button(left_frame, text='Input file', command=choose_input_file)
select_input_file_button.pack()
left_text = Text(left_frame, height=3)
left_text.drop_target_register(DND_FILES)
left_text.dnd_bind('<<Drop>>', on_drop_left)
left_text.pack()


select_output_button = Button(right_frame, text='Output file', command=choose_output_file)
select_output_button.pack()
right_text = Text(right_frame, height=3)
right_text.drop_target_register(DND_FILES)
right_text.dnd_bind('<<Drop>>', on_drop_right)
right_text.pack()

framerate_text = Text(right_frame, height=1, width=5)
framerate_text.pack()


convert_button = Button(right_frame, text='Convert', command=convert)
convert_button.pack()



root.mainloop()