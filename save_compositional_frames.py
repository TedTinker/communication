#%% 
import os
import re
import cv2
from pathlib import Path

def get_sorted_image_files(folder):
    def parse_filename(filename):
        match = re.match(r"data_(\d+)\.(\d+|None)\.png", filename)
        if match:
            data_epoch = int(match.group(1))
            smooth_frame = match.group(2)
            smooth_val = -1 if smooth_frame == "None" else int(smooth_frame)
            return (smooth_val == -1, data_epoch, smooth_val)
        return (1, float('inf'), float('inf'))  # put unmatchable ones last

    files = [f for f in os.listdir(folder) if f.endswith(".png")]
    files.sort(key=lambda x: parse_filename(x))
    return [os.path.join(folder, f) for f in files]

def images_to_video(image_folder, output_filename="output.mp4", fps=10):
    image_files = get_sorted_image_files(image_folder)
    if not image_files:
        raise ValueError("No image files found in the folder.")

    # Read the first image to get frame size
    first_frame = cv2.imread(image_files[0])
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for i, img_path in enumerate(image_files):
        print(round(i/len(image_files) * 100, 2))
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Skipping unreadable image {img_path}")
            continue
        video.write(frame)

    video.release()
    print(f"Video saved as {output_filename}")



# hq 
# command_voice_zq



def export_here(arg_name, agent_num, reducer_type, component):
    from_here = f"saved_deigo/thesis_pics/composition/{arg_name}/agent_{agent_num}/{reducer_type}/{component}/lda_050000"
    to_here = f"saved_deigo/thesis_pics/composition/{arg_name}_agent_{agent_num}_{reducer_type}_{component}.mp4"
    images_to_video(from_here, to_here, fps = 10)
    
export_here("ef", "1", "lda", "hq")
