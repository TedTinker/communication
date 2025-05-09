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
        print(i/len(image_files))
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Skipping unreadable image {img_path}")
            continue
        video.write(frame)

    video.release()
    print(f"Video saved as {output_filename}")


print(os.getcwd())
images_to_video(
    "saved_deigo/thesis_pics/composition/ef/agent_1/lda/command_voice_zq/lda_050000", 
    "saved_deigo/thesis_pics/composition/ef/agent_1/lda/command_voice_zq/lda_050000.mp4",
    10)
