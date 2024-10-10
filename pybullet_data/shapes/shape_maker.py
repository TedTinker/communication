#%% 

import os
import numpy as np
import pybullet as p
from math import pi, sin, cos, radians, tan
import matplotlib.pyplot as plt
from skimage.transform import resize
from copy import deepcopy

max_radius = .6



base = \
f"""
<?xml version="1.0"?>
<robot name="shape">
    <!-- Definition of the base -->

    <link name="base">
        <visual>
            <geometry>
                <cylinder length=".1" radius="{max_radius}"/>
            </geometry>
            <material name="base_material">
                <color rgba="1 1 1 1"/> 
            </material>
        </visual>
        <collision>
            <geometry>
                <cylinder length=".1" radius="{max_radius}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="100"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
    </link>
"""



def innards(lengths, radia):
    text = "<!-- Definition of the shape -->\n"
    
    length_so_far = .05
    for i, (length, radius) in enumerate(zip(lengths, radia)):
        text += \
f""" 
    <link name="{i}">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{length}" radius="{radius}"/>
            </geometry>
            <material name="red">
                <color rgba="1 1 1 1"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <cylinder length="{length}" radius="{radius}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value=".1"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
    </link>

    <joint name="{i}_joint" type="fixed">
        <parent link="base"/>
        <child link="{i}"/>
        <origin xyz="0 0 {length/2 + length_so_far}" rpy="0 0 0"/>
    </joint>
"""
        length_so_far += length
    text += "\n</robot>"
    return(text)

pillar = innards([1], [max_radius])
pole = innards([1], [.2])
dumbbell = innards([.9, .1], [.2, max_radius])
delta = innards([.1] * 10, [max_radius - i/18 for i in range(1, 11)])

hourglass_part = [max_radius - i/11 for i in range(6)]
hourglass_part_reversed = deepcopy(hourglass_part)
hourglass_part_reversed.reverse()
all_hourglass = hourglass_part[1:] + hourglass_part_reversed[1:]
hourglass = innards([.1] * 10, all_hourglass)

current_dir = os.getcwd()
last_folder = os.path.basename(os.getcwd())
if last_folder == "pybullet_data":
    new_dir = os.path.join(current_dir, "shapes")
    os.chdir(new_dir)
    
    
  
shapes = [pillar, pole, dumbbell, delta, hourglass]
names = ["PILLAR", "POLE", "DUMBBELL", "DELTA", "HOURGLASS"]
letters = ["M", "N", "O", "P", "Q"]
file_names = []

for i in range(len(shapes)):
    file_name = f"{letters[i]}_{names[i]}.urdf"
    shape = shapes[i]
    file_names.append(file_name)
    with open(file_name, 'w') as file:
        file.write(base + shape)
        


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath("pybullet_data")

these_colors = [
    (1, 0, 0, 1),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
    (0, 1, 1, 1),
    (1, 0, 1, 1)
]

for index, (i, file_name) in enumerate(zip([-2, -1, 0, 1, 2], file_names)):
    object_index = p.loadURDF("{}".format(file_name), (-5, 15 * i, 0), p.getQuaternionFromEuler([0, 0, pi/2]), 
                                                useFixedBase=False, globalScaling = 2, physicsClientId=physicsClient)
    color = these_colors[index]
    p.changeVisualShape(object_index, -1, rgbaColor = color, physicsClientId = physicsClient)
    for i in range(p.getNumJoints(object_index)):
        p.changeVisualShape(object_index, i, rgbaColor= color, physicsClientId = physicsClient)
  
  
  
fov_x_deg = 90
fov_y_deg = 90
fov_x_rad = radians(fov_x_deg)
fov_y_rad = radians(fov_y_deg)
near = .1
far = 15
right = near * tan(fov_x_rad / 2)
left = -right
top = near * tan(fov_y_rad / 2)
bottom = -top



def photo(pos, image_size):
    x, y = cos(pi), sin(pi)
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[pos[0], pos[1], 1], 
        cameraTargetPosition=[pos[0] + x*2, pos[1] + y*2, 1],
        cameraUpVector=[0, 0, 1], physicsClientId=physicsClient)
    proj_matrix = p.computeProjectionMatrix(left, right, bottom, top, near, far)
    _, _, rgba, depth, _ = p.getCameraImage(
        width=image_size * 2, height=image_size * 2,
        projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow=0,
        physicsClientId=physicsClient)
    rgb = np.divide(rgba[:, :, :-1], 255)
    rgb = resize(rgb, (image_size, image_size, 3))
    return rgb
  
def plot_these(all_rgbs):
    fig, axs = plt.subplots(len(all_rgbs[0]), len(all_rgbs), figsize = (20, 20))
    for i, rgbs in enumerate(all_rgbs):
        for j, rgb in enumerate(rgbs):
            axs[j, i].imshow(rgb)
    fig.show()
    #plt.close()
  

for image_size in [16]:
    all_rgbs = []
    for object_pos in [15 * i for i in [-2, -1, 0, 1, 2]]:
        rgbs = []
        for distance_pos in [p - 5 for p in [2.5, 3, 4, 5, 6, 7]]: 
            rgb = photo((distance_pos, object_pos), image_size)
            rgbs.append(rgb)
        all_rgbs.append(rgbs)
    plot_these(all_rgbs)
  
#p.disconnect(physicsClientId = physicsClient)



  