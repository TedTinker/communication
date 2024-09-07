#%% 

# I like these objects, but need some fine-tuning.

import os
import numpy as np
import pybullet as p
from math import pi, sin, cos, radians, tan
import matplotlib.pyplot as plt
from skimage.transform import resize
from copy import deepcopy

max_radius = .6
min_radius = .25
max_height = 1
min_height = .2



class Shape:
    def __init__(self, number, letter, name, lengths, radia, whites):
        self.file_name = f"{number}_{letter}_{name}.urdf"
        print(f"\n{self.file_name}")
        self.number = number 
        self.letter = letter 
        self.name = name 
        self.length_so_far = 0
        self.base_name = f"{'white' if whites[0] else ''}base"
                        
        self.text =\
f"""
<?xml version="1.0"?>
<robot name="shape">
"""
        self.add_link(length = .1, radius = max_radius, name = self.base_name, mass = 100)
        for i, (l, r, w) in enumerate(zip(lengths, radia, whites[1:])):
            name = f"{'white ' if w else ''}{i}"
            self.add_link(l, r, name)
            self.add_joint(name, l)
        self.text += "\n</robot>"
        
        
        
    def add_link(self, length, radius, name, mass = .1):
        print(length, radius, name, mass)
        self.text += \
f""" 
    <!-- Definition of {name} -->
    
    <link name="{name}">
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
            <mass value="{mass}"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
    </link>
"""
    
    
    
    def add_joint(self, child_link, length):
        self.text +=\
f"""
    <joint name="{child_link}_joint" type="fixed">
        <parent link="{self.base_name}"/>
        <child link="{child_link}"/>
        <origin xyz="0 0 {length/2 + self.length_so_far}" rpy="0 0 0"/>
    </joint>
"""         
        self.length_so_far += length
        
        

i = 0        
numbers = [5, 6, 7, 8, 9]
letters = ["Q", "R", "S", "T", "U"]
shapes = [] 

shapes.append(Shape(numbers[i], letters[i], "PILLAR",   
                    lengths = [max_height],        
                    radia = [max_radius],
                    whites = [False, False]))
i += 1 
shapes.append(Shape(numbers[i], letters[i], "ROD",     
                    lengths = [max_height],        
                    radia = [min_radius],
                    whites = [True, False]))      
i += 1 
shapes.append(Shape(numbers[i], letters[i], "POLE", 
                    lengths = [max_height],   
                    radia = [min_radius],
                    whites = [False, False]))     
i += 1 
shapes.append(Shape(numbers[i], letters[i], "CROWN", 
                    lengths = [max_height - min_height, min_height],   
                    radia = [min_radius, max_radius],
                    whites = [True, False, False]))     
i += 1 
shapes.append(Shape(numbers[i], letters[i], "DUMBBELL", 
                    lengths = [max_height - min_height, min_height],   
                    radia = [min_radius, max_radius],
                    whites = [False, False, False]))     
i += 1 




current_dir = os.getcwd()
last_folder = os.path.basename(os.getcwd())
if last_folder == "pybullet_data":
    new_dir = os.path.join(current_dir, "shapes")
    os.chdir(new_dir)
    


file_names = []

for s in shapes:
    file_names.append(s.file_name)
    with open(s.file_name, 'w') as file:
        file.write(s.text)
        


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
    
    link_name = p.getBodyInfo(object_index)[0].decode('utf-8')
    if "white" not in link_name.lower():
        p.changeVisualShape(object_index, -1, rgbaColor = color, physicsClientId = physicsClient)
    else:
        p.changeVisualShape(object_index, -1, rgbaColor = (1, 1, 1, 1), physicsClientId = physicsClient)
    for i in range(p.getNumJoints(object_index)):
        joint_info = p.getJointInfo(object_index, i, physicsClientId=physicsClient)
        joint_name = joint_info[1].decode("utf-8") 
        if "white" not in joint_name.lower(): 
            p.changeVisualShape(object_index, i, rgbaColor=color, physicsClientId=physicsClient)
  
  
  
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
  

for image_size in [16, 64]:
    all_rgbs = []
    for object_pos in [15 * i for i in [-2, -1, 0, 1, 2]]:
        rgbs = []
        for distance_pos in [p - 5 for p in [2.5, 3, 4, 5, 6, 7]]: 
            rgb = photo((distance_pos, object_pos), image_size)
            rgbs.append(rgb)
        all_rgbs.append(rgbs)
    plot_these(all_rgbs)
  
#p.disconnect(physicsClientId = physicsClient)



  