#%% 

def make_urdf(file_name, widths, heights):
    
    text = """<?xml version="1.0"?>

<robot name="{}">

    <!-- The base --> 
    <link name="base_link">
        <inertial>
            <origin xyz="0 0 .5" rpy="0 0 0"/>
            <mass value="1"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 .5" rpy="0 0 0"/>
            <geometry>
                <cylinder length="1" radius="0.6"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 .5" rpy="0 0 0"/>
            <geometry>
                <cylinder length="1" radius="0.6"/>
            </geometry>
        </collision>
    </link>""".format(file_name)
    
    for i, (width, height) in enumerate(zip(widths, heights)):
        prev = i-1 
        if(prev == -1): prev = "base"
        text += f"""\n
    <!-- {i} -->
    <link name="{i}_link">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{height}" radius="{width}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{height}" radius="{width}"/>
            </geometry>
        </collision>
    </link>

    <!-- Joint from {prev} to {i} -->
    <joint name="{prev}_{i}_joint" type="fixed">
        <parent link="{prev}_link"/>
        <child link="{i}_link"/>
        <origin xyz="0 0 {(heights[i-1]/2 if i > 0 else 0) + heights[i]/2}" rpy="0 0 0"/>
    </joint>"""
    
    text += "\n\n</robot>"
    
    file = open("shapes/{}.urdf".format(file_name), "w")
    file.write(text)
    file.close()

make_urdf("1_pole",     [.2],          [1])
make_urdf("2_T",        [.2, .6],      [.9, .1])
make_urdf("3_L",        [.6, .2],      [.1, .9])
make_urdf("4_cross",    [.2, .6, .2],  [.45, .1, .45])
make_urdf("5_I",        [.6, .2, .6],  [.1, .8, .1])



import pybullet as p
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product
from time import sleep

data_path = "shapes"
urdf_files = [f.name for f in os.scandir(data_path)] ; urdf_files.sort()
colors = [(1,0,0,1),(0,1,0,1),(0,0,1,1),(0,1,1,1),(1,0,1,1),(1,1,0,1)]

def get_physics():
    physicsClient = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(1, 0, 0, (0, 0, 0), physicsClientId = physicsClient)
    p.setAdditionalSearchPath(data_path)
    return(physicsClient)

difference = 1.3

class Arena:

    def __init__(self, image_sizes = [16], distances = [1, 1.5, 2, 4]):
        self.physicsClient = get_physics()
        p.setGravity(0,0,-10)

        for k, urdf_file in enumerate(urdf_files):
            angle = p.getQuaternionFromEuler([0,0,0])
            object = p.loadURDF("{}".format(urdf_file), (difference*k, 0, 0), angle, 
                                useFixedBase=False, physicsClientId=self.physicsClient)
            p.changeVisualShape(object, -1, rgbaColor = (0,0,0,0), physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object)):
                p.changeVisualShape(object, i, rgbaColor=colors[k], physicsClientId = self.physicsClient)
            p.loadURDF("plane.urdf", (10*k,0,0), globalScaling=.5, useFixedBase=True, physicsClientId=self.physicsClient)

        for k, urdf_file in enumerate(urdf_files):
            fig, axs = plt.subplots(len(image_sizes), len(distances), figsize = (5*len(distances), 5*len(image_sizes)))
            for (i,image_size), (j,distance) in product(enumerate(image_sizes), enumerate(distances)):
                ax = axs[i,j] if len(image_sizes) > 1 else axs[j]
                view_matrix = p.computeViewMatrix(
                    cameraEyePosition = [difference*k, distance, .5], 
                    cameraTargetPosition = [difference*k, 0, .5], 
                    cameraUpVector = [0, 0, 1], physicsClientId = self.physicsClient)
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov = 90, aspect = 1, nearVal = .01, 
                    farVal = 10, physicsClientId = self.physicsClient)
                _, _, rgba, _, _ = p.getCameraImage(
                    width=image_size, height=image_size,
                    projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
                    physicsClientId = self.physicsClient)
                rgb = np.divide(rgba[:,:,:-1], 255)
                ax.set_title("image size {}\ndistance {}".format(image_size, distance))
                ax.imshow(rgb)
            plt.suptitle(urdf_file)
            plt.show()
            plt.close()



arena = Arena()

while(True):
    p.stepSimulation(physicsClientId = arena.physicsClient)
    sleep(.1)
# %%
