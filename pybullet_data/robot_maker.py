#%% 

import os
import numpy as np
import pybullet as p
from math import pi



sensor_width = .02



robot = \
"""<?xml version="1.0"?>
<robot name="robot">"""



def make_part(name, mass, shape):
    global robot
    robot += \
f"""\n\n
    <!-- {name} -->
    <link name="{name}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="{mass}"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{shape[0]} {shape[1]} {shape[2]}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{shape[0]} {shape[1]} {shape[2]}"/>
            </geometry>
        </collision>
    </link>"""
    
    
    
def make_joint(parent, child, origin, axis, type = "fixed"):
    global robot
    robot += \
f"""\n\n
    <!-- Joint: {parent}, {child} -->
    <joint name="{parent}_{child}_joint" type="{type}"> 
        <parent link="{parent}"/> 
        <child link="{child}"/> 
        <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/> 
        <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/> 
    </joint>"""
    
    
    
def add_sensors(name, shape, angle = 0, quantity = 1):
    global robot
    if(quantity == 1):
        origin = (0, 0, 0)
    else:
        origin = (-shape[0]/2 + shape[0]/(2*quantity) if angle == 0 else 0, 
                  -shape[1]/2 + shape[1]/(2*quantity) if angle == 1 else 0, 
                  -shape[2]/2 + shape[2]/(2*quantity) if angle == 2 else 0)
    real_shape = [s if i != angle else s/quantity for i, s in enumerate(shape)]
    i = 0
    robot += \
f"""\n\n
    <!-- {name}_sensor -->
    <link name="{name}_sensor_{i}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{real_shape[0] + sensor_width} {real_shape[1] + sensor_width} {real_shape[2] + sensor_width}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{real_shape[0] + sensor_width} {real_shape[1] + sensor_width} {real_shape[2] + sensor_width}"/>
            </geometry>
        </collision>
    </link>
    
	<!-- Joint: {name}, {name}_sensor_{i} -->
    <joint name="{name}_{name}_sensor_{i}_joint" type="fixed"> 
        <parent link="{name}"/> 
        <child link="{name}_sensor_{i}"/> 
        <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/> 
        <axis xyz="0 0 0"/> 
    </joint>"""
    
    origin = (real_shape[0] if angle == 0 else 0, 
                real_shape[1] if angle == 1 else 0, 
                real_shape[2] if angle == 2 else 0)
    for i in range(1, quantity):
        robot += \
f"""\n\n
    <!-- {name}_sensor -->
    <link name="{name}_sensor_{i}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{real_shape[0] + sensor_width} {real_shape[1] + sensor_width} {real_shape[2] + sensor_width}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{real_shape[0] + sensor_width} {real_shape[1] + sensor_width} {real_shape[2] + sensor_width}"/>
            </geometry>
        </collision>
    </link>
    
    <!-- Joint: {name}_sensor_{i-1}, {name}_sensor_{i} -->
    <joint name="{name}_sensor_{i-1}_{name}_sensor_{i}_joint" type="fixed"> 
        <parent link="{name}_sensor_{i-1}"/> 
        <child link="{name}_sensor_{i}"/> 
        <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/> 
        <axis xyz="0 0 0"/> 
    </joint>"""



make_part(  
    name = "body",          
    mass = 100,     
    shape = (1, 1, 1))
add_sensors(
    name = "body",
    shape = (1, 1, 1),
    angle = 0,
    quantity = 1)



arrow_base_len = .4
arrow_base_start = .15
arrow_head_len = .4
arrow_head_width = .5
arrow_head_layers = 10
arrow_head_part_len = arrow_head_width/arrow_head_layers
make_part(  
    name = "arrow_base",          
    mass = 0,       
    shape = (arrow_base_len, .2, .001))
make_joint( 
    parent = "body",        
    child = "arrow_base", 
    origin = (-.5 + arrow_base_len/2 + arrow_base_start, 0, .50000001),    
    axis = (0, 0, 1),   
    type = "fixed")
for i in range(arrow_head_layers):
    make_part(  
        name = f"arrow_{i}",          
        mass = 0,       
        shape = (arrow_head_part_len, arrow_head_width - (arrow_head_width/arrow_head_layers) * i, .001))
    make_joint( 
        parent = "arrow_base" if i == 0 else f"arrow_{i-1}",        
        child = f"arrow_{i}", 
        origin = (arrow_base_len/2 if i== 0 else arrow_head_part_len/2, 0, 0),    
        axis = (0, 0, 1),   
        type = "fixed")



make_part(  
    name = "left_shoulder", 
    mass = .1,       
    shape = (.4, .5, .8))
make_joint( 
    parent = "body",        
    child = "left_shoulder", 
    origin = (0, .75, 0),    
    axis = (0, -1, 0),   
    type = "continuous")



make_part(  
    name = "left_arm", 
    mass = .1,       
    shape = (3, .4, .8))
add_sensors(
    name = "left_arm",
    shape = (3, .4, .8),
    angle = 0,
    quantity = 3)
make_joint( 
    parent = "left_shoulder",        
    child = "left_arm", 
    origin = (1.3, .25, 0),    
    axis = (1, 0, 0),   
    type = "fixed")



make_part(  
    name = "hand", 
    mass = .1,       
    shape = (.4, 1.6, .8))
add_sensors(
    name = "hand",
    shape = (.4, 1.6, .8),
    angle = 1,
    quantity = 2)
make_joint( 
    parent = "left_arm",        
    child = "hand", 
    origin = (1.3, -1, 0),    
    axis = (0, 0, 0),   
    type = "fixed")



make_part(  
    name = "right_arm", 
    mass = .1,       
    shape = (3, .4, .8))
add_sensors(
    name = "right_arm",
    shape = (3, .4, .8),
    angle = 0,
    quantity = 3)
make_joint( 
    parent = "hand",        
    child = "right_arm", 
    origin = (-1.3, -1, 0),    
    axis = (1, 0, 0),   
    type = "fixed")



make_part(  
    name = "right_shoulder", 
    mass = .1,       
    shape = (.4, .5, .8))
make_joint( 
    parent = "right_arm",        
    child = "right_shoulder", 
    origin = (-1.3, .25, 0),    
    axis = (0, -1, 0),   
    type = "fixed")



robot += \
"""\n\n</robot>"""



current_dir = os.getcwd()
last_folder = os.path.basename(current_dir)
if last_folder == "shapes":
    new_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    os.chdir(new_dir)
    
    

with open("robot.urdf", 'w') as file:
    file.write(robot)
    
    
    
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath("pybullet_data")

robot_index = p.loadURDF("{}".format("robot.urdf"), (-5, 0, 0), p.getQuaternionFromEuler([0, 0, pi/2]), 
                                            useFixedBase=False, globalScaling = 2, physicsClientId=physicsClient)
p.changeVisualShape(robot_index, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = physicsClient)

for link_index in range(p.getNumJoints(robot_index, physicsClientId = physicsClient)):
    joint_info = p.getJointInfo(robot_index, link_index, physicsClientId = physicsClient)
    link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
    p.changeDynamics(robot_index, link_index, maxJointVelocity = 10000)
    
    if("sensor" in link_name):
        p.changeVisualShape(robot_index, link_index, rgbaColor = (1, 0, 0, .5), physicsClientId = physicsClient)
    else:
        p.changeVisualShape(robot_index, link_index, rgbaColor = (0, 0, 0, 1), physicsClientId = physicsClient)
# %%
