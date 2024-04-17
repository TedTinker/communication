#%% 

import os
import numpy as np
import pybullet as p
from math import pi



class Part:
    def __init__(
        self, 
        name, 
        mass = 0, 
        shape = (1, 1, 1), 
        joint_parent = None, 
        joint_origin = (0, 0, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 0, 
        sensor_width = .02, 
        sensor_angle = 0,
        sensor_sides = ["start", "stop", "top", "bottom", "left", "right"]):
        
        params = locals()
        for param in params:
            if param != "self":
                setattr(self, param, params[param])
                
        self.shape_text = self.get_shape_text()
        self.sensor_text = self.get_sensors_text()
        self.joint_text = self.get_joint_text()
        
    def get_text(self):
        return(self.shape_text + self.sensor_text + self.joint_text)
        
        
        
    def get_shape_text(self):
        return(
f"""\n\n
    <!-- {self.name} -->
    <link name="{self.name}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="{self.mass}"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{self.shape[0]} {self.shape[1]} {self.shape[2]}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{self.shape[0]} {self.shape[1]} {self.shape[2]}"/>
            </geometry>
        </collision>
    </link>""")
        
        
        
    def make_sensor(self, i, side, shape, origin, minus, first_plus, second_plus):
        sensor = Part(
            name = f"{self.name}_sensor_{i}_{side}",
            mass = 0,
            shape = [s for s in shape],
            joint_parent = f"{self.name}",
            joint_origin = [
                o if (i + first_plus) % 3 == self.sensor_angle 
                else o - self.shape[i]/2 if (i + second_plus) % 3 == self.sensor_angle and minus 
                else o + self.shape[i]/2 if (i + second_plus) % 3 == self.sensor_angle and not minus 
                else o
                for i, o in enumerate(origin)],
            joint_axis = self.joint_axis,
            joint_type = "fixed")
        return(sensor.get_text())
        
        
    
    def get_sensors_text(self):
        if(self.sensors == 0):
            return("")
        text = ""
        if(self.sensors == 1):
            origin = (0, 0, 0)
        else:
            origin = (
                -self.shape[0]/2 + self.shape[0]/(2*self.sensors) if self.sensor_angle == 0 else 0, 
                -self.shape[1]/2 + self.shape[1]/(2*self.sensors) if self.sensor_angle == 1 else 0, 
                -self.shape[2]/2 + self.shape[2]/(2*self.sensors) if self.sensor_angle == 2 else 0)
            
        start_stop_shape = [
            s if (i + 2) % 3 == self.sensor_angle 
            else s if (i + 1) % 3 == self.sensor_angle
            else self.sensor_width 
            for i, s in enumerate(self.shape)]
        
        top_bottom_shape = [
            s / self.sensors if (i + 0) % 3 == self.sensor_angle 
            else s if (i + 2) % 3 == self.sensor_angle
            else self.sensor_width 
            for i, s in enumerate(self.shape)]
        
        left_right_shape = [
            s / self.sensors if (i + 0) % 3 == self.sensor_angle 
            else s if (i + 1) % 3 == self.sensor_angle
            else self.sensor_width 
            for i, s in enumerate(self.shape)]
        
        for i in range(self.sensors):
            if(i == 0 and "start" in self.sensor_sides):
                text += self.make_sensor(i, "start", start_stop_shape, (0, 0, 0), False, 2, 0)
            if(i == self.sensors - 1 and "stop" in self.sensor_sides):
                text += self.make_sensor(i, "stop", start_stop_shape, (0, 0, 0), True, 2, 0)
            if("top" in self.sensor_sides):
                text += self.make_sensor(i, "top", top_bottom_shape, origin, False, 2, 1)
            if("bottom" in self.sensor_sides):
                text += self.make_sensor(i, "bottom", top_bottom_shape, origin, True, 2, 1)
            if("left" in self.sensor_sides):
                text += self.make_sensor(i, "left", left_right_shape, origin, False, 0, 2)
            if("right" in self.sensor_sides):
                text += self.make_sensor(i, "right", left_right_shape, origin, True, 0, 2)
            origin = (
                    origin[0] + self.shape[0]/(self.sensors) if self.sensor_angle == 0 else origin[0], 
                    origin[1] + self.shape[1]/(self.sensors) if self.sensor_angle == 1 else origin[1],
                    origin[2] + self.shape[2]/(self.sensors) if self.sensor_angle == 2 else origin[2])
        return(text)        
        
    
    def get_joint_text(self):
        if(self.joint_parent == None):
            return("")
        return(
f"""\n\n
    <!-- Joint: {self.joint_parent}, {self.name} -->
    <joint name="{self.joint_parent}_{self.name}_joint" type="{self.joint_type}"> 
        <parent link="{self.joint_parent}"/> 
        <child link="{self.name}"/> 
        <origin xyz="{self.joint_origin[0]} {self.joint_origin[1]} {self.joint_origin[2]}" rpy="0 0 0"/> 
        <axis xyz="{self.joint_axis[0]} {self.joint_axis[1]} {self.joint_axis[2]}"/> 
    </joint>""")



parts = [
    Part(
        name = "body", 
        mass = 100, 
        shape = (1, 1, 1),
        sensors = 1),
    
    Part(
        name = "left_shoulder", 
        mass = .1, 
        shape = (.4, .3, .8), 
        joint_parent = "body", 
        joint_origin = (0, .65, 0), 
        joint_axis = (0, -1, 0),
        joint_type = "continuous"),
    
    Part(
        name = "left_arm", 
        mass = .1, 
        shape = (3, .4, .8), 
        joint_parent = "left_shoulder", 
        joint_origin = (1.3, .35, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 3,
        sensor_angle = 0),
    
    Part(
        name = "hand",
        mass = .1,
        shape = (.4, 1.6, .8),
        joint_parent = "left_arm", 
        joint_origin = (1.3, -1, 0), 
        joint_axis = (0, 0, 0),
        joint_type = "fixed",
        sensors = 2,
        sensor_angle = 1,
        sensor_sides = ["top", "bottom", "left", "right"]),
    
    Part(
        name = "right_arm",
        mass = .1,
        shape = (3, .4, .8),
        joint_parent = "hand", 
        joint_origin = (-1.3, -1, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 3,
        sensor_angle = 0),
    
    Part(
        name = "right_shoulder",
        mass = .1,
        shape = (.4, .3, .8),
        joint_parent = "right_arm", 
        joint_origin = (-1.3, .35, 0), 
        joint_axis = (0, -1, 0),
        joint_type = "fixed")
]



arrow_base_len = .3
arrow_base_width = .2
arrow_base_start = .1
arrow_head_len = .5
arrow_head_width = .5
arrow_head_layers = 10
arrow_head_part_len = arrow_head_len/arrow_head_layers

parts.append(Part(
    name = "arrow_base",
    mass = 0,
    shape = (arrow_base_len, arrow_base_width, .001),
    joint_parent = "body", 
    joint_origin = (-.5 + arrow_base_len/2 + arrow_base_start, 0, .50000001), 
    joint_axis = (0, 0, 1),
    joint_type = "fixed"))

for i in range(arrow_head_layers):
    parts.append(
        Part(    
            name = f"arrow_{i}",
            mass = 0,
            shape = (arrow_head_part_len, arrow_head_width - (arrow_head_width/arrow_head_layers) * i, .001),
            joint_parent = "arrow_base" if i == 0 else f"arrow_{i-1}", 
            joint_origin = (arrow_base_len/2 if i == 0 else arrow_head_part_len, 0, 0),    
            joint_axis = (0, 0, 1),
            joint_type = "fixed"))



robot = \
"""<?xml version="1.0"?>
<robot name="robot">"""
for part in parts:
    robot += part.get_text()
robot += \
"""\n\n\n</robot>"""



print("\n\n")
print(robot)
print("\n\n")



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
