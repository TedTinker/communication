#%%

import os
import numpy as np
import pybullet as p
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image



robot_name = "two_side_arm"



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
                
        self.sensor_positions = []
        self.sensor_dimensions = []
        
        self.shape_text = self.get_shape_text()
        self.sensor_text = ""  # Initialize as empty
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
        
    def make_sensor(self, i, side, shape, origin, minus, first_plus, second_plus, parts):
        sensor_origin = [
            o if (i + first_plus) % 3 == self.sensor_angle 
            else o - self.shape[i]/2 if (i + second_plus) % 3 == self.sensor_angle and minus 
            else o + self.shape[i]/2 if (i + second_plus) % 3 == self.sensor_angle and not minus 
            else o
            for i, o in enumerate(origin)]
        
        # Apply cumulative transformations
        parent_part = self
        cumulative_origin = [0, 0, 0]
        while parent_part:
            cumulative_origin = [cumulative_origin[j] + parent_part.joint_origin[j] for j in range(3)]
            parent_part = next((part for part in parts if part.name == parent_part.joint_parent), None)
        
        transformed_position = [cumulative_origin[j] + sensor_origin[j] for j in range(3)]
        
        self.sensor_positions.append(transformed_position)
        self.sensor_dimensions.append(shape)
                
        sensor = Part(
            name = f"{self.name}_sensor_{i}_{side}",
            mass = 0,
            shape = shape,
            joint_parent = f"{self.name}",
            joint_origin = sensor_origin,
            joint_axis = self.joint_axis,
            joint_type = "fixed")
        
        return(sensor.get_text())
    
    def get_sensors_text(self, parts):
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
                text += self.make_sensor(i, "start", start_stop_shape, (0, 0, 0), False, 2, 0, parts)
            if(i == self.sensors - 1 and "stop" in self.sensor_sides):
                text += self.make_sensor(i, "stop", start_stop_shape, (0, 0, 0), True, 2, 0, parts)
            if("top" in self.sensor_sides):
                text += self.make_sensor(i, "top", top_bottom_shape, origin, False, 2, 1, parts)
            if("bottom" in self.sensor_sides):
                text += self.make_sensor(i, "bottom", top_bottom_shape, origin, True, 2, 1, parts)
            if("left" in self.sensor_sides):
                text += self.make_sensor(i, "left", left_right_shape, origin, False, 0, 2, parts)
            if("right" in self.sensor_sides):
                text += self.make_sensor(i, "right", left_right_shape, origin, True, 0, 2, parts)
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





        
arm_thickness = .5

parts = [
    Part(
        name = "body", 
        mass = 100, 
        shape = (1, 1, 1),
        sensors = 1,
        sensor_sides = ["start", "stop", "top", "left", "right"]),
    
    Part(
        name = "joint_1", 
        mass = .1, 
        shape = (.4, .3, arm_thickness), 
        joint_parent = "body", 
        joint_origin = (0, .65, 0), 
        joint_axis = (0, -1, 0),
        joint_type = "continuous"),
    
    Part(
        name = "left_arm", 
        mass = .1, 
        shape = (3, .4, arm_thickness), 
        joint_parent = "joint_1", 
        joint_origin = (1.3, .35, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 3,
        sensor_angle = 0),
    
    
    Part(
        name = "left_hand",
        mass = .1,
        shape = (.4, .7, arm_thickness),
        joint_parent = "left_arm", 
        joint_origin = (1.3, -.55, 0), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        sensors = 1,
        sensor_angle = 1,
        sensor_sides = ["top", "bottom", "left", "right", "stop"]),

        
    Part(
        name = "joint_2",
        mass = .1,
        shape = (.4, .3, arm_thickness),
        joint_parent = "body", 
        joint_origin = (0, -.65, 0), 
        joint_axis = (0, -1, 0),
        joint_type = "continuous"),
    
    Part(
        name = "right_arm", 
        mass = .1, 
        shape = (3, .4, arm_thickness), 
        joint_parent = "joint_2", 
        joint_origin = (1.3, -.35, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 3,
        sensor_angle = 0),
    
    Part(
        name = "right_hand",
        mass = .1,
        shape = (.4, .7, arm_thickness),
        joint_parent = "right_arm", 
        joint_origin = (1.3, .55, 0), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        sensors = 1,
        sensor_angle = 1,
        sensor_sides = ["top", "bottom", "left", "right", "start"])
]

for part in parts:
    part.sensor_text = part.get_sensors_text(parts)
    part.joint_text = part.get_joint_text()



add_this = "pybullet_data/" if(os.getcwd().split("/")[-1] != "pybullet_data") else ""

squares_per_side = 9

image = Image.open(f"{add_this}robot_front.png")
image = image.convert("L")
pixels = image.load()
width, height = image.size
front_squares = [(x, -y + squares_per_side - 1) for x in range(width) for y in range(height) if pixels[x, y] == 0]


image = Image.open(f"{add_this}robot_top.png")
image = image.convert("L")
pixels = image.load()
width, height = image.size
top_squares = [(y, x) for x in range(width) for y in range(height) if pixels[x, y] == 0]


image = Image.open(f"{add_this}robot_back.png")
image = image.convert("L")
pixels = image.load()
width, height = image.size
back_squares = [(x, -y + squares_per_side - 1) for x in range(width) for y in range(height) if pixels[x, y] == 0]



i = 0
def make_face(x_y_list, which = "front"):
    global i
    for x, y in x_y_list:
        x = -.5 + (x + .5)/squares_per_side
        y = -.5 + (y + .5)/squares_per_side
        
        if(which == "front"):
            shape = (.002, 1/squares_per_side, 1/squares_per_side)
            joint_origin = (.501, x, y)
        if(which == "top"):
            shape = (1/squares_per_side, 1/squares_per_side, .002)
            joint_origin = (x, y, .501)
        if(which == "back"):
            shape = (.002, 1/squares_per_side, 1/squares_per_side)
            joint_origin = (-.501, x, y)
                        
        parts.append(Part(
            name = f"body_square_{i}",
            mass = 0,
            shape = shape,
            joint_parent = "body", 
            joint_origin = joint_origin, 
            joint_axis = (0, 0, 1),
            joint_type = "fixed"))
        i += 1
    
make_face(front_squares, which = "front")
make_face(top_squares, which = "top")
make_face(back_squares, which = "back")

#"""

robot = \
"""<?xml version="1.0"?>
<robot name="robot">"""
for part in parts:
    robot += part.get_text()
robot += \
"""\n\n\n</robot>"""

current_dir = os.getcwd()
last_folder = os.path.basename(current_dir)
if last_folder == "shapes":
    new_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    os.chdir(new_dir)



# Example usage with actual robot parts data
sensor_positions = []
sensor_dimensions = []
for part in parts:
    sensor_positions.extend(part.sensor_positions)
    sensor_dimensions.extend(part.sensor_dimensions)
sensor_values = [0.1] * len(sensor_positions)  # Adjust values for testing



# Plotting function
def how_to_plot_sensors(sensor_values, sensor_positions = sensor_positions, sensor_dimensions = sensor_dimensions, show = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def draw_sensor(ax, position, dimension, value):
        x, y, z = position
        dx, dy, dz = dimension

        # Create vertices for the sensor box
        vertices = np.array([
            [x - dx / 2, y - dy / 2, z - dz / 2],
            [x + dx / 2, y - dy / 2, z - dz / 2],
            [x + dx / 2, y + dy / 2, z - dz / 2],
            [x - dx / 2, y + dy / 2, z - dz / 2],
            [x - dx / 2, y - dy / 2, z + dz / 2],
            [x + dx / 2, y - dy / 2, z + dz / 2],
            [x + dx / 2, y + dy / 2, z + dz / 2],
            [x - dx / 2, y + dy / 2, z + dz / 2],
        ])

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[4], vertices[7], vertices[3], vertices[0]]
        ]

        poly3d = Poly3DCollection(faces, facecolors=(1, 0, 0, value), linewidths=0.5, edgecolors=(0, 0, 0, .1))
        ax.add_collection3d(poly3d)

    for i, (value, position, dimension) in enumerate(zip(sensor_values, sensor_positions, sensor_dimensions)):
        if(value > 1):
            value = 1
        draw_sensor(ax, position, dimension, value)

    # Set axis limits based on sensor positions
    sensor_positions = np.array(sensor_positions)
    if len(sensor_positions) > 0:
        x_limits = [np.min(sensor_positions[:, 0]), np.max(sensor_positions[:, 0])]
        y_limits = [np.min(sensor_positions[:, 1]), np.max(sensor_positions[:, 1])]
        z_limits = [np.min(sensor_positions[:, 2]), np.max(sensor_positions[:, 2])]
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)

        # Ensure equal aspect ratio and reduce white space
        ax.set_box_aspect([np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)])  

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Remove grid and spines
    ax.grid(False)
    ax.set_axis_off()

    if(show):
        plt.show()
        plt.close()
    else:
        plt.savefig('temp_plot.png')  # Save the plot as an image file
        plt.close()
        image = Image.open('temp_plot.png')
        image_array = np.array(image)
        os.remove('temp_plot.png')  # Delete the temporary image file
        return(image_array)



if(__name__ == "__main__"):
    
    with open(f"robot_{robot_name}.urdf", 'w') as file:
        file.write(robot)
    
    print("\n\n")
    print(robot)
    print("\n\n")
    
    how_to_plot_sensors(sensor_values, show = True)
        

    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath("pybullet_data")

    robot_index = p.loadURDF("{}".format(f"robot_{robot_name}.urdf"), (-5, 0, 0), p.getQuaternionFromEuler([0, 0, pi/2]), 
                                                useFixedBase=False, globalScaling = 2, physicsClientId=physicsClient)
    p.changeVisualShape(robot_index, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = physicsClient)

    for link_index in range(p.getNumJoints(robot_index, physicsClientId = physicsClient)):
        joint_info = p.getJointInfo(robot_index, link_index, physicsClientId = physicsClient)
        link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
        p.changeDynamics(robot_index, link_index, maxJointVelocity = 10000)
        
        if("sensor" in link_name):
            p.changeVisualShape(robot_index, link_index, rgbaColor = (1, 0, 0, .15), physicsClientId = physicsClient)
        else:
            p.changeVisualShape(robot_index, link_index, rgbaColor = (0, 0, 0, 1), physicsClientId = physicsClient)
