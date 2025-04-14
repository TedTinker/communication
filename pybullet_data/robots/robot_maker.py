#%%

import os
import numpy as np
import pybullet as p
from math import pi
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
from time import sleep
from scipy.spatial.transform import Rotation as R


                
try:
    from .part import Part  
    cluster = True
except ImportError:
    from part import Part  
    cluster = False
    
    
    
add_this = "pybullet_data/robots/" if cluster else ""



robot_dict = {}



def make_robot(robot_name, parts, face = True):
            
    for part in parts:
        part.sensor_text = part.get_sensors_text(parts)
        part.joint_text = part.get_joint_text()

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
    
    image = Image.open(f"{add_this}robot_side.png")
    image = image.convert("L")
    pixels = image.load()
    width, height = image.size
    left_squares = [(x, -y + squares_per_side - 1) for x in range(width) for y in range(height) if pixels[x, y] == 0]
    right_squares = [(x, -y + squares_per_side - 1) for x in range(width) for y in range(height) if pixels[x, y] == 0]

    def make_face(x_y_list, which = "front"):
        face_sizes = []
        face_positions = []
        for face_part_num, (x, y) in enumerate(x_y_list):
            x = -.5 + (x + .5)/squares_per_side
            y = -.5 + (y + .5)/squares_per_side
            
            if(which == "front"):
                size = (.002, 1/squares_per_side, 1/squares_per_side)
                joint_origin = (.501, x, y)
            if(which == "back"):
                size = (.002, 1/squares_per_side, 1/squares_per_side)
                joint_origin = (-.501, x, y)
            if(which == "left"):
                size = (1/squares_per_side, .002, 1/squares_per_side)
                joint_origin = (x, -.501, y)
            if(which == "right"):
                size = (1/squares_per_side, .002, 1/squares_per_side)
                joint_origin = (x, .501, y)
            if(which == "top"):
                size = (1/squares_per_side, 1/squares_per_side, .002)
                joint_origin = (x, y, .501)
            face_sizes.append(size)
            face_positions.append(joint_origin)
            
            parts.append(Part(
                name = f"body_face_{which}_{face_part_num}",
                mass = 0,
                size = size,
                joint_parent = "body", 
                joint_origin = joint_origin, 
                joint_axis = (0, 0, 1),
                joint_type = "fixed"))
            
        return(face_sizes, face_positions)
        
    if(face):
        front_face_sizes, front_face_positions = make_face(front_squares, which = "front")
        back_face_sizes, back_face_positions = make_face(back_squares, which = "back")
        left_face_sizes, left_face_positions = make_face(left_squares, which = "left")
        right_face_sizes, right_face_positions = make_face(right_squares, which = "right")
        top_face_sizes, top_face_positions = make_face(top_squares, which = "top")
    else:
        front_face_sizes, front_face_positions = [], []
        back_face_sizes, back_face_positions = [], []
        left_face_sizes, left_face_positions = [], []
        right_face_sizes, right_face_positions = [], []
        top_face_sizes, top_face_positions = [], []

    face_sizes = front_face_sizes + back_face_sizes + left_face_sizes + right_face_sizes + top_face_sizes
    face_positions = front_face_positions + back_face_positions + left_face_positions + right_face_positions + top_face_positions

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
        
    with open(f"{robot_name}.urdf", 'w') as file:
        file.write(robot)
        
    # Example usage with actual robot parts data
    sensor_positions = []
    sensor_dimensions = []
    sensor_angles = []
    wheel_positions = [] 
    wheel_dimensions = []
    wheel_angles = []
    for part in parts:
        sensor_positions.extend(part.sensor_positions)
        sensor_dimensions.extend(part.sensor_dimensions)
        sensor_angles.extend(part.sensor_angles)
        if("wheel" in part.name):
            wheel_positions.append(part.joint_origin)
            wheel_dimensions.append(part.size)
            wheel_angles.append(part.joint_axis)
    sensor_values = [0] * len(sensor_positions)  # Adjust values for testing

    def apply_rotation(vertices, position, angle):
        rotation = R.from_euler('xyz', angle, degrees=False)  # Ensure radians are used
        rotated_vertices = rotation.apply(vertices - position) + position
        return rotated_vertices



    # Plotting function
    def sensor_plotter(
        sensor_values, 
        sensor_positions = sensor_positions, 
        sensor_dimensions = sensor_dimensions, 
        sensor_angles = sensor_angles, 
        show = False,
        figsize = None,
        save_path = None):
        
        if(figsize == None):
            fig = plt.figure()
        else:
            fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111, projection='3d')

        def draw_sensor(ax, position, dimension, angle, value):
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
            
            vertices = apply_rotation(vertices, np.array(position), np.array(angle))

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

        for i, (value, position, dimension, angle) in enumerate(zip(sensor_values, sensor_positions, sensor_dimensions, sensor_angles)):
            if(value > 1):
                value = 1
            draw_sensor(ax, position, dimension, angle, value)
            
        def draw_face(ax, size, position):
            dx, dy, dz = size
            x, y, z = position
            z += 1

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
            
            vertices = apply_rotation(vertices, np.array(position), np.array(angle))

            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[4], vertices[7], vertices[3], vertices[0]]
            ]

            poly3d = Poly3DCollection(faces, facecolors=(0, 0, 0, .025), linewidths=0.5, edgecolors=(0, 0, 0, .025))
            ax.add_collection3d(poly3d)
            
        for face_size, face_position in zip(face_sizes, face_positions):
            draw_face(ax, face_size, face_position)
            
        def draw_wheel(ax, position, dimension, angle):
            x, y, z = position
            radius, _, _ = dimension
            z += 1  # optional z-offset

            # Circle in XZ plane (flat, like a wheel standing upright)
            num_sides = 60
            theta = np.linspace(0, 2 * np.pi, num_sides)
            circle = np.array([[radius * np.cos(t), 0, radius * np.sin(t)] for t in theta])

            # Rotate and translate
            circle = apply_rotation(circle, np.array([0, 0, 0]), np.array(angle))
            circle += np.array([x, y, z])

            # Make segments between points
            segments = [[circle[i], circle[(i + 1) % num_sides]] for i in range(num_sides)]

            # Draw as wireframe circle
            ring = Line3DCollection(segments, colors=(0, 0, 0, 0.1), linewidths=1)
            ax.add_collection3d(ring)
            
        for i, (position, dimension, angle) in enumerate(zip(wheel_positions, wheel_dimensions, wheel_angles)):
            angle = (0, 0, 0)
            draw_wheel(ax, position, dimension, angle)
            
            

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

        if(save_path != None):
            plt.savefig(save_path, bbox_inches = "tight")
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
        
    robot_dict[robot_name] = (sensor_plotter, sensor_values)
    
    

if(cluster):
    from .robot import parts
else:
    from robot import parts
make_robot("robot", parts)



if(__name__ == "__main__"):
    
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10, physicsClientId = physicsClient)
    p.resetDebugVisualizerCamera(1,90,-89, 3, physicsClientId = physicsClient)
    p.setAdditionalSearchPath("pybullet_data")
    
    num_bots = len(robot_dict)
    for i, robot_name in enumerate(robot_dict.keys()):
        sensor_plotter, sensor_values = robot_dict[robot_name]
        robot_index = p.loadURDF("{}.urdf".format(robot_name), (-1 + num_bots * 10 / 2 - i * 10, 0, 0), p.getQuaternionFromEuler([0, 0, pi/2]), 
                                                    useFixedBase=True, globalScaling = 2, physicsClientId=physicsClient)
        p.changeVisualShape(robot_index, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = physicsClient)
        for link_index in range(p.getNumJoints(robot_index, physicsClientId = physicsClient)):
            joint_info = p.getJointInfo(robot_index, link_index, physicsClientId = physicsClient)
            link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
            p.changeDynamics(robot_index, link_index, maxJointVelocity = 10000)
            if("sensor" in link_name):
                p.changeVisualShape(robot_index, link_index, rgbaColor = (1, 0, 0, 0), physicsClientId = physicsClient)
            elif("spoke" in link_name or "outline" in link_name):
                p.changeVisualShape(robot_index, link_index, rgbaColor = (1, 1, 1, 1), physicsClientId = physicsClient)
            else:
                p.changeVisualShape(robot_index, link_index, rgbaColor = (0, 0, 0, 1), physicsClientId = physicsClient)
        initial_position = (-5, 0, 0)  # Replace with the actual starting position
        initial_orientation = p.getQuaternionFromEuler([0, 0, pi/2])  # Replace with the actual starting orientation
        
        sensor_plotter(sensor_values, show = True, figsize = (10, 10), save_path = f"sensor_plots/{robot_name}_{str(0).zfill(3)}.png")
        for i in range(len(sensor_values)):
            sensor_values[i] = 1
            sensor_plotter(sensor_values, show = True, figsize = (10, 10), save_path = f"sensor_plots/{robot_name}_{str(i+1).zfill(3)}.png")
            sensor_values[i] = 0
        
    # Simulation loop
    while True:
        sleep(0.05)
        p.stepSimulation(physicsClientId=physicsClient)

# %%
