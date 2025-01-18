#%%
import os
    
import matplotlib.pyplot as plt
from random import uniform
import numpy as np
import pybullet as p
import math
from math import pi, sin, cos, tan, radians, degrees, sqrt, isnan
from time import sleep
from skimage.transform import resize
import tkinter as tk

from utils import args, default_args, shape_map, color_map, task_map, Goal, empty_goal, relative_to, opposite_relative_to, make_objects_and_task, duration#, print



from utils import args, default_args, shape_map, color_map, task_map, Goal, empty_goal, relative_to, opposite_relative_to, make_objects_and_task, duration#, print



from fractions import Fraction

def fraction_pi_string(angle, max_denominator=12):
    """
    Return a string representing 'angle' as a fraction of π, 
    e.g. "π/2", "3π/4", or "2π". Uses 'max_denominator' to keep
    fractions standard (like 1/2, 3/4, 5/6, etc.).
    """
    frac = Fraction(angle / math.pi).limit_denominator(max_denominator)
    num, den = frac.numerator, frac.denominator

    # If denominator = 1, just show "π", "2π", etc.
    if den == 1:
        # e.g. 1π, 2π
        return f"{num}π"
    else:
        # e.g. 1π/2, 3π/4, 5π/3
        return f"{num}π/{den}"

def percentage_between(min_val, current_val, max_val):
    """
    Return a string like "42%" indicating how far 'current_val' is between
    'min_val' and 'max_val'. If max_val == min_val, return "---%" to avoid
    division by zero.
    """
    span = max_val - min_val
    if abs(span) < 1e-9:
        return "---%"
    frac = (current_val - min_val) / span
    return f"{frac * 100:.0f}%"

def print_joint_angles(self, shoulder_pitch_angle, shoulder_roll_angle):
    # Convert min/max to fraction-of-π strings, and middle to percentage
    s1_min_str = fraction_pi_string(self.args.min_shoulder_pitch_angle)
    s1_mid_str = percentage_between(self.args.min_shoulder_pitch_angle, shoulder_pitch_angle, self.args.max_shoulder_pitch_angle)
    s1_max_str = fraction_pi_string(self.args.max_shoulder_pitch_angle)
    
    s2_min_str = fraction_pi_string(self.args.min_shoulder_roll_angle)
    s2_mid_str = percentage_between(self.args.min_shoulder_roll_angle, shoulder_roll_angle, self.args.max_shoulder_roll_angle)
    s2_max_str = fraction_pi_string(self.args.max_shoulder_roll_angle)

    # Find column widths so we can align everything nicely.
    min_col_width = max(len(s1_min_str), len(s2_min_str))
    mid_col_width = max(len(s1_mid_str), len(s2_mid_str))
    max_col_width = max(len(s1_max_str), len(s2_max_str))




def get_link_index_for_name(link_name, robot_index, physicsClient=None):
    num_joints = p.getNumJoints(robot_index, physicsClientId=physicsClient)
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_index, joint_index, physicsClientId=physicsClient)
        # The child link name is at index 12 (as bytes, so decode it)
        child_link_name = joint_info[12].decode('utf-8')
        if child_link_name == link_name:
            return joint_index
    raise ValueError(f"Link name '{link_name}' not found in the robot's URDF.")
    


def wait_for_button_press(button_label="Continue"):
    """
    Displays a tkinter button and waits for the user to click it.

    Parameters:
    button_label (str): The label for the button.

    Returns:
    None
    """
    def on_button_click():
        nonlocal continue_simulation
        continue_simulation = True
        root.destroy()

    # Create the tkinter window
    root = tk.Tk()
    root.title("Wait for Input")
    root.geometry("200x100")

    # Add the button
    button = tk.Button(root, text=button_label, command=on_button_click)
    button.pack(expand=True)

    continue_simulation = False

    # Run the tkinter main loop
    root.mainloop()
    
    

def get_physics(GUI, time_step, steps_per_step, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId = physicsClient)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, -10, physicsClientId = physicsClient)
    p.setTimeStep(time_step / steps_per_step, physicsClientId=physicsClient)  # More accurate time step
    p.setPhysicsEngineParameter(numSolverIterations=1, numSubSteps=1, physicsClientId=physicsClient)  # Increased solver iterations for potentially better stability
    return(physicsClient)
    
def get_joint_index(body_id, joint_name, physicsClient=0):
    num_joints = p.getNumJoints(body_id, physicsClientId = physicsClient)
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i, physicsClientId = physicsClient)
        if info[1].decode() == joint_name:
            return i
    return -1  # Return -1 if no joint with the given name is found

def get_link_index(body_id, link_name, physicsClientId=0):
    """
    Returns the link index of the given link_name in the specified robot_id.
    If no match is found, returns -1.
    """
    n_joints = p.getNumJoints(body_id, physicsClientId=physicsClientId)
    for i in range(n_joints):
        joint_info = p.getJointInfo(body_id, i, physicsClientId=physicsClientId)
        # joint_info[12] is the child link name in bytes, so decode to string
        child_link_name = joint_info[12].decode("utf-8")
        if child_link_name == link_name:
            return i
    return -1

def find_key_by_value(my_dict, target_value):
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None  # If the value is not found

# FOV of agent vision.
fov_x_deg = 90
fov_y_deg = 90
fov_x_rad = radians(fov_x_deg)
fov_y_rad = radians(fov_y_deg)
near = .91
far = 13
right = near * tan(fov_x_rad / 2)
left = -right
top = near * tan(fov_y_rad / 2)
bottom = -top

# I think these heights are okay
agent_upper_starting_pos = .3
object_upper_starting_pos = .3
object_lower_starting_pos = -8.85

if(__name__ == "__main__"):
    sensor_alpha = .5
else:
    sensor_alpha = 0



class Arena():
    def __init__(self, physicsClient, args = default_args):
        self.args = args
        self.physicsClient = physicsClient
        self.objects_in_play = {}
        self.durations = {"watch" : {}, "push" : {}, "pull" : {}, "left" : {}, "right" : {}}
        
        self.upper_starting_pos = object_upper_starting_pos
        self.lower_starting_pos = object_lower_starting_pos
        self.num_bars = 12
        self.hand_radius = .8
        
        floor_friction = 1
        # Make floor and lower level.
        plane_id = p.loadURDF("plane.urdf", [0, 0, -10], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
        p.changeDynamics(plane_id, -1, lateralFriction=floor_friction, physicsClientId=self.physicsClient)
        
        plane_positions = [([20, 20], "white"), ([20, -20], "black"), ([-20, 20], "black"), ([-20, -20], "white")]
        for position, color in plane_positions:
            plane_id = p.loadURDF("plane.urdf", position + [0], globalScaling=.4, useFixedBase=True, physicsClientId=self.physicsClient)
            p.changeDynamics(plane_id, -1, lateralFriction=floor_friction, physicsClientId=self.physicsClient)
            #if(color == "black"):
            #    p.changeVisualShape(plane_id, -1, rgbaColor = [0, 0, 0, 1])
        
        x_dist = 10
        y_dist = 8
        plane_positions = [([8, -8], "black"), ([-8, 8], "black")]
        angle = p.getQuaternionFromEuler([pi/4, pi/2, 0])
        for position, color in plane_positions:
            plane_id = p.loadURDF("plane.urdf", position + [5], angle, globalScaling=.225, useFixedBase=True, physicsClientId=self.physicsClient)
            #p.changeVisualShape(plane_id, -1, rgbaColor = [0, 0, 0, 1])
            p.changeDynamics(plane_id, -1, lateralFriction=floor_friction, physicsClientId=self.physicsClient)
                
            
        # Place robot. 
        self.default_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId = self.physicsClient)
        self.robot_index = p.loadURDF("pybullet_data/robot.urdf", (0, 0, agent_upper_starting_pos), self.default_orn, useFixedBase=True, globalScaling = self.args.body_size, physicsClientId = self.physicsClient)
        
        self.body_index = get_joint_index(self.robot_index, 'base_body_joint', physicsClient = self.physicsClient)
        self.shoulder_pitch_index = get_joint_index(self.robot_index, "body_shoulder_pitch_joint", self.physicsClient)
        self.arm_pull_index = get_joint_index(self.robot_index, "shoulder_pitch_arm_1_joint", self.physicsClient)
        
        for joint_index in [self.shoulder_pitch_index,  self.arm_pull_index]:
            joint_info = p.getJointInfo(self.robot_index, joint_index)
            joint_lower_limit = joint_info[8]
            joint_upper_limit = joint_info[9]
        
            p.setJointMotorControl2(
                bodyIndex=self.robot_index,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                force=1000)
        
        p.changeVisualShape(self.robot_index, -1, rgbaColor = (.5,.5,.5, 1), physicsClientId = self.physicsClient)
        p.changeDynamics(self.robot_index, -1, maxJointVelocity = 10000)
        self.sensors = []
        for link_index in range(p.getNumJoints(self.robot_index, physicsClientId = self.physicsClient)):
            joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId = self.physicsClient)
            link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
            p.changeDynamics(self.robot_index, link_index, maxJointVelocity = 10000)             
            if("sensor" in link_name):
                self.sensors.append((link_index, link_name))
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (1, 0, 0, sensor_alpha), physicsClientId = self.physicsClient)
                p.changeDynamics(self.robot_index, link_index, lateralFriction=0.01, physicsClientId=self.physicsClient)
            elif("face" in link_name):
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (0, 0, 0, 1), physicsClientId = self.physicsClient)
            else:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (.5, .5, .5, 1), physicsClientId = self.physicsClient)
                        

                
        # Place objects on lower level for future use.
        self.loaded = {key : [] for key in shape_map.keys()}
        self.object_indexs = []
        for i, shape in shape_map.items():
            for j in range(2):
                pos = (5*i, 5*j, self.lower_starting_pos)
                object_index = p.loadURDF("pybullet_data/shapes/{}".format(shape.file_name), pos, p.getQuaternionFromEuler([0, 0, pi/2]), 
                                          useFixedBase=False, globalScaling = self.args.object_size, physicsClientId=self.physicsClient)
                p.changeDynamics(object_index, -1, maxJointVelocity = 10000)
                for link_index in range(p.getNumJoints(object_index, physicsClientId = self.physicsClient)):
                    joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId = self.physicsClient)
                    p.changeDynamics(object_index, link_index, maxJointVelocity = 10000)
                    p.changeDynamics(object_index, link_index, lateralFriction=0.01, physicsClientId=self.physicsClient)
                self.loaded[i].append((object_index, (pos[0], pos[1], self.lower_starting_pos)))
                self.object_indexs.append(object_index)
                
                
                
    def end(self):
        for (_, _, idle_pos), object_index in self.objects_in_play.items():
            p.resetBasePositionAndOrientation(object_index, idle_pos, self.default_orn, physicsClientId = self.physicsClient)
            
    def stop(self):
        p.disconnect(physicsClientId = self.physicsClient)
        
        
                                
    def begin(self, objects, goal, parenting, set_positions = None):
        self.set_pos()
        self.set_yaw()
        self.set_wheel_speeds()
        self.set_arm_angles(shoulder_pitch = 1, arm_pull = 0)
        self.set_arm_speeds()
        self.reset_wrist_angles()
        self.goal = goal
        self.parenting = parenting
        
        self.objects_in_play = {}
        self.durations = {"watch" : {}, "push" : {}, "pull" : {}, "left" : {}, "right" : {}}
        already_in_play = {key : 0 for key in shape_map.keys()}
        if(set_positions == None):
            set_positions = self.generate_positions(len(objects))
        for i, (color, shape) in enumerate(objects):
            color_index = find_key_by_value(color_map, color)
            shape_index = find_key_by_value(shape_map, shape)
            object_index, idle_pos = self.loaded[shape_index][already_in_play[shape_index]]
            already_in_play[shape_index] += 1
            x, y = set_positions[i]
            p.resetBasePositionAndOrientation(object_index, (x, y, self.upper_starting_pos), (0, 0, 0, 1), physicsClientId = self.physicsClient)
            self.object_faces_up(object_index)
                
            link_name = p.getBodyInfo(object_index)[0].decode('utf-8')
            if "white" not in link_name.lower():
                p.changeVisualShape(object_index, -1, rgbaColor = color.rgba, physicsClientId = self.physicsClient)
            else:
                p.changeVisualShape(object_index, -1, rgbaColor = (1, 1, 1, 1), physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object_index)):
                joint_info = p.getJointInfo(object_index, i, physicsClientId=self.physicsClient)
                joint_name = joint_info[1].decode("utf-8") 
                if "white" not in joint_name.lower(): 
                    p.changeVisualShape(object_index, i, rgbaColor = color.rgba, physicsClientId=self.physicsClient)
                        
            self.objects_in_play[(color_index, shape_index, idle_pos)] = object_index
            for task in ["watch", "push", "pull", "left", "right"]:
                self.durations[task][object_index] = 0
            
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.object_positions()
        self.objects_end = self.object_positions()
        self.objects_touch = self.touching_any_object()
        for object_index, touch_dict in self.objects_touch.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
               
               
        
    def step(self, wheel, shoulder_pitch, arm_pull, verbose = False, sleep_time = None):
        
        if(shoulder_pitch > 0):
            shoulder_pitch = 1 
        else: 
            shoulder_pitch = -1
                        
        if(sleep_time != None):
            p.setTimeStep(self.args.time_step / self.args.steps_per_step, physicsClientId=self.physicsClient)  # More accurate time step
            
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.object_positions()
        touching = self.touching_any_object()
        for object_index, touch_dict in touching.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 

        if(verbose): 
            #WAITING = wait_for_button_press()
            pass
        for step in range(self.args.steps_per_step):
            shoulder_pitch_angle, arm_pull_amount = self.get_arm_angles()
            if(verbose):
                print_joint_angles(self, shoulder_pitch_angle, arm_pull_amount)
                
            #self.set_pos()
            
            if(shoulder_pitch_angle > self.args.max_shoulder_pitch_angle):
                self.set_arm_angles(shoulder_pitch = 1)
                shoulder_pitch = 0
            if(shoulder_pitch_angle < self.args.min_shoulder_pitch_angle):
                self.set_arm_angles(shoulder_pitch = -1)
                shoulder_pitch = 0
                
            if(arm_pull_amount > self.args.max_arm_pull_amount):
                self.set_arm_angles(arm_pull = 1)
                arm_pull = 0
            if(arm_pull_amount < self.args.min_arm_pull_amount):
                self.set_arm_angles(arm_pull = -1)
                arm_pull = 0
            
            self.set_arm_speeds(shoulder_pitch, arm_pull) 
            self.set_wheel_speeds(wheel)
            if(sleep_time != None):
                sleep(sleep_time / self.args.steps_per_step)
            p.stepSimulation(physicsClientId = self.physicsClient)
            #self.reset_wrist_angles()
                
            touching_now = self.touching_any_object()
            for object_index, touch_dict in touching_now.items():
                for body_part, value in touch_dict.items():
                    if(value):
                        touching[object_index][body_part] += 1/self.args.steps_per_step
                        if(touching[object_index][body_part]) > 1:
                            touching[object_index][body_part] = 1
                
        self.objects_end = self.object_positions()
        self.objects_touch = touching
            
        if(sleep_time != None):
            p.setTimeStep(self.args.time_step / self.args.steps_per_step, physicsClientId=self.physicsClient)  # More accurate time step
            p.setPhysicsEngineParameter(numSolverIterations=1, numSubSteps=1, physicsClientId=self.physicsClient)
            
            
            
    def set_pos(self, pos = (0, 0)):
        pos = (pos[0], pos[1], agent_upper_starting_pos)
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
        
    def set_yaw(self, yaw = 0):
        p.resetJointState(self.robot_index, self.body_index, yaw, physicsClientId=self.physicsClient)
        
    def set_wheel_speeds(self, wheel=0):
        wheel = relative_to(wheel, -self.args.max_speed, self.args.max_speed)
        angular_velocity = wheel * 2 * self.args.angular_scaler
        
        p.setJointMotorControl2(self.robot_index, self.body_index, controlMode = p.VELOCITY_CONTROL, targetVelocity = angular_velocity, physicsClientId=self.physicsClient)
        
    def get_pos_yaw_spe(self, index):
        pos, ors = p.getBasePositionAndOrientation(index, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors, physicsClientId = self.physicsClient)[-1]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
        (vx, vy, _), _ = p.getBaseVelocity(index, physicsClientId=self.physicsClient)
        velocity_vec = np.array([vx, vy])
        spe = float(np.dot(velocity_vec, forward_dir))
        return(pos, yaw, spe)
    
    def set_arm_speeds(self, shoulder_pitch = 0, arm_pull = 0):
        shoulder_pitch = relative_to(shoulder_pitch, -self.args.max_shoulder_speed, self.args.max_shoulder_speed)
        arm_pull = relative_to(arm_pull, -self.args.arm_pull_speed, self.args.arm_pull_speed)
        elbow_pitch = shoulder_pitch
        p.setJointMotorControl2(self.robot_index, self.shoulder_pitch_index, controlMode = p.VELOCITY_CONTROL, targetVelocity = shoulder_pitch, physicsClientId=self.physicsClient)
        p.setJointMotorControl2(self.robot_index, self.arm_pull_index, controlMode = p.VELOCITY_CONTROL, targetVelocity = arm_pull, physicsClientId=self.physicsClient)

    def set_arm_angles(self, shoulder_pitch = None, arm_pull = None):
        if(shoulder_pitch == None):
            pass 
        else:
            shoulder_pitch = relative_to(shoulder_pitch, self.args.min_shoulder_pitch_angle, self.args.max_shoulder_pitch_angle)
            p.resetJointState(self.robot_index, self.shoulder_pitch_index, shoulder_pitch, physicsClientId=self.physicsClient)

        if(arm_pull == None):
            pass 
        else:
            arm_pull = relative_to(arm_pull, self.args.min_arm_pull_amount, self.args.max_arm_pull_amount)
            p.resetJointState(self.robot_index, self.arm_pull_index, arm_pull, physicsClientId=self.physicsClient)
                        
    def get_arm_angles(self):
        shoulder_pitch_state = p.getJointState(self.robot_index, self.shoulder_pitch_index, physicsClientId=self.physicsClient)
        arm_pull_state = p.getJointState(self.robot_index, self.arm_pull_index, physicsClientId=self.physicsClient)
        return shoulder_pitch_state[0], arm_pull_state[0]
    
        
    def reset_wrist_angles(self):
        pass 


        
    def generate_positions(self, n):
        distance = self.args.max_object_distance
        base_angle = uniform(0, 2 * pi)
        x1 = distance * cos(base_angle)
        y1 = distance * sin(base_angle)
        r = distance 
        angle_step = (2 * pi) / n
        positions = [(x1, y1)]
        for i in range(1, n):
            current_angle = base_angle + (i * angle_step)
            x = r * cos(current_angle)
            y = r * sin(current_angle)
            positions.append((x, y))
        return positions
            
    def object_faces_up(self, object_index):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_index, physicsClientId=self.physicsClient)
        agent_pos, _ = p.getBasePositionAndOrientation(self.robot_index, physicsClientId=self.physicsClient)
        delta_x = agent_pos[0] - obj_pos[0]
        delta_y = agent_pos[1] - obj_pos[1]
        angle_to_agent = math.atan2(delta_y, delta_x)
        (roll, pitch, _) = p.getEulerFromQuaternion(obj_orn, physicsClientId=self.physicsClient)
        new_orn = p.getQuaternionFromEuler([0, 0, angle_to_agent if not isnan(angle_to_agent) else 0])
        p.resetBasePositionAndOrientation(object_index, (obj_pos[0], obj_pos[1], self.upper_starting_pos), new_orn, physicsClientId=self.physicsClient)
            
    def touching_object(self, object_index):
        touching = {}
        for sensor_index, link_name in self.sensors:
            touching_this = bool(p.getContactPoints(
                bodyA=self.robot_index, bodyB=object_index, linkIndexA=sensor_index, physicsClientId = self.physicsClient))
            touching[link_name] = 1 if touching_this else 0
        return(touching)
    
    def touching_any_object(self):
        touching = {}
        for object_index in self.objects_in_play.values():
            touching_this_object = self.touching_object(object_index)
            touching[object_index] = touching_this_object
        return(touching)
    
    def object_positions(self):
        object_positions = []
        for object_index in self.objects_in_play.values():
            pos, _, _ = self.get_pos_yaw_spe(object_index)
            object_positions.append(pos)
        return(object_positions)
        
        
        
    def rewards(self, verbose = False):
        win = False
        reward = 0
        v_rx = cos(self.robot_start_yaw)
        v_ry = sin(self.robot_start_yaw)
        
        if(verbose):
            for object_key, object_dict in self.objects_touch.items():
                for link_name, value in object_dict.items():
                    if(value):
                        print(f"Touching {object_key} with {link_name}.")
                        
        objects_goals = {}
                
        for i, ((color_index, shape_index, _), object_index) in enumerate(self.objects_in_play.items()):
            watched = False 
            pushed = False 
            pulled = False 
            lefted = False 
            righted = False
            
            # Is the agent touching the object?
            touching = any(self.objects_touch[object_index].values())
            
            # Distance and angle from agent to object.
            object_pos, _ = p.getBasePositionAndOrientation(object_index, physicsClientId=self.physicsClient)
            agent_pos, _ = p.getBasePositionAndOrientation(self.robot_index, physicsClientId = self.physicsClient)
            link_index = get_link_index(self.robot_index, "body")
            agent_ori = p.getLinkState(self.robot_index, link_index, physicsClientId=self.physicsClient)[1]
            distance_vector = np.subtract(object_pos[:2], agent_pos[:2])
            distance = np.linalg.norm(distance_vector)
            normalized_distance_vector = distance_vector / distance
            rotation_matrix = p.getMatrixFromQuaternion(agent_ori, physicsClientId = self.physicsClient)
            forward_vector = np.array([rotation_matrix[0], rotation_matrix[3]])
            forward_vector /= np.linalg.norm(forward_vector)
            dot_product = np.dot(forward_vector, normalized_distance_vector)
            angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))  
            angle_degrees = degrees(angle_radians)
            cross_product = np.cross(np.append(forward_vector, 0), np.append(normalized_distance_vector, 0))
            if cross_product[2] < 0:  
                angle_radians = -angle_radians
                
            # How is the object moving in relation to the agent?
            (x_before, y_before, z_before) = self.objects_start[i]
            (x_after, y_after, z_after) = self.objects_end[i]
            delta_x = x_after - x_before
            delta_y = y_after - y_before
            movement_forward = delta_x * v_rx + delta_y * v_ry
            movement_left = delta_x * (-v_ry) + delta_y * v_rx
            if(verbose):
                print("Object movement (forward):", round(movement_forward, 2))
                print("Object movement (left):", round(movement_left, 2))
                print("Angle of object movement:", round(v_rx, 2), round(v_ry, 2))
            
            # Is the agent watching an object?
            watching = abs(angle_radians) < pi/6 and not touching and distance <= self.args.watch_distance
                        
            # Is the object pushed/pulled away from its starting position, relative to the agent's starting position and angle?
            pushing = movement_forward >= self.args.push_amount and touching
            pulling = movement_forward <= -self.args.pull_amount and touching and abs(angle_radians) < pi/2
                        
            # Is the object pushed left/right from its starting position, relative to the agent's starting position and angle?
            lefting = movement_left >= self.args.left_right_amount and touching
            righting = movement_left <= -self.args.left_right_amount and touching
            
            if(verbose):
                print(f"\n\nWatching ({watching})")
                print(f"Pushing ({pushing}): \n\t{movement_forward} out of {self.args.push_amount}, Touching: {touching}")
                print(f"Pulling ({pulling}): \n\t{movement_forward} out of {-self.args.pull_amount}, Touching: {touching}")
                print(f"Lefting ({lefting}): \n\t{movement_left} out of {self.args.left_right_amount}, Touching: {touching}")
                print(f"Righting ({righting}): \n\t{movement_left} out of {-self.args.left_right_amount}, Touching: {touching}\n\n")
                    
            # List to hold all active -ing flags and their dramatic changes
            active_changes = []

            if pushing:
                active_changes.append(("pushing", movement_forward))
            if pulling:
                active_changes.append(("pulling", abs(movement_forward)))  # Take absolute value for pulling
            if lefting:
                active_changes.append(("lefting", movement_left))
            if righting:
                active_changes.append(("righting", abs(movement_left)))  # Take absolute value for righting

            # If there are multiple active -ing flags, retain only the one with the highest dramatic change
            if len(active_changes) > 1:
                # Find the -ing with the highest dramatic change
                active_changes.sort(key=lambda x: x[1], reverse=True)
                highest_change = active_changes[0][0]

                # Reset all -ing flags to False
                pushing, pulling, lefting, righting = False, False, False, False

                # Set only the highest_change flag to True
                if highest_change == "pushing":
                    pushing = True
                elif highest_change == "pulling":
                    pulling = True
                elif highest_change == "lefting":
                    lefting = True
                elif highest_change == "righting":
                    righting = True   
                    
            if(verbose):
                print(f"After consideration:")
                print(f"Pushing ({pushing})")
                print(f"Pulling ({pulling})")
                print(f"Lefting ({lefting})")
                print(f"Righting ({righting})\n\n")        
                
            def update_duration(action_name, action_now, object_index, duration_threshold):
                if action_now:
                    self.durations[action_name][object_index] += 1
                else:
                    self.durations[action_name][object_index] = 0
                return self.durations[action_name][object_index] >= duration_threshold
            
            watched = update_duration("watch", watching, object_index, self.args.watch_duration)
            pushed  = update_duration("push",  pushing,  object_index, self.args.push_duration)
            pulled  = update_duration("pull",  pulling,  object_index, self.args.pull_duration)
            lefted  = update_duration("left",  lefting,  object_index, self.args.left_duration)
            righted = update_duration("right", righting, object_index, self.args.right_duration)
            
            if(verbose):
                print(self.durations)
                
            objects_goals[(color_map[color_index], shape_map[shape_index])] = [watched, pushed, pulled, lefted, righted, watching, pushing, pulling, lefting, righting]
                        
        mother_voice = empty_goal
        wrong_object = False
        for (color, shape), (watched, pushed, pulled, lefted, righted, watching, pushing, pulling, lefting, righting) in objects_goals.items():
            # If any one task is accomplished, find the task/color/shape.
            if(sum([watched, pushed, pulled, lefted, righted]) == 1):
                # If the correct object, check the task.
                if(color == self.goal.color and shape == self.goal.shape):
                    if(
                        (self.goal.task.name == "WATCH" and watched and not (pushing or pulling or lefting or righting)) or 
                        (self.goal.task.name == "PUSH" and pushed and not (watching or pulling or lefting or righting)) or
                        (self.goal.task.name == "PULL" and pulled and not (watching or pushing or lefting or righting)) or
                        (self.goal.task.name == "LEFT" and lefted and not (watching or pushing or pulling or righting)) or
                        (self.goal.task.name == "RIGHT" and righted and not (watching or pushing or pulling or lefting))):   
                        win = True 
                        reward = self.args.reward
                # If a task is occuring with a wrong object, no reward.
                else:
                    wrong_object = True
                    
            # Mother's voice reflects ongoing processes
            task_in_progress = None
            if(sum([watching, pushing, pulling, lefting, righting]) == 1):
                if(watching): task_in_progress = task_map[1]
                if(pushing):  task_in_progress = task_map[2]
                if(pulling):  task_in_progress = task_map[3]
                if(lefting):  task_in_progress = task_map[4]
                if(righting): task_in_progress = task_map[5]
                mother_voice = Goal(task_in_progress, color, shape, parenting = False)
                
        if(wrong_object):
            win = False 
            reward = 0

        if(self.goal.task.name == "FREEPLAY"):
            reward = 0
            
        if(verbose):
            print(f"\nTry goal: \'{self.goal.human_text}\'")
            print(f"Done goal: \'{mother_voice.human_text}\'")
            print("Raw reward:", round(reward, 2))
            print("Total reward:", reward)
            print("Win:", win)
                        
        return(reward, win, mother_voice)
    
    
    
    def photo_from_above(self):
        pos, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        x, y = 4 * cos(-3*pi/4), 4 * sin(-3*pi/4)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [pos[0] + x, pos[1] + y, 10], 
            cameraTargetPosition = [pos[0], pos[1], 2],    # Camera / target position very important
            cameraUpVector = [0, 0, 1], physicsClientId = self.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 20, physicsClientId = self.physicsClient)
        _, _, rgba, _, _ = p.getCameraImage(
            width=256, height=256,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.physicsClient)
        return(rgba)
    
    def photo_for_agent(self):
        link_index = get_link_index(self.robot_index, "body")
        pos, ors = p.getLinkState(self.robot_index, link_index, physicsClientId=self.physicsClient)[:2]
        yaw = p.getEulerFromQuaternion(ors, physicsClientId=self.physicsClient)[-1]
        x, y = cos(yaw), sin(yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [pos[0] + x*.64, pos[1] + y*.64, 2], 
            cameraTargetPosition = [pos[0] + x * self.args.max_object_distance, pos[1] + y * self.args.max_object_distance, 1],    
            cameraUpVector = [0, 0, 1], physicsClientId = self.physicsClient)
        proj_matrix = proj_matrix = p.computeProjectionMatrix(left, right, bottom, top, near, far)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=self.args.image_size * 2, height=self.args.image_size * 2,
            #width=self.args.image_size, height=self.args.image_size,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.physicsClient)
        rgb = np.divide(rgba[:,:,:-1], 255)
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
        if(d.max() == d.min()): pass
        else: d = (d - d.min())/(d.max()-d.min())
        rgbd = np.concatenate([rgb, d], axis = -1)
        rgbd = resize(rgbd, (self.args.image_size, self.args.image_size, 4))
        return(rgbd)
        
    
    
if __name__ == "__main__":
    args = default_args
    
    """x = 4
    args.max_steps          = int(args.max_steps * x)
    args.time_step          = args.time_step / x
    args.push_amount        = args.push_amount / x
    args.pull_amount        = args.pull_amount / x
    args.left_right_amount  = args.left_right_amount / x
    args.left_amount        = args.left_right_amount
    args.right_amount       = args.left_right_amount
        
    physicsClient = get_physics(GUI = True, time_step = args.time_step, steps_per_step = args.steps_per_step)
    arena = Arena(physicsClient, args = args)
    sleep_time = .5
    
    def show_them():
        above_rgba = arena.photo_from_above()
        agent_rgbd = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgbd[:,:,:-1])
        plt.show()
        plt.close()
        
    
    
    task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(
        num_objects = 1,
        allowed_tasks_and_weights = [(0, 1)],
        allowed_colors = [0],
        allowed_shapes = [0],
        test = None)
    
    do_these = [
        "show_movements",
        "watch",
        "push",
        "pull",
        "left"
        ]
    
    verbose = True
    
    
    
    if("show_movements" in do_these): 
        print("\nSHOW MOVEMENTS")
        moves = [
            [-1, 0, 0], [1, 0, 0], # Spin
            [0, -1, 0], [0, 1, 0], # Shoulder pitch
            [0, 0, -1], [0, 0, 1], # Shoulder roll
            ]
        
        goal = empty_goal
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(10,0)])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
            
    
    
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\nMove {i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = verbose, sleep_time = sleep_time)
            sleep(.3)
            show_them()
            WAITING = wait_for_button_press()
            reward, win, mother_voice = arena.rewards(verbose = verbose)
            if(win):
                break
        arena.end()
    

    if("watch" in do_these):
        print("\nWATCH")
        goal = Goal(task_map[1], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(3,3)])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
        
        moves = [
           [1, 0, 0],
           [1, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
        ]
        
    
        
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\n{i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = True, sleep_time = sleep_time)
            show_them()
            WAITING = wait_for_button_press()
            #show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        arena.end()
    
        
        
    if("push" in do_these):
        print("\nPUSH")
        goal = Goal(task_map[2], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(args.max_object_distance,0)])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
        
        moves = [
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0], # Got the object!
            [0, 0, .5],
            [0, 0, .3],
            [0, 0, .3],
            [0, 0, .3],
            [0, 0, .3],
            [0, 0, .3]]
        
    
        
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\n{i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = verbose, sleep_time = sleep_time)
            show_them()
            WAITING = wait_for_button_press()
            #show_them()
            reward, win, mother_voice = arena.rewards(verbose = verbose)
            if(win):
                break
        arena.end()
        
            
    if("pull" in do_these):
        print("\nPULL")
        goal = Goal(task_map[3], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(args.max_object_distance,0)])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
        
        moves = [
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0], # Got the object!
            [0, 0, -.5],
            [0, 0, -.5],
            [0, 0, -.5],
            [0, 0, -.5],
            [0, 0, -.5],
            [0, 0, -.5],
            ]
    
        
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\n{i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = verbose, sleep_time = sleep_time)
            show_them()
            WAITING = wait_for_button_press()
            #show_them()
            reward, win, mother_voice = arena.rewards(verbose = verbose)
            if(win):
                break
        arena.end()
    
    
    if("left" in do_these):
        print("\nLEFT")
        goal = Goal(task_map[4], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(args.max_object_distance,0)])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
        
        moves = [
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0], # Got the object!
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            ]
    
        
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\n{i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = verbose, sleep_time = sleep_time)
            show_them()
            WAITING = wait_for_button_press()
            #show_them()
            reward, win, mother_voice = arena.rewards(verbose = verbose)
            if(win):
                break
        arena.end()"""
        
        
    x = 1
        
    physicsClient = get_physics(GUI = True, time_step = args.time_step, steps_per_step = args.steps_per_step)
    arena = Arena(physicsClient, args = args)
    sleep_time = .5
    
    def show_them():
        above_rgba = arena.photo_from_above()
        agent_rgbd = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgbd[:,:,:-1])
        plt.show()
        plt.close()
        
    
    
    task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(
        num_objects = 1,
        allowed_tasks_and_weights = [(0, 1)],
        allowed_colors = [0],
        allowed_shapes = [0],
        test = None)
    
    do_these = [
        #"show_movements",
        #"watch",
        "push",
        "pull",
        "left"
        ]
    
    verbose = True
    
    
    
    if("show_movements" in do_these): 
        print("\nSHOW MOVEMENTS")
        moves = [
            [-1, 0, 0], [1, 0, 0], # Spin
            [0, -1, 0], [0, 1, 0], # Shoulder pitch
            [0, 0, -1], [0, 0, 1], # Shoulder roll
            ]
        
        goal = empty_goal
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(10,0)])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
            
    
    
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\nMove {i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = verbose, sleep_time = sleep_time)
            sleep(.3)
            show_them()
            WAITING = wait_for_button_press()
            reward, win, mother_voice = arena.rewards(verbose = verbose)
            if(win):
                break
        arena.end()
    

    if("watch" in do_these):
        print("\nWATCH")
        
        object_angle = pi/4
        position = (args.max_object_distance * math.sin(object_angle), args.max_object_distance * math.cos(object_angle))
        
        goal = Goal(task_map[1], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [position])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
        
        moves = [
           [.5, 1, 0],
           [0, 1, 0],
           [0, 1, 0],
           [0, 1, 0],
        ]
        
    
        
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\n{i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = True, sleep_time = sleep_time)
            show_them()
            WAITING = wait_for_button_press()
            #show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        arena.end()
    
        
    
    object_angle = pi/2 - .2
    position = (args.max_object_distance * math.sin(object_angle), args.max_object_distance * math.cos(object_angle))
        
        
    if("push" in do_these):
        print("\nPUSH")
        goal = Goal(task_map[2], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [position])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
        
        moves = [
            [0, -1, 0], # Got the object!
            [0, 0, .25],
            [0, 0, .25],
            [0, 0, .25],
            [0, 0, .25]]
        
    
        
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\n{i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = verbose, sleep_time = sleep_time)
            show_them()
            WAITING = wait_for_button_press()
            #show_them()
            reward, win, mother_voice = arena.rewards(verbose = verbose)
            if(win):
                break
        arena.end()
        
            
    if("pull" in do_these):
        print("\nPULL")
        goal = Goal(task_map[3], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [position])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
        
        moves = [
            [0, -1, 0], # Got the object!
            [0, 0, -.5],
            [0, 0, -.5],
            [0, 0, -.5],
            ]
    
        
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\n{i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = verbose, sleep_time = sleep_time)
            show_them()
            WAITING = wait_for_button_press()
            #show_them()
            reward, win, mother_voice = arena.rewards(verbose = verbose)
            if(win):
                break
        arena.end()
    
    
    if("left" in do_these):
        print("\nLEFT")
        goal = Goal(task_map[4], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [position])
        show_them()
        arena.rewards(verbose = verbose)
        WAITING = wait_for_button_press()
        
        moves = [
            [0, -1, 0], # Got the object!
            [.5, 0, 0],
            [.5, 0, 0],
            [.5, 0, 0],
            [.5, 0, 0],
            ]
    
        
        for i, (w, a1, a2) in enumerate(moves):  
            print(f"\n\n{i} : {w, a1, a2}\n\n")
            arena.step(w, a1, a2, verbose = verbose, sleep_time = sleep_time)
            show_them()
            WAITING = wait_for_button_press()
            #show_them()
            reward, win, mother_voice = arena.rewards(verbose = verbose)
            if(win):
                break
        arena.end()
        
        
    goal = Goal(task_map[0], task_map[0],task_map[0], parenting = True)
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [position])
    while(True):
        p.stepSimulation(physicsClientId = arena.physicsClient)
        sleep(.0001)
