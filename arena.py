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

from utils import shape_map, color_map, task_map, Goal, empty_goal, relative_to, opposite_relative_to, make_objects_and_task, duration, wait_for_button_press#, print



def get_physics(GUI, args, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId = physicsClient)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, -100, physicsClientId = physicsClient)
    p.setTimeStep(args.time_step / args.steps_per_step, physicsClientId=physicsClient)  # More accurate time step
    p.setPhysicsEngineParameter(numSolverIterations=1, numSubSteps=1, physicsClientId=physicsClient)  # Increased solver iterations for potentially better stability
    return(physicsClient)
    
def get_joint_index(body_id, joint_name, physicsClient):
    num_joints = p.getNumJoints(body_id, physicsClientId = physicsClient)
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i, physicsClientId = physicsClient)
        if info[1].decode() == joint_name:
            return i
    return -1  # Return -1 if no joint with the given name is found

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
far = 9
right = near * tan(fov_x_rad / 2)
left = -right
top = near * tan(fov_y_rad / 2)
bottom = -top

agent_upper_starting_pos = 2.02
object_upper_starting_pos = 1.12
object_lower_starting_pos = -8.85

if(__name__ == "__main__"):
    sensor_alpha = .5
else:
    sensor_alpha = 0



class Arena():
    def __init__(self, physicsClient, args):
        self.args = args
        self.physicsClient = physicsClient
        self.objects_in_play = {}
        self.durations = {"watch" : {}, "push" : {}, "pull" : {}, "left" : {}, "right" : {}}
        
        self.upper_starting_pos = object_upper_starting_pos
        self.lower_starting_pos = object_lower_starting_pos
        
        # Make floor and lower level.
        plane_positions = [[0, 0]]
        plane_ids = []
        for position in plane_positions:
            plane_id = p.loadURDF("plane.urdf", position + [0], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
            plane_ids.append(plane_id)
            plane_id = p.loadURDF("plane.urdf", position + [-10], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
            plane_ids.append(plane_id)
            
        # Place robot. 
        self.default_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId = self.physicsClient)
        self.robot_index = p.loadURDF(f"pybullet_data/robots/robot_{self.args.robot_name}.urdf", (0, 0, agent_upper_starting_pos), self.default_orn, useFixedBase=False, globalScaling = self.args.body_size, physicsClientId = self.physicsClient)
        
        self.joint_1_index = get_joint_index(self.robot_index, 'body_joint_1_joint', physicsClient = self.physicsClient)
        if(self.args.robot_name == "two_side_arm"):
            self.joint_2_index = get_joint_index(self.robot_index, 'body_joint_2_joint', physicsClient = self.physicsClient)
        elif(self.args.robot_name == "two_head_arm"):
            self.joint_2_index = get_joint_index(self.robot_index, 'joint_1_joint_2_joint', physicsClient = self.physicsClient)
        else:
            self.joint_2_index = None
        
        p.changeVisualShape(self.robot_index, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = self.physicsClient)
        p.changeDynamics(self.robot_index, -1, maxJointVelocity = 10000)
        self.sensors = []
        for link_index in range(p.getNumJoints(self.robot_index, physicsClientId = self.physicsClient)):
            joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId = self.physicsClient)
            link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
            p.changeDynamics(self.robot_index, link_index, maxJointVelocity = 10000)
            if("sensor" in link_name):
                self.sensors.append((link_index, link_name))
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (1, 0, 0, sensor_alpha), physicsClientId = self.physicsClient)
            else:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (0, 0, 0, 1), physicsClientId = self.physicsClient)
                        
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
        if(self.args.robot_name == "two_side_arm"):
            joint_1_angle = self.args.max_joint_1_angle 
            joint_2_angle = self.args.max_joint_2_angle
        else:
            joint_1_angle = (self.args.max_joint_1_angle + self.args.min_joint_1_angle) / 2
            joint_2_angle = (self.args.max_joint_2_angle + self.args.min_joint_2_angle) / 2
        self.set_joint_angles(joint_1_angle, joint_2_angle)
        self.set_joint_speeds()
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
               
               
        
    def step(self, left_wheel_speed, right_wheel_speed, joint_1_speed, joint_2_speed, verbose = False, sleep_time = None, waiting = False):
        if(self.args.smooth_steps):
            self.smooth_step(left_wheel_speed, right_wheel_speed, joint_1_speed, joint_2_speed, verbose, sleep_time, waiting)
        else:
            self.rough_step(left_wheel_speed, right_wheel_speed, joint_1_speed, joint_2_speed, verbose, sleep_time, waiting)
        
        
   
    def smooth_step(self, left_wheel_speed, right_wheel_speed, joint_1_speed, joint_2_speed, verbose = False, sleep_time = None, waiting = False):
                                
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.object_positions()
        touching = self.touching_any_object()
        for object_index, touch_dict in touching.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
               
        if(waiting): 
            WAITING = wait_for_button_press()
            
        left_wheel_speed_end = relative_to(left_wheel_speed, -self.args.max_wheel_speed, self.args.max_wheel_speed)
        right_wheel_speed_end = relative_to(right_wheel_speed, -self.args.max_wheel_speed, self.args.max_wheel_speed)
        joint_1_speed_end = relative_to(joint_1_speed , -self.args.max_joint_1_speed, self.args.max_joint_1_speed)
        if(self.joint_2_index != None):
            joint_2_speed_end = relative_to(joint_2_speed , -self.args.max_joint_2_speed, self.args.max_joint_2_speed)
            
        left_wheel_speed_start, right_wheel_speed_start = self.get_wheel_speeds()
        change_in_left_wheel = left_wheel_speed_end - left_wheel_speed_start
        change_in_left_wheel_per_step = change_in_left_wheel / self.args.steps_per_step
        
        change_in_right_wheel = right_wheel_speed_end - right_wheel_speed_start
        change_in_right_wheel_per_step = change_in_right_wheel / self.args.steps_per_step
        
        joint_1_speed_start, joint_2_speed_start = self.get_joint_speeds()
        change_in_joint_1 = joint_1_speed_end - joint_1_speed_start
        change_in_joint_1_per_step = change_in_joint_1 / self.args.steps_per_step
        
        if(self.joint_2_index != None):
            change_in_joint_2 = joint_2_speed_end - joint_2_speed_start
            change_in_joint_2_per_step = change_in_joint_2 / self.args.steps_per_step
        
        #print(f"\n\nleft_wheel_speed_start: {left_wheel_speed_start} \tleft_wheel_speed_end: {left_wheel_speed_end} \tchange_in_left_wheel: {change_in_left_wheel} \tchange_in_left_wheel_per_step: {change_in_left_wheel_per_step}")
        #print(f"right_wheel_speed_start: {right_wheel_speed_start} \tright_wheel_speed_end: {right_wheel_speed_end} \tchange_in_right_wheel: {change_in_right_wheel} \tchange_in_right_wheel_per_step: {change_in_right_wheel_per_step}")
        #print(f"\n\njoint_1_speed_start: {joint_1_speed_start} \tjoint_1_speed_end: {joint_1_speed_end} \tchange_in_joint_1: {change_in_joint_1} \tchange_in_joint_1_per_step: {change_in_joint_1_per_step}")
        #print(f"joint_2_speed_start: {joint_2_speed_start} \tjoint_2_speed_end: {joint_2_speed_end} \tchange_in_joint_2: {change_in_joint_2} \tchange_in_joint_2_per_step: {change_in_joint_2_per_step}")
            
        for step in range(self.args.steps_per_step):
            joint_1_step = joint_1_speed_start + change_in_joint_1_per_step * (step + 1)
            if(self.joint_2_index != None):
                joint_2_step = joint_2_speed_start + change_in_joint_2_per_step * (step + 1)
            else:
                joint_2_step = 0
            self.set_joint_speeds(joint_1_step, joint_2_step)          
              
            left_wheel_step = left_wheel_speed_start + change_in_left_wheel_per_step * (step + 1)
            right_wheel_step = right_wheel_speed_start + change_in_right_wheel_per_step * (step + 1)
            #print(f"\nleft_wheel_steed_step {step}: {left_wheel_step}")
            #print(f"right_wheel_steed_step {step}: {right_wheel_step}")
            self.set_wheel_speeds(left_wheel_step, right_wheel_step)     
            
            if(sleep_time != None):
                sleep(sleep_time / self.args.steps_per_step)
            #print(f"Get Wheel Speeds (before step): {self.get_wheel_speeds()}")
            p.stepSimulation(physicsClientId = self.physicsClient)
            #print(f"Get Wheel Speeds (after step): {self.get_wheel_speeds()}")
                        
            
            joint_1_angle, joint_2_angle = self.get_joint_angles()
            if(joint_1_angle > self.args.max_joint_1_angle):
                if(self.args.robot_name == "two_side_arm"):
                    self.set_joint_angles(joint_1_angle = self.args.max_joint_1_angle)
                joint_1_speed = 0
                joint_1_speed_start = change_in_joint_1_per_step = 0
            if(joint_1_angle < self.args.min_joint_1_angle):
                if(self.args.robot_name == "two_side_arm"):
                    self.set_joint_angles(joint_1_angle = self.args.min_joint_1_angle)
                joint_1_speed = 0
                joint_1_speed_start = change_in_joint_1_per_step = 0
                
            if(self.joint_2_index != None):
                if(joint_2_angle > self.args.max_joint_2_angle):
                    self.set_joint_angles(joint_2_angle = self.args.max_joint_2_angle)
                    joint_2_speed = 0
                    joint_2_speed_start = 0 
                    change_in_joint_2_per_step = 0
                if(joint_2_angle < self.args.min_joint_2_angle):
                    self.set_joint_angles(joint_2_angle = self.args.min_joint_2_angle)
                    joint_2_speed = 0
                    joint_2_start = 0 
                    change_in_joint_2_per_step = 0
                                                
            self.face_upward()
                                                    
            touching_now = self.touching_any_object()
            for object_index, touch_dict in touching_now.items():
                for body_part, value in touch_dict.items():
                    if(value):
                        touching[object_index][body_part] += 1/self.args.steps_per_step
                        if(touching[object_index][body_part]) > 1:
                            touching[object_index][body_part] = 1
                                                                    
        self.objects_end = self.object_positions()
        self.objects_touch = touching
            
            
        
    def rough_step(self, left_wheel_speed, right_wheel_speed, joint_1_speed, joint_2_speed, verbose = False, sleep_time = None, waiting = False):
        
        if(sleep_time != None):
            p.setTimeStep(self.args.time_step / self.args.steps_per_step, physicsClientId=self.physicsClient)  # More accurate time step
            
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.object_positions()
        touching = self.touching_any_object()
        for object_index, touch_dict in touching.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
 
        # We'll need to ditch this if using 50 steps.
        if(self.args.robot_name == "two_side_arm"):
            if(joint_1_speed < 0): 
                joint_1_speed = -1
            else:
                joint_1_speed = 1
            if(joint_2_speed < 0):
                joint_2_speed = -1
            else:
                joint_2_speed = 1
                
        left_wheel_speed = relative_to(left_wheel_speed, -self.args.max_wheel_speed, self.args.max_wheel_speed)
        right_wheel_speed = relative_to(right_wheel_speed, -self.args.max_wheel_speed, self.args.max_wheel_speed)
        joint_1_speed = relative_to(joint_1_speed , -self.args.max_joint_1_speed, self.args.max_joint_1_speed)
        if(self.joint_2_index != None):
            joint_2_speed = relative_to(joint_2_speed , -self.args.max_joint_2_speed, self.args.max_joint_2_speed)

        if(waiting): 
            WAITING = wait_for_button_press()
            
        for step in range(self.args.steps_per_step):
            self.set_wheel_speeds(left_wheel_speed, right_wheel_speed)
            self.set_joint_speeds(joint_1_speed, joint_2_speed) 
            if(sleep_time != None):
                sleep(sleep_time / self.args.steps_per_step)
            p.stepSimulation(physicsClientId = self.physicsClient)
            
            # THIS SHOULD NOT BE DONE WITH THE YAW-ARM
            joint_1_angle, joint_2_angle = self.get_joint_angles()
            if(joint_1_angle > self.args.max_joint_1_angle):
                if(self.args.robot_name == "two_side_arm"):
                    self.set_joint_angles(joint_1_angle = self.args.max_joint_1_angle)
                joint_1_speed = 0
            if(joint_1_angle < self.args.min_joint_1_angle):
                if(self.args.robot_name == "two_side_arm"):
                    self.set_joint_angles(joint_1_angle = self.args.min_joint_1_angle)
                joint_1_speed = 0
                
            if(self.joint_2_index != None):
                if(joint_2_angle > self.args.max_joint_2_angle):
                    self.set_joint_angles(joint_2_angle = self.args.max_joint_2_angle)
                    joint_2_speed = 0
                if(joint_2_angle < self.args.min_joint_2_angle):
                    self.set_joint_angles(joint_2_angle = self.args.min_joint_2_angle)
                    joint_2_speed = 0
                
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
            
            
            
    def get_pos_yaw_spe(self, index):
        pos, ors = p.getBasePositionAndOrientation(index, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors, physicsClientId = self.physicsClient)[-1]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
        (vx, vy, _), _ = p.getBaseVelocity(index, physicsClientId=self.physicsClient)
        velocity_vec = np.array([vx, vy])
        spe = float(np.dot(velocity_vec, forward_dir))
        return(pos, yaw, spe)
    
    def face_upward(self):
        pos, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_index, physicsClientId=self.physicsClient)
        orientation = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orientation, physicsClientId=self.physicsClient)
        p.resetBaseVelocity(self.robot_index, linearVelocity=linear_velocity, angularVelocity=angular_velocity, physicsClientId = self.physicsClient)
        
    def set_pos(self, pos = (0, 0)):
        pos = (pos[0], pos[1], agent_upper_starting_pos)
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
        
    def set_yaw(self, yaw = 0):
        orn = p.getQuaternionFromEuler([0, 0, yaw], physicsClientId = self.physicsClient)
        pos, _, _ = self.get_pos_yaw_spe(self.robot_index)
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
        
    def get_robot_velocities(self):
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_index, physicsClientId=self.physicsClient)
        vx, vy, _ = linear_velocity  # Get only x, y velocities
        _, _, wz = angular_velocity  # Get yaw rotation
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        local_vx = cos(yaw) * vx + sin(yaw) * vy  # Forward speed in local frame
        return local_vx, wz  
        
    def get_wheel_speeds(self):
        linear_velocity, angular_velocity = self.get_robot_velocities()
        left_wheel = linear_velocity - (angular_velocity / self.args.angular_scaler)/2
        right_wheel = linear_velocity + (angular_velocity / self.args.angular_scaler)/2
        return left_wheel, right_wheel
            
    def set_wheel_speeds(self, left_wheel_speed = 0, right_wheel_speed = 0):
        linear_velocity = (left_wheel_speed + right_wheel_speed) / 2
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        x = linear_velocity * cos(yaw)
        y = linear_velocity * sin(yaw)
        angular_velocity = (right_wheel_speed - left_wheel_speed) * self.args.angular_scaler
        p.resetBaseVelocity(self.robot_index, linearVelocity=[x, y, 0], angularVelocity=[0, 0, angular_velocity], physicsClientId = self.physicsClient)
        
        
    
    def get_joint_speeds(self):
        joint_1_state = p.getJointState(self.robot_index, self.joint_1_index, physicsClientId=self.physicsClient)[1]  # Get velocity
        if(self.args.robot_name.startswith("two")):
            joint_2_state = p.getJointState(self.robot_index, self.joint_2_index, physicsClientId=self.physicsClient)[1]  # Get velocity
        else:
            joint_2_state = 0  # No joint_2 in this case
        return joint_1_state, joint_2_state
    
    def get_joint_angles(self):
        joint_1_state = p.getJointState(self.robot_index, self.joint_1_index, physicsClientId=self.physicsClient)
        if(self.joint_2_index != None):
            joint_2_state = p.getJointState(self.robot_index, self.joint_2_index, physicsClientId=self.physicsClient)
        else:
            joint_2_state = [0]
        return joint_1_state[0], joint_2_state[0]
    
    def set_joint_speeds(self, joint_1_speed = 0, joint_2_speed = 0):
        p.setJointMotorControl2(self.robot_index, self.joint_1_index, controlMode = p.VELOCITY_CONTROL, targetVelocity = joint_1_speed , physicsClientId=self.physicsClient)
        if(self.joint_2_index != None):
            p.setJointMotorControl2(self.robot_index, self.joint_2_index, controlMode = p.VELOCITY_CONTROL, targetVelocity = joint_2_speed , physicsClientId=self.physicsClient)
        
    def set_joint_angles(self, joint_1_angle = None, joint_2_angle = None):
        if(joint_1_angle == None):
            pass 
        else:
            p.resetJointState(self.robot_index, self.joint_1_index, joint_1_angle, physicsClientId=self.physicsClient)
        if(self.joint_2_index != None):
            if(joint_2_angle == None):
                pass 
            else:
                p.resetJointState(self.robot_index, self.joint_2_index, joint_2_angle, physicsClientId=self.physicsClient)
        

        
    def generate_positions(self, n, distence = 4):
        base_angle = uniform(0, 2 * pi)
        x1 = distence * cos(base_angle)
        y1 = distence * sin(base_angle)
        r = distence 
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
        
    def object_positions(self):
        object_positions = []
        for object_index in self.objects_in_play.values():
            pos, _, _ = self.get_pos_yaw_spe(object_index)
            object_positions.append(pos)
        return(object_positions)
            
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
            agent_pos, agent_ori = p.getBasePositionAndOrientation(self.robot_index, physicsClientId = self.physicsClient)
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
                print(f"Object: {color_map[color_index].name} {shape_map[shape_index].name}")
                print("Movement forward:", round(movement_forward, 2))
                print("Movement left:", round(movement_left, 2))
                print("Angle of movement:", round(v_rx, 2), round(v_ry, 2))
            
            # Is the agent watching an object?
            watching = abs(angle_radians) < pi/6 and not touching and distance <= self.args.watch_distance
                        
            # Is the object pushed/pulled away from its starting position, relative to the agent's starting position and angle?
            pushing = movement_forward >= self.args.push_amount and touching
            pulling = movement_forward <= -self.args.pull_amount and touching and abs(angle_radians) < pi/2
                        
            # Is the object pushed left/right from its starting position, relative to the agent's starting position and angle?
            lefting = movement_left >= self.args.left_right_amount and touching
            righting = movement_left <= -self.args.left_right_amount and touching
            
            
            
            if(verbose):
                print(f"\n\nTouching: {touching}")
                print(f"Watching ({watching}): \t\t{self.durations['watch'][object_index]} steps")
                print(f"Pushing ({pushing}): \n\t{round(movement_forward, 2)} out of {self.args.push_amount}, \t{self.durations['push'][object_index]} steps")
                print(f"Pulling ({pulling}): \n\t{round(movement_forward, 2)} out of {-self.args.pull_amount}, \t{self.durations['pull'][object_index]} steps")
                print(f"Lefting ({lefting}): \n\t{round(movement_left, 2)} out of {self.args.left_right_amount}, \t{self.durations['left'][object_index]} steps")
                print(f"Righting ({righting}): \n\t{round(movement_left, 2)} out of {-self.args.left_right_amount}, \t{self.durations['right'][object_index]} steps\n\n")
                
            if(self.args.consideration):
                active_changes = []
                if pushing:
                    active_changes.append(("pushing", movement_forward))
                if pulling:
                    active_changes.append(("pulling", abs(movement_forward)))  
                if lefting:
                    active_changes.append(("lefting", movement_left))
                if righting:
                    active_changes.append(("righting", abs(movement_left))) 
                if len(active_changes) > 1:
                    active_changes.sort(key=lambda x: x[1], reverse=True)
                    highest_change = active_changes[0][0]
                    pushing, pulling, lefting, righting = False, False, False, False
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
                    print(f"Watching: ({watching})")
                    print(f"Pushing: ({pushing})")
                    print(f"Pulling: ({pulling})")
                    print(f"Lefting: ({lefting})")
                    print(f"Righting: ({righting})\n")
                    
                    
                    
            """if(self.args.hard_mode):
                if(sum([watching, pushing, pulling, lefting, righting]) >= 2):
                    watching = False 
                    pushing = False 
                    pulling = False 
                    lefting = False 
                    righting = False
                if(verbose):
                    print(f"After hard-mode:")
                    print(f"Watching: ({watching})")
                    print(f"Pushing: ({pushing})")
                    print(f"Pulling: ({pulling})")
                    print(f"Lefting: ({lefting})")
                    print(f"Righting: ({righting})\n")"""
                    
            
            
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
            
            
                
            key = (color_map[color_index], shape_map[shape_index])
            new_value = [watched, pushed, pulled, lefted, righted, watching, pushing, pulling, lefting, righting]

            # If there are multiple of the same object, consider them all.
            if key in objects_goals:
                objects_goals[key] = [old or new for old, new in zip(objects_goals[key], new_value)]
            else:
                objects_goals[key] = new_value
                                    
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
            print(f"\nMother voice: \'{mother_voice.human_text}\'")
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
        pos, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        x, y = cos(yaw), sin(yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [pos[0] + x*.1, pos[1] + y*.1, 2], 
            cameraTargetPosition = [pos[0] + x*2, pos[1] + y*2, 2],    
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
    from utils import args
    physicsClient = get_physics(GUI = True, args = args)
    arena = Arena(physicsClient, args = args)