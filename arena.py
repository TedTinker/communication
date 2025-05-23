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
import threading
import pkg_resources

from utils import shape_map, color_map, task_map, Goal, empty_goal, relative_to, opposite_relative_to, make_objects_and_task, duration, wait_for_button_press#, print
from arena_navigator import run_tk



def get_physics(GUI, args, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        start_cam = (1, 90, -89, (w/2, h/2, w))
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
        tk_thread = threading.Thread(target=run_tk, args=(physicsClient, start_cam))
        tk_thread.daemon = True
        tk_thread.start()
    else:   
        physicsClient = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId = physicsClient)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, args.gravity, physicsClientId = physicsClient)
    p.setTimeStep(args.time_step, physicsClientId=physicsClient)  # More accurate time step
    p.setPhysicsEngineParameter(numSolverIterations=1, numSubSteps=1, physicsClientId=physicsClient)  # Increased solver iterations for potentially better stability
    return(physicsClient)
    
def get_joint_index(body_id, joint_name, physicsClient):
    num_joints = p.getNumJoints(body_id, physicsClientId = physicsClient)
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i, physicsClientId = physicsClient)
        if info[1].decode() == joint_name:
            return i
    return -1  # Return -1 if no joint with the given name is found

def get_joint_indices(body_id, physicsClient):
    num_joints = p.getNumJoints(body_id, physicsClientId=physicsClient)
    joint_indices = {}
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i, physicsClientId=physicsClient)
        joint_name = info[1].decode()
        joint_name_parts = joint_name.split("_")[:-1]
        if "sensor" not in joint_name and "joint" in joint_name_parts:
            if joint_name_parts[-2] == "joint":
                x = joint_name_parts[-1]
                joint_indices[int(x)] = i
    return joint_indices

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
        self.durations = {"watch" : {}, "be_near": {}, "top": {}, "push" : {}, "left" : {}, "right" : {}}
                        
        # Make floor and lower level.
        plane_positions = [[0, 0]]
        plane_ids = []
        for position in plane_positions:
            plane_id = p.loadURDF(f"pybullet_data/plane.urdf", position + [0], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
            plane_ids.append(plane_id)
            plane_id = p.loadURDF(f"pybullet_data/plane.urdf", position + [-10], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
            plane_ids.append(plane_id)
            
        # Place robot. 
        self.default_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId = self.physicsClient)
                
        robot_urdf_path = f"pybullet_data/robots/{self.args.robot_name}.urdf"
        self.robot_index = p.loadURDF(robot_urdf_path, (0, 0, agent_upper_starting_pos), self.default_orn, useFixedBase=False, globalScaling = self.args.body_size, physicsClientId = self.physicsClient)
        self.joint_indices = get_joint_indices(self.robot_index, physicsClient=self.physicsClient) 
        self.wheel_accelerations = [0, 0]
        self.joint_accelerations = {key: 0 for key in self.joint_indices.keys()}
                        
        p.changeVisualShape(self.robot_index, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = self.physicsClient)
        p.changeDynamics(self.robot_index, -1, maxJointVelocity = 10000)
        self.sensors = []
        self.wheels = []
        for link_index in range(p.getNumJoints(self.robot_index, physicsClientId = self.physicsClient)):
            joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId = self.physicsClient)
            link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
            p.changeDynamics(self.robot_index, link_index, maxJointVelocity = 10000)
            if("wheel" in link_name):
                self.wheels.append((link_index, link_name))
            if("sensor" in link_name):
                self.sensors.append((link_index, link_name))
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (1, 0, 0, 0), physicsClientId = self.physicsClient)
            elif("spoke" in link_name or "outline" in link_name):
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (1, 1, 1, 1), physicsClientId = self.physicsClient)
            else:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (0, 0, 0, 1), physicsClientId = self.physicsClient)
                        
        # Place objects on lower level for future use.
        linearDamping = 1
        angularDamping = 100
        self.loaded = {key : [] for key in shape_map.keys()}
        self.object_indexs = []
        for i, shape in shape_map.items():
            for j in range(2):
                pos = (5*i, 5*j, object_lower_starting_pos)
                shape_urdf_file = f"pybullet_data/shapes/{shape.file_name}"
                object_index = p.loadURDF(shape_urdf_file, pos, p.getQuaternionFromEuler([0, 0, pi/2]), 
                                          useFixedBase=False, globalScaling = self.args.object_size, physicsClientId=self.physicsClient)
                p.changeDynamics(object_index, -1, maxJointVelocity = 10000, linearDamping=linearDamping, angularDamping=angularDamping)
                for link_index in range(p.getNumJoints(object_index, physicsClientId = self.physicsClient)):
                    joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId = self.physicsClient)
                    p.changeDynamics(object_index, link_index, maxJointVelocity = 10000, linearDamping=linearDamping, angularDamping=angularDamping)
                self.loaded[i].append((object_index, (pos[0], pos[1], object_lower_starting_pos)))
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
        joint_angles = {joint_num: (getattr(self.args, f'max_joint_{joint_num}_angle') + getattr(self.args, f'min_joint_{joint_num}_angle')) / 2 for joint_num in self.joint_indices.keys()}
        self.set_joint_angles(joint_angles)
        self.set_joint_speeds()
        self.goal = goal
        self.parenting = parenting
        self.wheel_accelerations = [0, 0]
        self.joint_accelerations = {key: 0 for key in self.joint_indices.keys()}
        
        self.objects_in_play = {}
        self.durations = {"watch" : {}, "be_near" : {}, "top" : {}, "push" : {}, "left" : {}, "right" : {}}
        already_in_play = {key : 0 for key in shape_map.keys()}
        if(set_positions == None):
            set_positions = self.generate_positions(len(objects), self.args.max_object_distance)
        for i, (color, shape) in enumerate(objects):
            color_index = find_key_by_value(color_map, color)
            shape_index = find_key_by_value(shape_map, shape)
            object_index, idle_pos = self.loaded[shape_index][already_in_play[shape_index]]
            already_in_play[shape_index] += 1
            x, y = set_positions[i]
            p.resetBasePositionAndOrientation(object_index, (x, y, object_upper_starting_pos), (0, 0, 0, 1), physicsClientId = self.physicsClient)
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
            for task in ["watch", "be_near", "top", "push", "left", "right"]:
                self.durations[task][object_index] = 0
            
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.get_object_positions()
        self.objects_end = self.get_object_positions()
        self.objects_touch = self.touching_any_object()
        for object_index, touch_dict in self.objects_touch.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
               
        self.objects_local_pos_start = {}
        self.objects_local_pos_end = {}
        self.objects_angle_start = {}
        self.objects_angle_end = {}
        start_agent_pos, start_agent_orn = p.getBasePositionAndOrientation(self.robot_index)
        for object_index in self.objects_in_play.values():
            self.objects_local_pos_start[object_index] = self.get_local_position_of_object(object_index, start_agent_pos, start_agent_orn)
            self.objects_local_pos_end[object_index] = self.get_local_position_of_object(object_index, start_agent_pos, start_agent_orn)
            self.objects_angle_start[object_index] = self.get_object_angle(object_index)
            self.objects_angle_end[object_index] = self.get_object_angle(object_index)
            
            
            
    def step(self, left_wheel_speed, right_wheel_speed, joint_target_positions, verbose = False, sleep_time = None, waiting = False):
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.get_object_positions()

        self.objects_local_pos_start = {}
        start_agent_pos, start_agent_orn = p.getBasePositionAndOrientation(self.robot_index)
        for object_index in self.objects_in_play.values():
            self.objects_local_pos_start[object_index] = self.get_local_position_of_object(object_index, start_agent_pos, start_agent_orn)
            self.objects_angle_start[object_index] = self.get_object_angle(object_index)
        
        touching = self.touching_any_object()
        for object_index, touch_dict in touching.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
               
        if(waiting): 
            WAITING = wait_for_button_press()
            
        left_wheel_speed_end = relative_to(left_wheel_speed, -self.args.max_wheel_speed, self.args.max_wheel_speed)
        right_wheel_speed_end = relative_to(right_wheel_speed, -self.args.max_wheel_speed, self.args.max_wheel_speed)
        
        left_wheel_speed_start, right_wheel_speed_start = self.get_wheel_speeds()
        change_in_left_wheel = left_wheel_speed_end - left_wheel_speed_start
        change_in_left_wheel_per_step = change_in_left_wheel / self.args.steps_per_step
        
        change_in_right_wheel = right_wheel_speed_end - right_wheel_speed_start
        change_in_right_wheel_per_step = change_in_right_wheel / self.args.steps_per_step   
        
        for key, value in joint_target_positions.items():
            joint_target_positions[key] = relative_to(
                joint_target_positions[key], 
                getattr(self.args, f'min_joint_{key}_angle'), 
                getattr(self.args, f'max_joint_{key}_angle'))
            
        for step in range(self.args.steps_per_step):   
            left_wheel_step = left_wheel_speed_start + change_in_left_wheel_per_step * (step + 1)
            right_wheel_step = right_wheel_speed_start + change_in_right_wheel_per_step * (step + 1)
            #print(f"\nleft_wheel_steed_step {step}: {left_wheel_step}")
            #print(f"right_wheel_steed_step {step}: {right_wheel_step}")
            self.set_wheel_speeds(left_wheel_step, right_wheel_step)     
            
            if(sleep_time != None):
                sleep(sleep_time / self.args.steps_per_step)
            p.stepSimulation(physicsClientId = self.physicsClient)
            self.face_upward()
                                                                                                    
            touching_now = self.touching_any_object()
            for object_index, touch_dict in touching_now.items():
                for body_part, value in touch_dict.items():
                    if(value):
                        touching[object_index][body_part] += 1/self.args.steps_per_step
                        if(touching[object_index][body_part]) > 1:
                            touching[object_index][body_part] = 1
                            
            joint_target_positions = self.fix_joints(joint_target_positions) # This should output something to make the joint's target into the min/max if needed.
            self.set_joint_target_positions(joint_target_positions)  
                                                                    
        self.objects_end = self.get_object_positions()
        self.objects_touch = touching
        
        self.objects_local_pos_end = {}
        stop_agent_pos, stop_agent_orn = p.getBasePositionAndOrientation(self.robot_index)
        for object_index in self.objects_in_play.values():
            self.objects_local_pos_end[object_index] = self.get_local_position_of_object(object_index, stop_agent_pos, stop_agent_orn)
            self.objects_angle_end[object_index] = self.get_object_angle(object_index)
        
        
        
        
    # Functions for objects
    def generate_positions(self, n, distance):
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
        p.resetBasePositionAndOrientation(object_index, (obj_pos[0], obj_pos[1], object_upper_starting_pos), new_orn, physicsClientId=self.physicsClient)
        
    def get_object_positions(self):
        object_positions = {}
        for object_index in self.objects_in_play.values():
            pos, _, _ = self.get_pos_yaw_spe(object_index)
            object_positions[object_index] = pos
        return(object_positions)
    
    def get_local_position_of_object(self, object_id, agent_pos, agent_orn):
        inv_agent_pos, inv_agent_orn = p.invertTransform(agent_pos, agent_orn)
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
        local_obj_pos, _ = p.multiplyTransforms(
            inv_agent_pos, inv_agent_orn,  
            obj_pos, obj_orn)
        return local_obj_pos # (x_local, y_local, z_local)
    
    def get_object_angle(self, object_index):
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
        return(angle_radians)
            
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
            
            
            
    # Functions for agent positions/angles
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
        linear_velocity = [linear_velocity[0], linear_velocity[1], 0]
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
            
    
    
    # Functions for agent speed
    def set_wheel_speeds(self, left_wheel_speed = 0, right_wheel_speed = 0):
        linear_velocity = (left_wheel_speed + right_wheel_speed) / 2
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        x = linear_velocity * cos(yaw)
        y = linear_velocity * sin(yaw)
        angular_velocity = (right_wheel_speed - left_wheel_speed) * self.args.angular_scaler
        p.resetBaseVelocity(self.robot_index, linearVelocity=[x, y, 0], angularVelocity=[0, 0, angular_velocity], physicsClientId = self.physicsClient)
        for index, name in self.wheels:
            if(name == "left_wheel"):
                speed = left_wheel_speed 
            else:
                speed = right_wheel_speed
            p.setJointMotorControl2(self.robot_index, index, controlMode = p.VELOCITY_CONTROL, targetVelocity = -4 * speed, physicsClientId=self.physicsClient)

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
    
    def get_joint_speeds(self):
        joint_speeds = {}
        for key, index in self.joint_indices.items(): 
            joint_speeds[key] = p.getJointState(self.robot_index, index, physicsClientId=self.physicsClient)[1]  
        return joint_speeds
        
        

    # Functions for agent joints
    def set_joint_speeds(self, joint_speeds = None):
        if(joint_speeds == None):
            joint_speeds = {key : 0 for key in self.joint_indices}
        for key, index in self.joint_indices.items():
            if(joint_speeds[key] != None):
                p.setJointMotorControl2(self.robot_index, index, controlMode = p.VELOCITY_CONTROL, targetVelocity = joint_speeds[key] , physicsClientId=self.physicsClient)
        
    def set_joint_angles(self, joint_angles = None):
        if(joint_angles == None):
            joing_angles = {key: None for key in self.joint_indices}
        for key, index in self.joint_indices.items():
            if(joint_angles[key] != None):
                p.resetJointState(self.robot_index, index, joint_angles[key], physicsClientId=self.physicsClient)
                
    def set_joint_target_positions(self, joint_target_positions = None):
        if(joint_target_positions == None):
            joint_target_positions = {key: None for key in self.joint_indices}
        for key, index in self.joint_indices.items():
            if(joint_target_positions[key] != None):
                p.setJointMotorControl2(self.robot_index, index, controlMode=p.POSITION_CONTROL, 
                        targetPosition=joint_target_positions[key], force=self.args.force)
                
    def get_joint_speeds(self):
        joint_speeds = {}
        for key, index in self.joint_indices.items(): 
            joint_speeds[key] = p.getJointState(self.robot_index, index, physicsClientId=self.physicsClient)[1]  
        return joint_speeds

    def get_joint_angles(self):
        joint_angles = {}
        for key, index in self.joint_indices.items():
            joint_angles[key] = p.getJointState(self.robot_index, index, physicsClientId=self.physicsClient)[0]
        return joint_angles
    
    def fix_joints(self, joint_target_positions):
        joint_angles = self.get_joint_angles()
        joint_speeds = self.get_joint_speeds()
        new_joint_speeds = {key: None for key in self.joint_indices.keys()}
        for key in self.joint_indices.keys():
            max_angle = getattr(self.args, f'max_joint_{key}_angle')
            min_angle = getattr(self.args, f'min_joint_{key}_angle')
            max_speed = self.args.max_joint_speed
            
            if(joint_angles[key] > max_angle):     
                if(joint_target_positions[key] >= max_angle):
                    joint_target_positions[key] = max_angle - .01
            if(joint_angles[key] < min_angle):
                if(joint_target_positions[key] <= min_angle):
                    joint_target_positions[key] = min_angle + .01
                
            if(joint_speeds[key] > max_speed):
                joint_speeds[key] = max_speed
            if(joint_speeds[key] < -max_speed):
                joint_speeds[key] = -max_speed
                
        return(joint_target_positions)
        
        
        
    def rewards(self, verbose = False):
        win = False
        reward = 0
        v_rx = cos(self.robot_start_yaw)
        v_ry = sin(self.robot_start_yaw)
        
        if(verbose):
            printed_touching = False
            for object_key, object_dict in self.objects_touch.items():
                for link_name, value in object_dict.items():
                    if(value):
                        print(f"Touching {object_key} with {link_name}.")
                        printed_touching = True 
            if(printed_touching):
                print("")
                        
        objects_goals = {}
                
        for i, ((color_index, shape_index, _), object_index) in enumerate(self.objects_in_play.items()):
            watched = False 
            been_near = False
            topped = False
            pushed = False 
            lefted = False 
            righted = False
            
            # Is the agent touching the object?
            objects_touch = self.objects_touch[object_index]
            objects_touch_body = {key: value for key, value in objects_touch.items() if "body" in key}
            touching = any(objects_touch.values())
            touching_body = any(objects_touch_body.values())
            
            # Distance and angle from agent to object.
            object_pos, _ = p.getBasePositionAndOrientation(object_index, physicsClientId=self.physicsClient)
            agent_pos, agent_ori = p.getBasePositionAndOrientation(self.robot_index, physicsClientId = self.physicsClient)
            distance_vector = np.subtract(object_pos[:2], agent_pos[:2])
            distance = np.linalg.norm(distance_vector)
                
            # How is the object moving in relation to the agent?
            (x_before, y_before, z_before) = self.objects_start[object_index]
            (x_after, y_after, z_after) = self.objects_end[object_index]
            delta_x = x_after - x_before
            delta_y = y_after - y_before
            global_movement_forward = delta_x * v_rx + delta_y * v_ry
            global_movement_left = delta_x * (-v_ry) + delta_y * v_rx
            
            # Not used yet: local positions, not global positions.
            object_local_pos_start = self.objects_local_pos_start[object_index]
            object_local_pos_end = self.objects_local_pos_end[object_index]
            local_movement_forward = object_local_pos_end[0] - object_local_pos_start[0]
            local_movement_left = object_local_pos_end[1] - object_local_pos_start[1]

            # Changes in angle relative of agent.
            object_angle_start = self.objects_angle_start[object_index]
            object_angle_end = self.objects_angle_end[object_index]
            angle_change = object_angle_end - object_angle_start
            object_angle_start_degrees = degrees(object_angle_start)
            object_angle_end_degrees = degrees(object_angle_end)
            angle_change_degrees = degrees(angle_change)
            
            if(verbose):
                print(f"Object: {color_map[color_index].name} {shape_map[shape_index].name}")
                print(f"Angle from agent to object: \t{round(object_angle_start_degrees, 2)} before, \t{round(object_angle_end_degrees, 2)} after, \t{round(angle_change_degrees, 2)} change")
                print(f"Movement forward: \t{round(global_movement_forward, 2)} global, \t{round(local_movement_forward, 2)} local")
                print(f"Movement left: \t\t{round(global_movement_left, 2)} global, \t{round(local_movement_left, 2)} local")
                print(f"Angle of movement: {round(v_rx, 2), round(v_ry, 2)}")
            
            pointing_at_object_for_watch = abs(object_angle_end) < self.args.pointing_at_object_for_watch
            pointing_at_object_for_left = abs(object_angle_end) < self.args.pointing_at_object_for_left
            
            # Is the agent watching an object?
            watching = pointing_at_object_for_watch and not touching and distance <= self.args.watch_distance
            
            # Is the agent near an object?
            being_near = watching and distance <= self.args.be_near_distance
            
            
            
            # Let's try using local push: large for pushing, small for touching top.
            #print(f"\n\nglobal_movement_forward: {global_movement_forward}. \nglobal_movement_left: {global_movement_left}\n\n")
            #print(f"\n\nlocal_movement_forward: {local_movement_forward}. \nlocal_movement_left: {local_movement_left}\n\n")
            
            
            
            # Is the object touched by the arm, while the arm-angle is high?
            topping = pointing_at_object_for_watch and touching and not touching_body and -self.get_joint_angles()[2] >= self.args.top_arm_min_angle            
                                    
            # Is the object pushed away from its starting position, relative to the agent's starting position and angle?
            pushing = touching and (global_movement_forward >= self.args.global_push_amount) and pointing_at_object_for_watch 
                        
            # Is the object pushed left/right from its starting position, relative to the agent's starting position and angle?
            left_wheel_speed, right_wheel_speed = self.get_wheel_speeds()
            good_speed = max([abs(left_wheel_speed), abs(right_wheel_speed)]) < self.args.max_wheel_speed_for_left
            arm_speed = self.get_joint_speeds()[1]
            good_arm_speed_left = arm_speed > self.args.min_arm_speed_for_left
            good_arm_speed_right = arm_speed < -self.args.min_arm_speed_for_left
            good_arm_speed = good_arm_speed_left or good_arm_speed_right
            lefting = touching and (global_movement_left >= self.args.global_left_right_amount) and pointing_at_object_for_left and good_speed and good_arm_speed_left
            righting = touching and (global_movement_left <= -self.args.global_left_right_amount) and pointing_at_object_for_left and good_speed  and good_arm_speed_right            
            
            if(verbose):
                print(f"\n\nTouching: {touching}. Touching body: {touching_body}.")
                print(f"Watching ({watching}): \t\t{round(angle_change_degrees, 2)} degrees out of {round(degrees(self.args.pointing_at_object_for_watch))} limit \t{self.durations['watch'][object_index]} steps")
                print(f"Being Near ({being_near}): \t\t{round(angle_change_degrees, 2)} degrees out of {round(degrees(self.args.pointing_at_object_for_watch))} limit \t{round(distance, 2)} units out of {round(self.args.be_near_distance, 2)} \t{self.durations['be_near'][object_index]} steps")
                print(f"Topping ({topping}): \n\t{-round(degrees(self.get_joint_angles()[2]), 2)} degrees vs {round(degrees(self.args.top_arm_min_angle), 2)} limit \t{self.durations['top'][object_index]} steps")
                print(f"Pushing ({pushing}): \n\t{round(global_movement_forward, 2)} out of {self.args.global_push_amount} global, \t{round(object_angle_end_degrees, 2)} degrees out of {round(degrees(self.args.pointing_at_object_for_watch))} limit \t{self.durations['push'][object_index]} steps")
                print(f"Lefting ({lefting}): \n\t{round(global_movement_left, 2)} out of {self.args.global_left_right_amount} global, \t{round(degrees(object_angle_end), 2)} degrees out of {round(degrees(self.args.pointing_at_object_for_left))} limit \t{round(left_wheel_speed, 2), round(right_wheel_speed, 2)} out of {self.args.max_wheel_speed_for_left} wheel speed\t{round(arm_speed, 2)} out of {self.args.min_arm_speed_for_left} arm speed\t{self.durations['left'][object_index]} steps")
                print(f"Righting ({righting}): \n\t{round(global_movement_left, 2)} out of {-self.args.global_left_right_amount} global, \t{round(degrees(object_angle_end), 2)} degrees out of {round(degrees(self.args.pointing_at_object_for_left))} limit \t{round(left_wheel_speed, 2), round(right_wheel_speed, 2)} out of {self.args.max_wheel_speed_for_left} wheel speed\t{round(arm_speed, 2)} out of {self.args.min_arm_speed_for_left} arm speed\t{self.durations['right'][object_index]} steps \n\n")
            
            # If pushing forward and/or pushing left or right, choose one.
            active_changes = []
            if pushing:
                active_changes.append(("pushing", global_movement_forward))
            if lefting:
                active_changes.append(("lefting", global_movement_left))
            if righting:
                active_changes.append(("righting", -global_movement_left)) 
            if len(active_changes) > 1:
                active_changes.sort(key=lambda x: x[1], reverse=True)
                highest_change = active_changes[0][0]
                pushing, lefting, righting = False, False, False
                if highest_change == "pushing":
                    pushing = True
                elif highest_change == "lefting":
                    lefting = True
                elif highest_change == "righting":
                    righting = True 
            
            # Clear shared tasks.
            if(being_near):
                watching = False
            if(pushing or lefting or righting):
                watching = False
                being_near = False
                topping = False
                            
                        
            if(verbose):
                print(f"After consideration:")
                print(f"Watching: {watching}")
                print(f"Being Near: {being_near}")
                print(f"Topping: {topping}") 
                print(f"Pushing: {pushing}")
                print(f"Lefting: {lefting}")
                print(f"Righting: {righting}\n")
                

            
            def update_duration(action_name, action_now, object_index, duration_threshold):
                if action_now:
                    self.durations[action_name][object_index] += 1
                else:
                    self.durations[action_name][object_index] = 0
                return self.durations[action_name][object_index] >= duration_threshold

            watched     = update_duration("watch",      watching,   object_index, self.args.watch_duration)
            been_near   = update_duration("be_near",    being_near, object_index, self.args.be_near_duration)
            topped      = update_duration("top",        topping,    object_index, self.args.top_duration)
            pushed      = update_duration("push",       pushing,    object_index, self.args.push_duration)
            lefted      = update_duration("left",       lefting,    object_index, self.args.left_duration)
            righted     = update_duration("right",      righting,   object_index, self.args.right_duration)
                        
            if(verbose):
                print(f"Finished:")
                print(f"\nINGs: {sum([watching, being_near, topping, pushing, lefting, righting])}\n")
                print(f"Watching: \t{watching} \tWatched: \t{watched} \t {self.durations['watch'][object_index]} steps")
                print(f"Being near: \t{being_near} \tBeen Near: \t{been_near} \t {self.durations['be_near'][object_index]} steps")
                print(f"Topping: \t{topping} \tTopped: \t{topped} \t {self.durations['top'][object_index]} steps")
                print(f"Pushing: \t{pushing} \tPushed: \t{pushed} \t {self.durations['push'][object_index]} steps")
                print(f"Lefting: \t{lefting} \tLefted: \t{lefted} \t {self.durations['left'][object_index]} steps")
                print(f"Righting: \t{righting} \tRighted: \t{righted} \t {self.durations['right'][object_index]} steps\n")
                
            key = (color_map[color_index], shape_map[shape_index])
            new_value = [watched, been_near, topped, pushed, lefted, righted, watching, being_near, topping, pushing, lefting, righting]
            
            # If there are multiple of the same object, consider them all.
            if key in objects_goals:
                objects_goals[key] = [old or new for old, new in zip(objects_goals[key], new_value)]
            else:
                objects_goals[key] = new_value
                                                    
        report_voice = empty_goal
        wrong_object = False
        task_performed = None
                
        for (color, shape), (watched, been_near, topped, pushed, lefted, righted, watching, being_near, topping, pushing, lefting, righting) in objects_goals.items():
            # If any one task is accomplished, find the task/color/shape.
                        
            if(sum([watched, been_near, topped, pushed, lefted, righted]) == 1):
                # If the correct object, check the task.
                if(watched):
                    task_performed = "watched"  
                else:
                    task_performed = "other"
                if(color == self.goal.color and shape == self.goal.shape):
                    if(
                        (self.goal.task.name == "WATCH"         and watched     and not (               being_near or   topping or  pushing or  lefting or  righting)) or 
                        (self.goal.task.name == "BE NEAR"       and been_near   and not (watching or                    topping or  pushing or  lefting or  righting)) or 
                        (self.goal.task.name == "TOUCH THE TOP" and topped      and not (watching or    being_near or               pushing or  lefting or  righting)) or 
                        (self.goal.task.name == "PUSH FORWARD"  and pushed      and not (watching or    being_near or   topping or              lefting or  righting)) or
                        (self.goal.task.name == "PUSH LEFT"     and lefted      and not (watching or    being_near or   topping or  pushing or              righting)) or
                        (self.goal.task.name == "PUSH RIGHT"    and righted     and not (watching or    being_near or   topping or  pushing or  lefting))):   
                        win = True 
                        reward = self.args.reward
                # If a task is occuring with a wrong object, no reward.
                else:
                    wrong_object = True
                    
            # Report's voice reflects ongoing processes
            task_in_progress = None
            if(sum([watching, being_near, topping, pushing, lefting, righting]) >= 1):
                if(watching):   task_in_progress = task_map[1]
                if(being_near): task_in_progress = task_map[2]
                if(topping):    task_in_progress = task_map[3]
                if(pushing):    task_in_progress = task_map[4]
                if(lefting):    task_in_progress = task_map[5] # If pushing but also lefting/righting,
                if(righting):   task_in_progress = task_map[6] # use lefting/righting
                report_voice = Goal(task_in_progress, color, shape, parenting = False)
                
        if(wrong_object):
            win = False 
            if(task_performed == "watched"):
                reward = 0
            else:
                reward = self.args.wrong_object_punishment
            
        if(verbose):
            print(f"\nReport voice: \'{report_voice.human_text}\'")
            print("Total reward:", reward)
            print("Win:", win)
                        
        return(reward, win, report_voice)
    
    
    
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
        proj_matrix = proj_matrix = p.computeProjectionMatrix(
            left, 
            right, 
            bottom, top, near, far)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=self.args.image_size * 2, 
            height=self.args.image_size * 2,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.physicsClient)
        
        if(type(rgba) == np.ndarray):
            pass
        else:
            rgba = np.array(rgba).reshape(32, 32, 4)
            depth = np.array(depth).reshape(32, 32)
            
        rgb = np.divide(rgba[:,:,:-1], 255)
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
        if(d.max() == d.min()): pass
        else: d = (d - d.min())/(d.max()-d.min())
        vision = np.concatenate([rgb, d], axis = -1)
        vision = resize(vision, (self.args.image_size, self.args.image_size, 4))
        return(vision)
        
    
    
if __name__ == "__main__":
    from utils import args
    physicsClient = get_physics(GUI = True, args = args)
    arena = Arena(physicsClient, args = args)
    
    while True:
        sleep(0.05)
        p.stepSimulation(physicsClientId=arena.physicsClient)