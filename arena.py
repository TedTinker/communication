#%%
import os
from random import choices, uniform, shuffle
import numpy as np
import pybullet as p
from math import pi, sin, cos
from time import sleep

from utils import default_args, print, shape_map, color_map, action_map, relative_to

def get_physics(GUI, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, -9.8, physicsClientId = physicsClient)
    return(physicsClient)
    
def get_joint_index(body_id, joint_name):
    num_joints = p.getNumJoints(body_id)
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i)
        if info[1].decode() == joint_name:
            return i
    return -1  # Return -1 if no joint with the given name is found



def generate_angles(n):
    random_angle = uniform(0, 2*pi)
    step_size = 2*pi / n
    angles = []
    for i in range(n):
        angle = (random_angle + (i * step_size)) % (2*pi)
        angles.append(angle)
    return angles



class Arena():
    def __init__(self, GUI = False, args = default_args):
        self.args = args
        self.physicsClient = get_physics(GUI)
        self.objects_in_play = {}
        self.watching = {}

        # Make floor and lower level.
        plane_positions = [
            [0, 0], 
            [10, 0], [0, 10], [10, 10],
            [-10, 0], [-10, 10], [-10, -10], 
            [0, -10], [10, -10], [-10, -10]]
        plane_ids = []
        for position in plane_positions:
            plane_id = p.loadURDF("plane.urdf", position + [0], globalScaling=.5, useFixedBase=True, physicsClientId=self.physicsClient)
            p.changeVisualShape(plane_id, -1, rgbaColor=(0,0,0,1), physicsClientId = self.physicsClient)
            plane_ids.append(plane_id)
            plane_id = p.loadURDF("plane.urdf", position + [-10], globalScaling=.5, useFixedBase=True, physicsClientId=self.physicsClient)
            plane_ids.append(plane_id)

        # Place robot. 
        pos = (0, 0, 1)
        self.default_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.body_num = p.loadURDF("robot.urdf", pos, self.default_orn,
                           globalScaling = self.args.body_size, 
                           physicsClientId = self.physicsClient)
        p.changeDynamics(self.body_num, 0, maxJointVelocity=10000)
        self.setBaseVelocity(0, 0)
        p.changeVisualShape(self.body_num, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = self.physicsClient)
        for i in range(p.getNumJoints(self.body_num)):
            p.changeVisualShape(self.body_num, i, rgbaColor=(1, 0, 0.5, 1), physicsClientId = self.physicsClient)
        
        self.arm_num = None
        self.hand_num = None
        num_joints = p.getNumJoints(self.body_num, physicsClientId=self.physicsClient)
        for joint_index in range(num_joints): 
            joint_info = p.getJointInfo(self.body_num, joint_index, physicsClientId=self.physicsClient) 
            link_name = joint_info[12].decode('utf-8') 
            if link_name == "arm_link": 
                self.hand_num = joint_index 
            if link_name == "left_hand_link": 
                self.hand_num = joint_index 
        
        # Place objects on lower level for future use.
        self.loaded = {key : [] for key in shape_map.keys()}
        for i, (shape, shape_file) in enumerate(shape_map.items()):
            for j in range(self.args.objects):
                pos = (3*i, 3*j, -10)
                object_num = p.loadURDF("pybullet_data/shapes/{}".format(shape_file), pos, self.default_orn, globalScaling = self.args.body_size, physicsClientId=self.physicsClient)
                self.loaded[shape].append((object_num, (pos[0], pos[1], pos[2] + 1)))
                p.changeDynamics(object_num, 0, maxJointVelocity=10000)
                                
    def begin(self, objects, goal):
        self.setBaseVelocity(0, 0)
        self.setBasePositionAndOrientation((0, 0, 1), 0)
        self.setArmsAndHands()
        self.goal = goal
        self.objects_in_play = {}
        self.watching = {}
        already_in_play = {key : 0 for key in shape_map.keys()}
        random_positions = generate_angles(len(objects))
        for i, (shape_index, color_index) in enumerate(objects):
            shape = list(shape_map)[shape_index]
            color = list(color_map.values())[color_index]
            pos = random_positions[i]
            object_num, old_pos = self.loaded[shape][already_in_play[shape]]
            already_in_play[shape] += 1
            p.resetBasePositionAndOrientation(object_num, (self.args.object_distance * sin(pos), self.args.object_distance * cos(pos), 1), self.default_orn, physicsClientId = self.physicsClient)
            p.changeVisualShape(object_num, -1, rgbaColor = (0,0,0,0), physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object_num)):
                p.changeVisualShape(object_num, i, rgbaColor=color, physicsClientId = self.physicsClient)
            self.objects_in_play[(shape, color, old_pos)] = object_num
            self.watching[object_num] = 0
        
    def step(self):
        p.stepSimulation(physicsClientId = self.physicsClient)
        for (shape, color, old_pos), object in self.objects_in_play.items():
            pos, _ = p.getBasePositionAndOrientation(object, physicsClientId = self.physicsClient)
            v, o = p.getBaseVelocity(object, physicsClientId=self.physicsClient)
            p.resetBasePositionAndOrientation(object, pos, self.default_orn, physicsClientId = self.physicsClient)
            p.resetBaseVelocity(object, v, o, physicsClientId = self.physicsClient)    
        _, ors = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        self.setBasePositionAndOrientation((0, 0, 1), yaw)
            
    def end(self):
        for (shape, color, old_pos), object in self.objects_in_play.items():
            p.resetBasePositionAndOrientation(object, old_pos, self.default_orn, physicsClientId = self.physicsClient)
        self.setArmsAndHands()
        self.setBaseVelocity(0, 0)
        self.setBasePositionAndOrientation((0, 0, 1), 0)
            
    def stop(self):
        p.disconnect(self.physicsClient)
        
    def get_pos_yaw_spe(self):
        pos, ors = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
        (vx, vy, _), _ = p.getBaseVelocity(self.body_num, physicsClientId=self.physicsClient)
        velocity_vec = np.array([vx, vy])
        spe = float(np.dot(velocity_vec, forward_dir))
        return(pos, yaw, spe)
    
    def get_arm_angles(self):
        angles = []
        joint_names = [
            'body_shoulder_joint',
            'shoulder_arm_joint']
        for joint_name in joint_names:
            joint_index = get_joint_index(self.body_num, joint_name)
            joint_state = p.getJointState(self.body_num, joint_index, physicsClientId=self.physicsClient)
            angles.append(joint_state[0])  # joint_state[0] is the position (angle) of the joint
        return angles
    
    def setBaseVelocity(self, x, y):    
        p.resetBaseVelocity(self.body_num, (x,y,0), (0,0,0), physicsClientId = self.physicsClient)    
    
    def setBasePositionAndOrientation(self, pos, yaw):
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        pos = (pos[0], pos[1], 1)
        (x, y, _), _ = p.getBaseVelocity(self.body_num, physicsClientId = self.physicsClient)
        p.resetBasePositionAndOrientation(self.body_num, pos, orn, physicsClientId = self.physicsClient)
        self.setBaseVelocity(x, y)
        
    def setArmsAndHands(self, shoulder = 0, arm = 0):
        if(shoulder >= 0): shoulder = self.args.min_shoulder
        else:              shoulder = self.args.max_shoulder
        arm = relative_to(arm, self.args.min_arm, self.args.max_arm)
        for limb_name, target in [
            ('body_shoulder_joint', shoulder), 
            ('shoulder_arm_joint', arm)]:
            limb_index = get_joint_index(self.body_num, limb_name)
            p.resetJointState(self.body_num, limb_index, target, physicsClientId=self.physicsClient)
        
    def rewards(self):
        reward = False
        goal_action = self.goal[0]
        goal_shape = list(shape_map)[self.goal[1][0]]
        goal_color = list(color_map.values())[self.goal[1][1]]
        for (shape, color, old_pos), object_num in self.objects_in_play.items():
            if(shape != goal_shape or color != goal_color): pass 
            else:
                if(goal_action.upper() == "WATCH"):
                    object_pos, _ = p.getBasePositionAndOrientation(object_num, physicsClientId=self.physicsClient)
                    agent_pos, agent_ori = p.getBasePositionAndOrientation(self.body_num)
                    distance_vector = np.subtract(object_pos, agent_pos)
                    distance = np.linalg.norm(distance_vector)
                    normalized_distance_vector = distance_vector / distance
                    rotation_matrix = p.getMatrixFromQuaternion(agent_ori)
                    forward_vector = np.array([rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]])
                    forward_vector /= np.linalg.norm(forward_vector)
                    dot_product = np.dot(forward_vector, normalized_distance_vector)
                    angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))  
                    cross_product = np.cross(forward_vector, normalized_distance_vector)
                    if cross_product[2] < 0:  # Assuming Z-axis is up
                        angle_radians = -angle_radians
                    watching = abs(angle_radians) < pi/16
                    if watching: self.watching[object_num] += 1
                    else:        self.watching[object_num] = 0 
                    if(self.watching[object_num] >= self.args.watch_duration):
                        reward = True
                else:
                    body_pos, _ = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
                    object_pos, _ = p.getBasePositionAndOrientation(object_num, physicsClientId = self.physicsClient)
                    velocity, _ = p.getBaseVelocity(object_num, physicsClientId = self.physicsClient)
                    body_pos = np.array(body_pos)
                    object_pos = np.array(object_pos)
                    velocity = np.array(velocity)
                    direction_vector = body_pos - object_pos
                    direction_vector /= np.linalg.norm(direction_vector)
                    speed_toward_body = np.dot(velocity, direction_vector)
                    up_vector = np.array([0, 0, 1]) 
                    right_vector = np.cross(direction_vector, up_vector) 
                    right_vector /= np.linalg.norm(right_vector) 
                    speed_right = -np.dot(velocity, right_vector)
                    # Speed right isn't good; it gives a win if the agent is turning relative to the object!
                    if(goal_action.upper() == "PUSH"):
                        if(speed_toward_body <= -self.args.push_speed):
                            reward = True
                    if(goal_action.upper() == "PULL"):
                        if(speed_toward_body >= self.args.push_speed):
                            reward = True
                    if(goal_action.upper() == "LEFT"):
                        if(speed_right <= -self.args.push_speed):
                            reward = True 
                    if(goal_action.upper() == "RIGHT"):
                        if(speed_right >= self.args.push_speed):
                            reward = True
                
        win = reward
        reward = self.args.reward if reward else 0
        return(reward, win)
    
    def photo_from_above(self):
        pos, _, _ = self.get_pos_yaw_spe()
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [pos[0], pos[1], 8], 
            cameraTargetPosition = [pos[0], pos[1], -1],    # Camera / target position very important
            cameraUpVector = [-1, 0, 0], physicsClientId = self.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 10, physicsClientId = self.physicsClient)
        _, _, rgba, _, _ = p.getCameraImage(
            width=256, height=256,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.physicsClient)
        return(rgba)
    
    def photo_for_agent(self):
        pos, yaw, spe = self.get_pos_yaw_spe()
        x, y = cos(yaw), sin(yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [pos[0] + x, pos[1] + y, 1], 
            cameraTargetPosition = [pos[0] + x*2, pos[1] + y*2, 1],    # Camera / target position very important
            cameraUpVector = [0, 0, 1], physicsClientId = self.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 10, physicsClientId = self.physicsClient)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=self.args.image_size, height=self.args.image_size,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.physicsClient)
        rgb = np.divide(rgba[:,:,:-1], 255)
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
        if(d.max() == d.min()): pass
        else: d = (d.max() - d)/(d.max()-d.min())
        rgbd = np.concatenate([rgb, d], axis = -1)
        return(rgbd)
        
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    args = default_args
    from utils import make_object
    arena = Arena(GUI = True)
    objects = [make_object() for i in range(args.objects)]
    goal = [choices(action_map)[0], objects[0]]
    arena.begin(objects = objects, goal = goal)
    i = -1
    going_up = True
    i_size = .1
    arena.setArmsAndHands(args.max_shoulder, args.max_arm)
    while(True):
        arena.setArmsAndHands(-1, i)
        i += i_size 
        if((going_up and i >= 1) or (not going_up and i <= -1)):
            i_size *= -1
            going_up = not going_up
        arena.step()
        arena.rewards()
        rgba = arena.photo_from_above()
        sleep(.1)
# %%
