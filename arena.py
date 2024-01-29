#%%
import os
from random import choices, uniform, shuffle
import numpy as np
import pybullet as p
from math import pi, sin, cos
from time import sleep

from utils import default_args, print, shape_map, color_map, action_map

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
        
        # Place objects on lower level for future use.
        self.loaded = {key : [] for key in shape_map.keys()}
        for i, (shape, shape_file) in enumerate(shape_map.items()):
            for j in range(self.args.objects):
                pos = (3*i, 3*j, -10)
                object_num = p.loadURDF("pybullet_data/shapes/{}".format(shape_file), pos, self.default_orn, globalScaling = self.args.body_size, physicsClientId=self.physicsClient)
                self.loaded[shape].append((object_num, pos))
                                
    def begin(self, objects, goal):
        self.setBaseVelocity(0, 0)
        self.setBasePositionAndOrientation((0, 0, 1), 0)
        self.setArmsAndHands(0, 0, 0, 0)
        p.stepSimulation(physicsClientId = self.physicsClient)
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
            p.resetBasePositionAndOrientation(object_num, (5*sin(pos), 5*cos(pos), 0), self.default_orn, physicsClientId = self.physicsClient)
            p.changeVisualShape(object_num, -1, rgbaColor = (0,0,0,0), physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object_num)):
                p.changeVisualShape(object_num, i, rgbaColor=color, physicsClientId = self.physicsClient)
            p.changeDynamics(object_num, 0, maxJointVelocity=10000)
            self.objects_in_play[(shape, color, old_pos)] = object_num
            self.watching[object_num] = 0
        
    def step(self):
        p.stepSimulation(physicsClientId = self.physicsClient)
        pos, ors = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        self.setBasePositionAndOrientation(pos, yaw)
            
    def end(self):
        for (shape, color, old_pos), object in self.objects_in_play.items():
            p.resetBasePositionAndOrientation(object, old_pos, self.default_orn, physicsClientId = self.physicsClient)
        self.setArmsAndHands(0, 0, 0, 0)
        self.setBaseVelocity(0, 0)
        self.setBasePositionAndOrientation((0, 0, 1), 0)
        p.stepSimulation(physicsClientId = self.physicsClient)
            
    def stop(self):
        p.disconnect(self.physicsClient)
        
    def get_pos_yaw_spe(self):
        pos, ors = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        (x, y, _), _ = p.getBaseVelocity(self.body_num, physicsClientId = self.physicsClient)
        spe = (x**2 + y**2)**.5
        return(pos, yaw, spe)
    
    def setBaseVelocity(self, x, y):    
        p.resetBaseVelocity(self.body_num, (x,y,0), (0,0,0), physicsClientId = self.physicsClient)    
    
    def setBasePositionAndOrientation(self, pos, yaw):
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        pos = (pos[0], pos[1], 1)
        (x, y, _), _ = p.getBaseVelocity(self.body_num, physicsClientId = self.physicsClient)
        p.resetBasePositionAndOrientation(self.body_num, pos, orn, physicsClientId = self.physicsClient)
        self.setBaseVelocity(x, y)
        
    def setArmsAndHands(self, right_arm, right_hand, left_arm, left_hand):
        for limb_name, target in [
            ('body_right_arm_joint', right_arm), 
            ('body_left_arm_joint', left_arm), 
            ('right_arm_right_hand_joint', right_hand), 
            ('left_arm_left_hand_joint', left_hand)]:
            limb_index = get_joint_index(self.body_num, limb_name)
            p.setJointMotorControl2(self.body_num, limb_index, p.POSITION_CONTROL, targetPosition = target, physicsClientId = self.physicsClient)
        
    def rewards(self):
        reward = False
        goal_action = self.goal[0]
        goal_shape = list(shape_map)[self.goal[1][0]]
        goal_color = list(color_map.values())[self.goal[1][1]]
        for (shape, color, old_pos), object_num in self.objects_in_play.items():
            if(shape != goal_shape or color != goal_color): pass 
            else:
                if(goal_action == "watch"):
                    object_pos, _ = p.getBasePositionAndOrientation(object_num, physicsClientId = self.physicsClient)
                    body_num_pos, body_num_ori = p.getBasePositionAndOrientation(self.body_num)
                    distance_to_object = np.subtract(object_pos, body_num_pos)
                    matrix = p.getMatrixFromQuaternion(body_num_ori)
                    forward_vector = [matrix[0], matrix[3], matrix[6]]
                    distance_to_object /= np.linalg.norm(distance_to_object)
                    distance = np.linalg.norm(distance_to_object)
                    forward_vector /= np.linalg.norm(forward_vector)
                    dot_product = np.dot(forward_vector, distance_to_object)
                    watching = dot_product >= .9 and distance > 2
                    if watching: self.watching[object_num] += 1
                    else:        self.watching[object_num] = 0 
                    if(self.watching[object_num] >= 2 and distance_to_object > 100):
                        reward = True
                
                if(goal_action == "touch"):
                    col = p.getContactPoints(object_num, self.body_num, physicsClientId = self.physicsClient)
                    if(len(col)) > 0: 
                        reward = True
                            
                if(goal_action == "lift"):
                    pos, _ = p.getBasePositionAndOrientation(object_num, physicsClientId = self.physicsClient)
                    if(pos[-1] > 2):
                        reward = True
                        
                if(goal_action == "pull"):
                    body_pos, _ = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
                    object_pos, _ = p.getBasePositionAndOrientation(object_num, physicsClientId = self.physicsClient)
                    velocity, _ = p.getBaseVelocity(object_num, physicsClientId = self.physicsClient)
                    body_pos = np.array(body_pos)
                    object_pos = np.array(object_pos)
                    velocity = np.array(velocity)
                    direction_vector = body_pos - object_pos
                    direction_vector /= np.linalg.norm(direction_vector)
                    speed_toward_body = np.dot(velocity, direction_vector)
                    if(speed_toward_body > 1):
                        reward = True
                        
                if(goal_action == "spin"):
                    pass
                    #reward = True
        win = reward
        reward = self.args.reward if reward else 0
        return(reward, win)
    
    def photo_from_above(self):
        pos, _, _ = self.get_pos_yaw_spe()
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [pos[0], pos[1], 5], 
            cameraTargetPosition = [pos[0], pos[1], 2],    # Camera / target position very important
            cameraUpVector = [-1, 0, 0], physicsClientId = self.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 10, physicsClientId = self.physicsClient)
        _, _, rgba, _, _ = p.getCameraImage(
            width=128, height=128,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.physicsClient)
        return(rgba)
        
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    args = default_args
    from utils import make_object
    arena = Arena(GUI = True)
    objects = [make_object() for i in range(args.objects)]
    goal = [choices(action_map)[0], objects[0]]
    arena.begin(objects = objects, goal = goal)
    while(True):
        arena.step()
        arena.rewards()
        rgba = arena.photo_from_above()
        sleep(.01)
# %%
