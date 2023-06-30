#%%
from random import choices, uniform, shuffle
import pandas as pd
import numpy as np
import pybullet as p
import cv2, os
from itertools import product
from math import pi, sin, cos
from time import sleep

from utils import default_args, args, print, shapes, colors, goals

def get_physics(GUI, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, -9.8, physicsClientId = physicsClient)
    return(physicsClient)

def enable_opengl():
    import pkgutil
    egl = pkgutil.get_loader('eglRenderer')
    import pybullet_data

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    # print("plugin=", plugin)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
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
    def __init__(self, arms = True, GUI = False, args = default_args):
        #enable_opengl()
        self.arms = arms
        self.args = args
        self.physicsClient = get_physics(GUI)
        self.objects = {}
        self.watching = {}

        plane_positions = [
            [0, 0, 0], 
            [10, 0, 0], [0, 10, 0], [10, 10, 0],
            [-10, 0, 0], [-10, 10, 0], [-10, -10, 0], 
            [0, -10, 0], [10, -10, 0], [-10, -10, 0]]
        plane_ids = []
        for position in plane_positions:
            plane_id = p.loadURDF("plane.urdf", position, globalScaling=.5, useFixedBase=True, physicsClientId=self.physicsClient)
            p.changeVisualShape(plane_id, -1, rgbaColor=(0,0,0,1), physicsClientId = self.physicsClient)
            plane_ids.append(plane_id)

        robot_file = "robot_arms" if arms else "robot"
        inherent_roll = 0
        inherent_pitch = 0
        yaw = 0
        spe = self.args.min_speed
        pos = (0, 0, 1)
        orn = p.getQuaternionFromEuler([inherent_roll, inherent_pitch, yaw])
        self.body_num = p.loadURDF("robots/{}.urdf".format(robot_file), pos, orn,
                           globalScaling = self.args.body_size, 
                           physicsClientId = self.physicsClient)
        p.changeDynamics(self.body_num, 0, maxJointVelocity=10000)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        p.changeVisualShape(self.body_num, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = self.physicsClient)
        for i in range(p.getNumJoints(self.body_num)):
            p.changeVisualShape(self.body_num, i, rgbaColor=(1, 0, 0.5, 1), physicsClientId = self.physicsClient)
            
    def begin(self, objects):
        yaw = 0
        spe = self.args.min_speed
        pos = (0, 0, 1)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        self.resetBasePositionAndOrientation(pos, yaw)
        if(self.arms): self.resetArmsAndHands(1, 1)
        
        self.objects = {} ; self.watching = {}
        random_positions = generate_angles(len(objects))
        orn = p.getQuaternionFromEuler([0,0,0])
        shuffle(objects)
        for i, (shape, color, goal) in enumerate(objects):
            pos = random_positions[i]
            object = p.loadURDF("shapes/{}".format(shape), (5*sin(pos), 5*cos(pos), 0), orn, globalScaling = self.args.body_size,
                                physicsClientId=self.physicsClient)
            p.changeVisualShape(object, -1, rgbaColor = (0,0,0,0), physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object)):
                p.changeVisualShape(object, i, rgbaColor=color, physicsClientId = self.physicsClient)
            p.changeDynamics(object, 0, maxJointVelocity=10000)
            self.objects[(shape, color, goal)] = object
            self.watching[object] = 0
            
    def end(self):
        for (shape, color, goal), object in self.objects.items():
            p.removeBody(object, physicsClientId = self.physicsClient)
        self.objects = {} ; self.watching = {}
        
    def get_pos_yaw_spe(self):
        pos, ors = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        (x, y, _), _ = p.getBaseVelocity(self.body_num, physicsClientId = self.physicsClient)
        spe = (x**2 + y**2)**.5
        return(pos, yaw, spe)
    
    def resetBasePositionAndOrientation(self, pos, yaw):
        inherent_roll = 0
        inherent_pitch = 0
        orn = p.getQuaternionFromEuler([inherent_roll, inherent_pitch, yaw])
        p.resetBasePositionAndOrientation(self.body_num, pos, orn, physicsClientId = self.physicsClient)
        
    def resetArmsAndHands(self, arms, hands):
        right_arm = get_joint_index(self.body_num, 'body_right_arm_joint')
        left_arm = get_joint_index(self.body_num, 'body_left_arm_joint')
        right_hand = get_joint_index(self.body_num, 'right_arm_right_hand_joint')
        left_hand = get_joint_index(self.body_num, 'left_arm_left_hand_joint')
        p.setJointMotorControl2(self.body_num, right_arm,  p.POSITION_CONTROL, targetPosition = arms,   physicsClientId = self.physicsClient)
        p.setJointMotorControl2(self.body_num, right_hand, p.POSITION_CONTROL, targetPosition = hands,  physicsClientId = self.physicsClient)
        p.setJointMotorControl2(self.body_num, left_arm,   p.POSITION_CONTROL, targetPosition = -arms,  physicsClientId = self.physicsClient)
        p.setJointMotorControl2(self.body_num, left_hand,  p.POSITION_CONTROL, targetPosition = -hands, physicsClientId = self.physicsClient)
        
    def resetBaseVelocity(self, x, y):    
        p.resetBaseVelocity(self.body_num, (x,y,0), (0,0,0), physicsClientId = self.physicsClient)        
        
    def rewards(self):
        reward = 0
        to_delete = []
        for (shape, color, goal), object in self.objects.items():
                        
            if(goal == "watch"):
                object_pos, _ = p.getBasePositionAndOrientation(object, physicsClientId = self.physicsClient)
                body_num_pos, body_num_ori = p.getBasePositionAndOrientation(self.body_num)
                vector_to_object = np.subtract(object_pos, body_num_pos)
                matrix = p.getMatrixFromQuaternion(body_num_ori)
                forward_vector = [matrix[0], matrix[3], matrix[6]]
                vector_to_object /= np.linalg.norm(vector_to_object)
                distance = np.linalg.norm(vector_to_object)
                forward_vector /= np.linalg.norm(forward_vector)
                dot_product = np.dot(forward_vector, vector_to_object)
                watching = dot_product >= .9 and distance > 2
                if watching: self.watching[object] += 1
                else:        self.watching[object] = 0 
                if(self.watching[object] >= 2):
                    reward += 1
                    to_delete.append((shape, color, goal, object))
            
            if(goal == "touch"):
                col = p.getContactPoints(object, self.body_num, physicsClientId = self.physicsClient)
                if(len(col)) > 0: 
                    link_index = col[0][3]
                    num_joints = p.getNumJoints(self.body_num)
                    link_name = None

                    for joint_index in range(num_joints):
                        joint_info = p.getJointInfo(self.body_num, joint_index)
                        if joint_info[0] == link_index:
                            link_name = joint_info[12].decode("utf-8")
                            break
                    
                    if(link_name.startswith("body") or link_name.startswith("nose")):
                        reward += 1
                        to_delete.append((shape, color, goal, object))
                        
            if(goal == "lift"):
                pos, _ = p.getBasePositionAndOrientation(object, physicsClientId = self.physicsClient)
                if(pos[-1] > 2):
                    reward += 1
                    to_delete.append((shape, color, goal, object))
                    
            if(goal == "push"):
                body_pos, _ = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
                object_pos, _ = p.getBasePositionAndOrientation(object, physicsClientId = self.physicsClient)
                velocity, _ = p.getBaseVelocity(object, physicsClientId = self.physicsClient)
                body_pos = np.array(body_pos)
                object_pos = np.array(object_pos)
                velocity = np.array(velocity)
                direction_vector = body_pos - object_pos
                direction_vector /= np.linalg.norm(direction_vector)
                speed_toward_body = np.dot(velocity, direction_vector)
                if(speed_toward_body < -1):
                    reward += 1
                    to_delete.append((shape, color, goal, object))
                    
            if(goal == "pull"):
                body_pos, _ = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
                object_pos, _ = p.getBasePositionAndOrientation(object, physicsClientId = self.physicsClient)
                velocity, _ = p.getBaseVelocity(object, physicsClientId = self.physicsClient)
                body_pos = np.array(body_pos)
                object_pos = np.array(object_pos)
                velocity = np.array(velocity)
                direction_vector = body_pos - object_pos
                direction_vector /= np.linalg.norm(direction_vector)
                speed_toward_body = np.dot(velocity, direction_vector)
                if(speed_toward_body > 1):
                    reward += 1
                    to_delete.append((shape, color, goal, object))
                    
            if(goal == "topple"):
                _, object_quaternion = p.getBasePositionAndOrientation(object, physicsClientId=self.physicsClient)
                orn = p.getEulerFromQuaternion(object_quaternion)
                rotation_z = np.rad2deg(orn[2])
                if(np.abs(rotation_z) > 45):
                    reward += 1
                    to_delete.append((shape, color, goal, object))
                

        for (shape, color, goal, object) in to_delete:
            p.removeBody(object, physicsClientId = self.physicsClient)
            del self.objects[(shape, color, goal)]
            del self.watching[object]

        return(reward)
            
    def stop(self):
        p.disconnect(self.physicsClient)
        
        

if __name__ == "__main__":
    objects = [(shape, color) for shape, color in zip(shapes, colors)]
    gs = [choices(goals)[0] for _ in objects]
    objects = [(shape, color, goal) for (shape, color), goal in zip(objects, gs)]
    arena = Arena(arms = False, GUI = True)
    arena.begin(objects = objects)
    while(True):
        p.stepSimulation(physicsClientId = arena.physicsClient)
        sleep(.1)
# %%
