#%%
from random import choices
import pandas as pd
import numpy as np
import pybullet as p
import cv2, os
from itertools import product
from math import pi, sin, cos

from utils import default_args, args, print

def get_physics(GUI, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data/")
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



class Arena():
    def __init__(self, GUI = False, args = default_args):
        #enable_opengl()
        self.args = args
        self.physicsClient = get_physics(GUI)

        #plane_positions = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]]
        #plane_ids = []
        #for position in plane_positions:
        #    plane_id = p.loadURDF("plane.urdf", position, globalScaling=.5, useFixedBase=True, physicsClientId=self.physicsClient)
        #    plane_ids.append(plane_id)

        inherent_roll = 0
        inherent_pitch = 0
        yaw = 0
        spe = self.args.min_speed
        file = "robot.urdf"
        pos = (0, 0, 1)
        orn = p.getQuaternionFromEuler([inherent_roll, inherent_pitch, yaw])
        self.body_num = p.loadURDF(file, pos, orn,
                           globalScaling = self.args.body_size, 
                           physicsClientId = self.physicsClient)
        p.changeDynamics(self.body_num, 0, maxJointVelocity=10000)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        p.changeVisualShape(self.body_num, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = self.physicsClient)
        for i in range(p.getNumJoints(self.body_num)):
            p.changeVisualShape(self.body_num, i, rgbaColor=(1, 0, 0.5, 1), physicsClientId = self.physicsClient)
            
    def begin(self):
        yaw = 0
        spe = self.args.min_speed
        pos = (0, 0, 1)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        self.resetBasePositionAndOrientation(pos, yaw)
        
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
        p.setJointMotorControl2(self.body_num, right_arm, p.POSITION_CONTROL, targetPosition=arms)
        p.setJointMotorControl2(self.body_num, right_hand, p.POSITION_CONTROL, targetPosition=hands)
        p.setJointMotorControl2(self.body_num, left_arm, p.POSITION_CONTROL, targetPosition=arms)
        p.setJointMotorControl2(self.body_num, left_hand, p.POSITION_CONTROL, targetPosition=-hands)
        
    def resetBaseVelocity(self, x, y):    
        p.resetBaseVelocity(self.body_num, (x,y,0), (0,0,0), physicsClientId = self.physicsClient)
            
    def stop(self):
        p.disconnect(self.physicsClient)
        
        

if __name__ == "__main__":
    arena = Arena(GUI = True)
    arena.begin()
    while(True):
        p.stepSimulation(physicsClientId = arena.physicsClient)
# %%
