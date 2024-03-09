#%%
import os
from random import choices, uniform, shuffle
import numpy as np
import pybullet as p
from math import pi, sin, cos, tan, radians, atan2, sqrt, isnan
from time import sleep
from skimage.transform import resize

from utils import default_args, shape_map, color_map, action_map, relative_to#, print

def get_physics(w = 10, h = 10):
    physicsClient = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, -10, physicsClientId = physicsClient)
    p.setTimeStep(.001)
    p.setPhysicsEngineParameter(numSolverIterations=0, physicsClientId = physicsClient)
    return(physicsClient)
    
def get_joint_index(body_id, joint_name):
    num_joints = p.getNumJoints(body_id)
    
    
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i)
        if info[1].decode() == joint_name:
            return i
    return -1  # Return -1 if no joint with the given name is found



# FOV of agent vision.
fov_x_deg = 90
fov_y_deg = 90
near = 1.1
far = 4
fov_x_rad = radians(fov_x_deg)
fov_y_rad = radians(fov_y_deg)
right = near * tan(fov_x_rad / 2)
left = -right
top = near * tan(fov_y_rad / 2)
bottom = -top



class Arena():
    def __init__(self, args = default_args):
        self.args = args
        self.physicsClient = get_physics()
        
        # Make floor and lower level.
        plane_positions = [
            [0, 0]]
        plane_ids = []
        for position in plane_positions:
            plane_id = p.loadURDF("plane.urdf", position + [-.2], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
            p.setCollisionFilterGroupMask(plane_id, -1, collisionFilterGroup = 1, collisionFilterMask = 1)
            plane_ids.append(plane_id)
            plane_id = p.loadURDF("plane.urdf", position + [-10.2], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
            p.setCollisionFilterGroupMask(plane_id, -1, collisionFilterGroup = 1, collisionFilterMask = 1)
            plane_ids.append(plane_id)

        # Place robot. 
        pos = (0, 0, 1)
        self.default_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_index = p.loadURDF("robot.urdf", pos, self.default_orn, useFixedBase=False, globalScaling = self.args.body_size, physicsClientId = self.physicsClient)
        p.setCollisionFilterGroupMask(self.robot_index, -1, collisionFilterGroup = 1, collisionFilterMask = 1)
        
        p.changeVisualShape(self.robot_index, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = self.physicsClient)
        self.sensors = []
        for link_index in range(p.getNumJoints(self.robot_index)):
            joint_info = p.getJointInfo(self.robot_index, link_index)
            link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
            if(link_name == "left_wheel"):
                self.left_wheel = link_index 
            if(link_name == "right_wheel"):
                self.right_wheel = link_index
            if("sensor" in link_name):
                self.sensors.append((link_index, link_name))
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (1, 0, 0, .5), physicsClientId = self.physicsClient)
            else:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (0, 0, 0, 1), physicsClientId = self.physicsClient)
        
        # Place object.
        shape, shape_file = list(shape_map.items())[0]
        pos = (10, 0, 1)
        self.object_index = p.loadURDF("pybullet_data/shapes/{}".format(shape_file), pos, p.getQuaternionFromEuler([0, 0, pi/2]), useFixedBase=False, globalScaling = self.args.object_size, physicsClientId=self.physicsClient)
        p.setCollisionFilterGroupMask(self.object_index, -1, collisionFilterGroup = 1, collisionFilterMask = 1)
        
    def object_faces_up(self, object_index):
        pos, orn = p.getBasePositionAndOrientation(object_index, physicsClientId = self.physicsClient)
        x = pos[0]
        y = pos[1]
        (a, b, c) = p.getEulerFromQuaternion(orn)
        orn = p.getQuaternionFromEuler([0, 0, c if not isnan(c) else 0])
        p.resetBasePositionAndOrientation(object_index, (x, y, 2), orn, physicsClientId = self.physicsClient)
                                
    def begin(self, object_pos):
        self.set_pos()
        self.set_yaw()
        self.set_speeds()
        self.set_shoulder_pos()
        
        p.resetBasePositionAndOrientation(self.object_index, (object_pos, 0, 2), (0, 0, 0, 0), physicsClientId = self.physicsClient)
        #self.object_faces_up(self.object_index)
        p.changeVisualShape(self.object_index, -1, rgbaColor = [0, 1, 0, 1], physicsClientId = self.physicsClient)
        for i in range(p.getNumJoints(self.object_index)):
                p.changeVisualShape(self.object_index, i, rgbaColor=[0, 1, 0, 1], physicsClientId = self.physicsClient)
        
    def step(self, left_wheel, right_wheel, shoulder, verbose = False):
        self.set_speeds(left_wheel, right_wheel, shoulder)
        for step in range(self.args.steps_per_step):
            if(verbose): 
                print(step)
                if(step % 10 == 0):
                    WAITING = input("WAITING")
            p.stepSimulation(physicsClientId = self.physicsClient)
            self.set_speeds(left_wheel, right_wheel, shoulder)
            print("Sensors:", self.touching_object())
            
    def end(self):
        for (shape, color, old_pos), object in self.objects_in_play.items():
            p.resetBasePositionAndOrientation(object, old_pos, self.default_orn, physicsClientId = self.physicsClient)
            
    def stop(self):
        p.disconnect(self.physicsClient)
        
    def get_pos_yaw_spe(self):
        pos, ors = p.getBasePositionAndOrientation(self.robot_index, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
        (vx, vy, _), _ = p.getBaseVelocity(self.robot_index, physicsClientId=self.physicsClient)
        velocity_vec = np.array([vx, vy])
        spe = float(np.dot(velocity_vec, forward_dir))
        return(pos, yaw, spe)
    
    def get_joint_angles(self):
        angles = []
        joint_names = [
            'body_shoulder_joint']
        for joint_name in joint_names:
            joint_index = get_joint_index(self.robot_index, joint_name)
            joint_state = p.getJointState(self.robot_index, joint_index, physicsClientId=self.physicsClient)
            angles.append(joint_state[0])  
        return angles
    
    def set_pos(self, pos = (0, 0)):
        pos = (pos[0], pos[1], 1)
        _, yaw, _ = self.get_pos_yaw_spe()
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
        
    def set_yaw(self, yaw = 0):
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        pos, _, _ = self.get_pos_yaw_spe()
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
    
    def set_speeds(self, left_wheel = 0, right_wheel = 0, shoulder = 0):
        left_wheel = relative_to(left_wheel, self.args.min_speed, self.args.max_speed)
        right_wheel = relative_to(right_wheel, self.args.min_speed, self.args.max_speed)
        linear_velocity = (left_wheel + right_wheel) / 2
        angular_velocity = (right_wheel - left_wheel) / self.args.angular_scaler
        _, yaw, _ = self.get_pos_yaw_spe()
        x = linear_velocity * cos(yaw)
        y = linear_velocity * sin(yaw)
        p.resetBaseVelocity(self.robot_index, linearVelocity=[x, y, 0], angularVelocity=[0, 0, angular_velocity])
        
        shoulder = relative_to(shoulder, self.args.min_shoulder_angle, self.args.max_shoulder_angle)
        self.set_shoulder_pos(shoulder)
            
    def set_shoulder_pos(self, shoulder = -pi/2):
        for limb_name, target in [
            ('body_shoulder_joint', shoulder)]:
            limb_index = get_joint_index(self.robot_index, limb_name)
            p.resetJointState(self.robot_index, limb_index, target, physicsClientId=self.physicsClient)
            
    def find_link_index(self, object_id, link_name):
        num_joints = p.getNumJoints(object_id)
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(object_id, joint_index)
            if joint_info[12].decode('utf-8') == link_name:
                return joint_index
            
    def touching_object(self):
        touching = []
        for sensor_index, _ in self.sensors:
            touching.append(bool(p.getContactPoints(bodyA=self.robot_index, bodyB=self.object_index, linkIndexA=sensor_index)))
        return(touching)
    
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
        pos, yaw, _ = self.get_pos_yaw_spe()
        
        def get_photo(pos, yaw):
            x, y = cos(yaw), sin(yaw)
            view_matrix = p.computeViewMatrix(
                cameraEyePosition = [pos[0], pos[1], 1.1], 
                cameraTargetPosition = [pos[0] + x*2, pos[1] + y*2, 1.1],    # Camera / target position very important
                cameraUpVector = [0, 0, 1], physicsClientId = self.physicsClient)
            proj_matrix = p.computeProjectionMatrix(left, right, bottom, top, near, far)
            _, _, rgba, depth, _ = p.getCameraImage(
                width=self.args.image_size * 2, height=self.args.image_size * 2,
                projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
                physicsClientId = self.physicsClient)
            rgb = np.divide(rgba[:,:,:-1], 255)
            d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
            if(d.max() == d.min()): pass
            else: d = (d.max() - d)/(d.max()-d.min())
            rgbd = np.concatenate([rgb, d], axis = -1)
            rgbd = resize(rgbd, (self.args.image_size, self.args.image_size, 4))
            return(rgbd)
            
        rgbd = get_photo(pos, yaw)
        
        return(rgbd)
        
        

if __name__ == "__main__":
    args = default_args
    arena = Arena()
    
    
    
    
    
    print("\nSPIN")
    arena.begin(5)
    for step in range(50):
        arena.step(-1, 1, -1, verbose = True)
        rgba = arena.photo_from_above()
    
    """print("\nPUSHING WITH BODY")
    arena.begin(5)
    for step in range(3):
        arena.step(1, 1, -1, verbose = True)
        rgba = arena.photo_from_above()
    
    print("\nPUSHING WITH HAND")
    arena.begin(10)
    for step in range(3):
        arena.step(1, 1, 1, verbose = True)
        rgba = arena.photo_from_above()
        
    print("\nPULLING WITH HAND")
    arena.begin(3)
    for step in range(3):
        arena.step(-1, -1, 1, verbose = True)
        rgba = arena.photo_from_above()
        
    print("\nPUSHING LEFT")
    arena.begin(4)
    arena.step(1, -1, -1, verbose = True)
    for step in range(3):
        arena.step(-1, 1, 1, verbose = True)
        rgba = arena.photo_from_above()
        
    print("\nPUSHING RIGHT")
    arena.begin(4)
    arena.step(-1, 1, -1, verbose = True)
    for step in range(3):
        arena.step(1, -1, 1, verbose = True)
        rgba = arena.photo_from_above()"""
# %%
