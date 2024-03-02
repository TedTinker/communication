#%%
import os
from random import choices, uniform, shuffle
import numpy as np
import pybullet as p
from math import pi, sin, cos, tan, radians, atan2
from time import sleep
from skimage.transform import resize

from utils import default_args, print, shape_map, color_map, action_map, relative_to

def get_physics(GUI, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, 0, physicsClientId = physicsClient)
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
    def __init__(self, GUI = False, args = default_args):
        self.args = args
        self.physicsClient = get_physics(GUI)
        self.objects_in_play = {}
        self.watching = {}

        # Place robot. 
        pos = (0, 0, 1)
        self.default_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.body_num = p.loadURDF("robot.urdf", pos, self.default_orn, useFixedBase=True,
                           globalScaling = self.args.body_size, 
                           physicsClientId = self.physicsClient)
        p.changeDynamics(self.body_num, 0, maxJointVelocity=10000)
        p.changeVisualShape(self.body_num, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = self.physicsClient)
        for i in range(p.getNumJoints(self.body_num)):
            p.changeVisualShape(self.body_num, i, rgbaColor=(0, 0, 0, 1), physicsClientId = self.physicsClient)
        
        self.arm_num = None
        num_joints = p.getNumJoints(self.body_num, physicsClientId=self.physicsClient)
        for joint_index in range(num_joints): 
            joint_info = p.getJointInfo(self.body_num, joint_index, physicsClientId=self.physicsClient) 
            link_name = joint_info[12].decode('utf-8') 
            if link_name == "arm_link": 
                self.arm_num = joint_index 
        
        # Place objects on lower level for future use.
        self.loaded = {key : [] for key in shape_map.keys()}
        for i, (shape, shape_file) in enumerate(shape_map.items()):
            for j in range(self.args.objects):
                pos = (5*i, 5*j, -10)
                object_num = p.loadURDF("pybullet_data/shapes/{}".format(shape_file), pos, self.default_orn, useFixedBase=True, globalScaling = self.args.object_size, physicsClientId=self.physicsClient)
                self.loaded[shape].append((object_num, (pos[0], pos[1], pos[2] + 1)))
                p.changeDynamics(object_num, 0, maxJointVelocity=10000)
                
    def object_faces_body(self, object_num):
        object_pos, _ = p.getBasePositionAndOrientation(object_num, physicsClientId = self.physicsClient)
        body_pos, _ = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
        x = body_pos[0] - object_pos[0]
        y = body_pos[1] - object_pos[1]
        yaw_angle_radians = atan2(y, x)
        orn = p.getQuaternionFromEuler([0, 0, yaw_angle_radians])
        p.resetBasePositionAndOrientation(object_num, (object_pos[0], object_pos[1], 1), orn, physicsClientId = self.physicsClient)
                                
    def begin(self, objects, goal, parented):
        self.set_pos((0, 0, 1))
        self.set_speed_and_yaw_speed(-1, 0)
        self.set_shoulder_pos()
        self.set_shoulder_speed()
        self.goal = goal
        self.parented = parented
        self.objects_in_play = {}
        self.watching = {}
        already_in_play = {key : 0 for key in shape_map.keys()}
        random_yaws = generate_angles(len(objects))
        for i, (shape_index, color_index) in enumerate(objects):
            shape = list(shape_map)[shape_index]
            color = list(color_map.values())[color_index]
            yaw = random_yaws[i]
            object_num, old_pos = self.loaded[shape][already_in_play[shape]]
            already_in_play[shape] += 1
            x = self.args.object_distance * sin(yaw)
            y = self.args.object_distance * cos(yaw)
            p.resetBasePositionAndOrientation(object_num, (x, y, 1), (0, 0, 0, 0), physicsClientId = self.physicsClient)
            self.object_faces_body(object_num)
            for i in range(p.getNumJoints(object_num)):
                if(i in [0, 1, 2, 3]): 
                    p.changeVisualShape(object_num, i, rgbaColor = (0,0,0,0), physicsClientId = self.physicsClient)
                else:
                    p.changeVisualShape(object_num, i, rgbaColor=color, physicsClientId = self.physicsClient)
            new_pos, _ = p.getBasePositionAndOrientation(object_num, physicsClientId = self.physicsClient)
            self.objects_in_play[(shape, color, old_pos)] = object_num
            self.watching[object_num] = 0
        
    def step(self):
        p.stepSimulation(physicsClientId = self.physicsClient)
        pos, _, _ = self.get_pos_yaw_spe()
        linear_velocity, angular_velocity = p.getBaseVelocity(self.body_num, physicsClientId=self.physicsClient)
        self.set_pos(pos)
        p.resetBaseVelocity(self.body_num, linearVelocity = linear_velocity, angularVelocity = angular_velocity, physicsClientId=self.physicsClient)
        for object_num in self.objects_in_play.values():
            self.object_faces_body(object_num)
            
    def end(self):
        for (shape, color, old_pos), object in self.objects_in_play.items():
            p.resetBasePositionAndOrientation(object, old_pos, self.default_orn, physicsClientId = self.physicsClient)
        self.set_shoulder_pos()
        self.set_shoulder_speed()
            
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
    
    def get_joint_angles(self):
        angles = []
        joint_names = [
            'body_shoulder_joint']
        for joint_name in joint_names:
            joint_index = get_joint_index(self.body_num, joint_name)
            joint_state = p.getJointState(self.body_num, joint_index, physicsClientId=self.physicsClient)
            angles.append(joint_state[0])  
        return angles
    
    def set_pos(self, pos):
        pos = (pos[0], pos[1], 1)
        _, yaw, _ = self.get_pos_yaw_spe()
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.body_num, pos, orn, physicsClientId = self.physicsClient)
    
    def set_speed_and_yaw_speed(self, speed, yaw_speed):
        speed = relative_to(speed, 0, self.args.max_speed)
        _, yaw, _ = self.get_pos_yaw_spe()
        speed = (speed * cos(yaw), speed * sin(yaw), 0)
        yaw_speed = relative_to(yaw_speed, -self.args.max_yaw_speed, self.args.max_yaw_speed)
        p.resetBaseVelocity(self.body_num, linearVelocity=0, angularVelocity=[0, 0, yaw_speed], physicsClientId=self.physicsClient)

    def set_shoulder_speed(self, shoulder = 0):
        shoulder = relative_to(shoulder, -self.args.max_shoulder_speed, self.args.max_shoulder_speed)
        for limb_name, target in [
            ('body_shoulder_joint', shoulder)]:
            limb_index = get_joint_index(self.body_num, limb_name)
            p.setJointMotorControl2(bodyUniqueId=self.body_num, 
                                    jointIndex=limb_index, 
                                    controlMode=p.VELOCITY_CONTROL, 
                                    targetVelocity=target, 
                                    force=1000)
            
    def set_shoulder_pos(self, shoulder = pi/2):
        for limb_name, target in [
            ('body_shoulder_joint', shoulder)]:
            limb_index = get_joint_index(self.body_num, limb_name)
            p.resetJointState(self.body_num, limb_index, target, physicsClientId=self.physicsClient)
            
    def find_link_index(self, object_id, link_name):
        num_joints = p.getNumJoints(object_id)
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(object_id, joint_index)
            if joint_info[12].decode('utf-8') == link_name:
                return joint_index
            
    def touching_sides(self, object_num):
        arm_link_index = self.find_link_index(self.body_num, "arm_link")
        left_side_index = self.find_link_index(object_num, "left")
        right_side_index = self.find_link_index(object_num, "right")
        top_side_index = self.find_link_index(object_num, "top")
        bottom_side_index = self.find_link_index(object_num, "bottom")
        
        is_touching_left = bool(p.getContactPoints(bodyA=self.body_num, bodyB=object_num, linkIndexA=arm_link_index, linkIndexB=left_side_index))
        is_touching_right = bool(p.getContactPoints(bodyA=self.body_num, bodyB=object_num, linkIndexA=arm_link_index, linkIndexB=right_side_index))
        is_touching_top = bool(p.getContactPoints(bodyA=self.body_num, bodyB=object_num, linkIndexA=arm_link_index, linkIndexB=top_side_index))
        is_touching_bottom = bool(p.getContactPoints(bodyA=self.body_num, bodyB=object_num, linkIndexA=arm_link_index, linkIndexB=bottom_side_index))
        
        return(is_touching_left, is_touching_right, is_touching_top, is_touching_bottom)
        
    def rewards(self):
        success = False
        failure = False
        goal_action = self.goal[0]
        goal_shape = list(shape_map)[self.goal[1][0]]
        goal_color = list(color_map.values())[self.goal[1][1]]
        for (shape, color, old_pos), object_num in self.objects_in_play.items():
            performing_something = None
            is_touching_left, is_touching_right, is_touching_top, is_touching_bottom = self.touching_sides(object_num)

            touching = is_touching_left or is_touching_right or is_touching_top or is_touching_bottom
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
            watching = abs(angle_radians) < pi/8 and not touching
            if watching: self.watching[object_num] += 1
            else:        self.watching[object_num] = 0 
            if(self.watching[object_num] >= self.args.watch_duration):
                performing_something = "WATCH"

            if(is_touching_top):
                performing_something = "TOP"
            if(is_touching_bottom):
                performing_something = "BOTTOM"
            if(is_touching_left):
                performing_something = "PORT"
            if(is_touching_right):
                performing_something = "STAR"
            
            if(performing_something != None):
                if(shape == goal_shape and color == goal_color and performing_something == goal_action.upper()): 
                    success = True
                else:
                    failure = True                
        win = success
        reward = self.args.reward if success else self.args.punishment if (failure and self.parented) else 0
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
            
        forward_rgbd = get_photo(pos, yaw)
        left_rgbd    = get_photo(pos, yaw + pi/2)
        back_rgbd    = get_photo(pos, yaw + pi)
        right_rgbd   = get_photo(pos, yaw - pi/2)
        
        rgbd = np.concatenate((back_rgbd[:,self.args.image_size//2:], left_rgbd, forward_rgbd, right_rgbd, back_rgbd[:,:self.args.image_size//2]), axis = 1)
        return(rgbd)
        
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    args = default_args
    from utils import make_objects_and_action
    arena = Arena(GUI = True)
    action, shape_colors_1, shape_colors_2 = make_objects_and_action(2, 5)
    goal = [choices(action_map)[0], shape_colors_1[0]]
    arena.begin(objects = shape_colors_1[:1], goal = goal, parented = False)
    i = -1
    going_up = True
    i_size = .1
    arena.set_shoulder_speed(args.max_shoulder_speed)
    while(True):
        arena.set_shoulder_speed(i)
        arena.set_speed_and_yaw_speed(i, i)
        i += i_size 
        if((going_up and i >= 3) or (not going_up and i <= -3)):
            i_size *= -1
            going_up = not going_up
        arena.step()
        arena.rewards()
        rgba = arena.photo_from_above()
        sleep(.1)
# %%
