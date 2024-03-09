#%%
import os
from random import choices, uniform, shuffle
import numpy as np
import pybullet as p
from math import pi, sin, cos, tan, radians, atan2, sqrt, isnan
from time import sleep
from skimage.transform import resize

from utils import default_args, shape_map, color_map, action_map, relative_to, make_objects_and_action#, print

def get_physics(GUI, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, -10, physicsClientId = physicsClient)
    p.setTimeStep(.005, physicsClientId=physicsClient)  # More accurate time step
    p.setPhysicsEngineParameter(numSolverIterations=50, physicsClientId=physicsClient)  # Increased solver iterations for potentially better stability
    return(physicsClient)
    
def get_joint_index(body_id, joint_name, physicsClient):
    num_joints = p.getNumJoints(body_id, physicsClientId = physicsClient)
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i, physicsClientId = physicsClient)
        if info[1].decode() == joint_name:
            return i
    return -1  # Return -1 if no joint with the given name is found



# FOV of agent vision.
fov_x_deg = 90
fov_y_deg = 90
near = 1.1
far = 8
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
        
        # Make floor and lower level.
        plane_positions = [[0, 0]]
        plane_ids = []
        
        for position in plane_positions:
            plane_id = p.loadURDF("plane.urdf", position + [0], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
            p.setCollisionFilterGroupMask(plane_id, -1, collisionFilterGroup = 1, collisionFilterMask = 1)
            plane_ids.append(plane_id)
            plane_id = p.loadURDF("plane.urdf", position + [-10], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
            p.setCollisionFilterGroupMask(plane_id, -1, collisionFilterGroup = 1, collisionFilterMask = 1)
            plane_ids.append(plane_id)
            
        # Place robot. 
        self.default_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId = self.physicsClient)
        self.robot_index = p.loadURDF("robot.urdf", (0, 0, 2.01), self.default_orn, useFixedBase=False, globalScaling = self.args.body_size, physicsClientId = self.physicsClient)
        
        p.changeVisualShape(self.robot_index, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = self.physicsClient)
        self.sensors = []
        for link_index in range(p.getNumJoints(self.robot_index, physicsClientId = self.physicsClient)):
            joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId = self.physicsClient)
            link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
            
            if(link_name == "left_wheel"):
                self.left_wheel = link_index 
            if(link_name == "right_wheel"):
                self.right_wheel = link_index
            if("sensor" in link_name):
                self.sensors.append((link_index, link_name))
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (1, 0, 0, 0), physicsClientId = self.physicsClient)
            else:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (0, 0, 0, 1), physicsClientId = self.physicsClient)
        
        # Place objects on lower level for future use.
        self.loaded = {key : [] for key in shape_map.keys()}
        self.object_indexs = []
        for i, (shape, shape_file) in enumerate(shape_map.items()):
            for j in range(self.args.objects):
                pos = (5*i, 5*j, -8.9)
                object_index = p.loadURDF("pybullet_data/shapes/{}".format(shape_file), pos, p.getQuaternionFromEuler([0, 0, pi/2]), useFixedBase=False, globalScaling = self.args.object_size, physicsClientId=self.physicsClient)
                self.loaded[shape].append((object_index, (pos[0], pos[1], -8.9)))
                self.object_indexs.append(object_index)
                                
    def begin(self, objects, goal, parented):
        self.set_pos()
        self.set_yaw()
        self.set_speeds()
        self.set_shoulder_pos()
        self.goal = goal
        self.parented = parented
        
        self.objects_in_play = {}
        self.watching = {}
        already_in_play = {key : 0 for key in shape_map.keys()}
        random_positions = self.generate_positions(len(objects))
        for i, (shape_index, color_index) in enumerate(objects):
            shape = list(shape_map)[shape_index]
            color = list(color_map.values())[color_index]
            object_index, old_pos = self.loaded[shape][already_in_play[shape]]
            already_in_play[shape] += 1
            x, y = random_positions[i]
            p.resetBasePositionAndOrientation(object_index, (x, y, 1.1), (0, 0, 0, 1), physicsClientId = self.physicsClient)
            self.object_faces_up(object_index)
            p.changeVisualShape(object_index, -1, rgbaColor = color, physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object_index)):
                p.changeVisualShape(object_index, i, rgbaColor=color, physicsClientId = self.physicsClient)
            self.objects_in_play[(shape, color, old_pos)] = object_index
            self.watching[object_index] = 0
        
    def step(self, left_wheel, right_wheel, shoulder, verbose = False):
        self.set_speeds(left_wheel, right_wheel, shoulder)
        for step in range(self.args.steps_per_step):
            if(verbose): WAITING = input("WAITING")
            p.stepSimulation(physicsClientId = self.physicsClient)
            self.set_speeds(left_wheel, right_wheel, shoulder)
            
    def set_speeds(self, left_wheel = 0, right_wheel = 0, shoulder = 0):
        left_wheel = relative_to(left_wheel, self.args.min_speed, self.args.max_speed)
        right_wheel = relative_to(right_wheel, self.args.min_speed, self.args.max_speed)
        linear_velocity = (left_wheel + right_wheel) / 2
        angular_velocity = (right_wheel - left_wheel) / self.args.angular_scaler
        _, yaw, _ = self.get_pos_yaw_spe()
        x = linear_velocity * cos(yaw)
        y = linear_velocity * sin(yaw)
        p.resetBaseVelocity(self.robot_index, linearVelocity=[x, y, 0], angularVelocity=[0, 0, angular_velocity], physicsClientId = self.physicsClient)
        shoulder = relative_to(shoulder, self.args.min_shoulder_angle, self.args.max_shoulder_angle)
        self.set_shoulder_pos(shoulder)
                        
    def end(self):
        for (shape, color, old_pos), object in self.objects_in_play.items():
            p.resetBasePositionAndOrientation(object, old_pos, self.default_orn, physicsClientId = self.physicsClient)
            
    def stop(self):
        p.disconnect(physicsClientId = self.physicsClient)
        
    def generate_positions(self, n):
        positions = [(0, 0)]
        while len(positions) < n + 1:
            angle = uniform(0, 2 * pi)
            radius = uniform(0, self.args.max_object_distance)
            x = radius * cos(angle)
            y = radius * sin(angle)
            if all(sqrt((x - px) ** 2 + (y - py) ** 2) >= self.args.min_object_separation for px, py in positions):
                positions.append((x, y))
        return positions[1:]
        
    def object_faces_up(self, object_index):
        pos, orn = p.getBasePositionAndOrientation(object_index, physicsClientId = self.physicsClient)
        x = pos[0]
        y = pos[1]
        (a, b, c) = p.getEulerFromQuaternion(orn, physicsClientId = self.physicsClient)
        orn = p.getQuaternionFromEuler([0, 0, c if not isnan(c) else 0])
        p.resetBasePositionAndOrientation(object_index, (x, y, 1.1), orn, physicsClientId = self.physicsClient)
        
    def get_pos_yaw_spe(self):
        pos, ors = p.getBasePositionAndOrientation(self.robot_index, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors, physicsClientId = self.physicsClient)[-1]
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
            joint_index = get_joint_index(self.robot_index, joint_name, physicsClient = self.physicsClient)
            joint_state = p.getJointState(self.robot_index, joint_index, physicsClientId=self.physicsClient)
            angles.append(joint_state[0])  
        return angles
    
    def set_pos(self, pos = (0, 0)):
        pos = (pos[0], pos[1], 2.01)
        _, yaw, _ = self.get_pos_yaw_spe()
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
        
    def set_yaw(self, yaw = 0):
        orn = p.getQuaternionFromEuler([0, 0, yaw], physicsClientId = self.physicsClient)
        pos, _, _ = self.get_pos_yaw_spe()
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
            
    def set_shoulder_pos(self, shoulder = pi/2):
        for limb_name, target in [
            ('body_shoulder_joint', shoulder)]:
            limb_index = get_joint_index(self.robot_index, limb_name, physicsClient = self.physicsClient)
            p.resetJointState(self.robot_index, limb_index, target, physicsClientId=self.physicsClient)
            
    def touching_object(self, object_index):
        touching = []
        for sensor_index, _ in self.sensors:
            touching.append(bool(p.getContactPoints(bodyA=self.robot_index, bodyB=object_index, linkIndexA=sensor_index, physicsClientId = self.physicsClient)))
        return(touching)
    
    def touching_anything(self):
        touching = [False for _ in self.sensors]
        for _, object_index in self.objects_in_play.items():
            touching_this_object = self.touching_object(object_index)
            for i, touch in enumerate(touching_this_object):
                if(touch):
                    touching[i] = True
        return(touching)
        
    def rewards(self):
        success = False
        failure = False
        goal_action = self.goal[0]
        goal_shape = list(shape_map)[self.goal[1][0][0]]
        goal_color = list(color_map.values())[self.goal[1][0][1]]
        for (shape, color, old_pos), object_index in self.objects_in_play.items():
            performing_something = None
            touching = self.touching_object(object_index)
            object_pos, _ = p.getBasePositionAndOrientation(object_index, physicsClientId=self.physicsClient)
            agent_pos, agent_ori = p.getBasePositionAndOrientation(self.robot_index, physicsClientId = self.physicsClient)
            distance_vector = np.subtract(object_pos, agent_pos)
            distance = np.linalg.norm(distance_vector)
            normalized_distance_vector = distance_vector / distance
            rotation_matrix = p.getMatrixFromQuaternion(agent_ori, physicsClientId = self.physicsClient)
            forward_vector = np.array([rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]])
            forward_vector /= np.linalg.norm(forward_vector)
            dot_product = np.dot(forward_vector, normalized_distance_vector)
            angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))  
            cross_product = np.cross(forward_vector, normalized_distance_vector)
            if cross_product[2] < 0:  
                angle_radians = -angle_radians
            watching = abs(angle_radians) < pi/8 and not any(touching) and distance <= self.args.watch_distance
            if watching: self.watching[object_index] += 1
            else:        self.watching[object_index] = 0 
            
            if(self.watching[object_index] >= self.args.watch_duration):
                performing_something = "WATCH"
            
            if(performing_something != None):
                if(shape == goal_shape and color == goal_color and performing_something == goal_action.upper()): 
                    success = True
                else:
                    failure = True                
        reward = self.args.reward if success else self.args.punishment if (failure and self.parented) else 0
        return(reward, success)
    
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
            proj_matrix = p.computeProjectionMatrix(left, right, bottom, top, near, far, physicsClientId = self.physicsClient)
            _, _, rgba, depth, _ = p.getCameraImage(
                width=self.args.image_size * 2, height=self.args.image_size * 2,
                projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
                physicsClientId = self.physicsClient)
            rgb = np.divide(rgba[:,:,:-1], 255)
            d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
            if(d.max() == d.min()): pass
            else: d = (d - d.min())/(d.max()-d.min())
            rgbd = np.concatenate([rgb, d], axis = -1)
            rgbd = resize(rgbd, (self.args.image_size, self.args.image_size, 4))
            return(rgbd)
  
        rgbd = get_photo(pos, yaw)
        
        return(rgbd)
        
    
    
if __name__ == "__main__":
    args = default_args
    arena = Arena(GUI = True, args = args)
    action, shape_colors_1, shape_colors_2 = make_objects_and_action(args.objects, 5)
    goal = [choices(action_map)[0], shape_colors_1]
    
    objectStartPosition = [5, 5, 1.5]  # Ensure object starts clearly above the floor
    objectStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    objectId = p.loadURDF("pybullet_data/shapes/1_WHOLE.urdf", objectStartPosition, objectStartOrientation, physicsClientId=arena.physicsClient)
    
    print("\nSPIN")
    arena.begin(objects = shape_colors_1, goal = goal, parented = False)
    for step in range(50):
        arena.step(-1, 1, -1, verbose = True)
        rgba = arena.photo_from_above()
        rgba = arena.photo_for_agent()
    
    print("\nPUSHING WITH BODY")
    arena.begin(objects = shape_colors_1, goal = goal, parented = False)
    for step in range(3):
        arena.step(1, 1, -1, verbose = True)
        rgba = arena.photo_from_above()
    arena.end()
    
    print("\nPUSHING WITH HAND")
    arena.begin(objects = shape_colors_1, goal = goal, parented = False)
    for step in range(3):
        arena.step(1, 1, 1, verbose = True)
        rgba = arena.photo_from_above()
    arena.end()
    
    print("\nPULLING WITH HAND")
    arena.begin(objects = shape_colors_1, goal = goal, parented = False)
    
    for step in range(3):
        arena.step(-1, -1, 1, verbose = True)
        rgba = arena.photo_from_above()
    arena.end()
        
    print("\nPUSHING LEFT")
    arena.begin(objects = shape_colors_1, goal = goal, parented = False)
    arena.step(1, -1, -1, verbose = True)
    for step in range(3):
        arena.step(-1, 1, 1, verbose = True)
        rgba = arena.photo_from_above()
    arena.end()
        
    print("\nPUSHING RIGHT")
    arena.begin(objects = shape_colors_1, goal = goal, parented = False)
    arena.step(-1, 1, -1, verbose = True)
    for step in range(3):
        arena.step(1, -1, 1, verbose = True)
        rgba = arena.photo_from_above()
    arena.end()