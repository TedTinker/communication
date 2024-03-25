#%%
import os
import matplotlib.pyplot as plt
from random import uniform
import numpy as np
import pybullet as p
from math import pi, sin, cos, tan, radians, sqrt, isnan
from time import sleep
from skimage.transform import resize

from utils import default_args, shape_map, color_map, action_map, relative_to, opposite_relative_to, make_objects_and_action#, print

def get_physics(GUI, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, -10, physicsClientId = physicsClient)
    p.setTimeStep(.005, physicsClientId=physicsClient)  # More accurate time step
    p.setPhysicsEngineParameter(numSolverIterations=10, physicsClientId=physicsClient)  # Increased solver iterations for potentially better stability
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
    def __init__(self, physicsClient, args = default_args):
        self.args = args
        
        self.physicsClient = physicsClient
        self.objects_in_play = {}
        self.watching = {}
        
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
        for i, (shape, shape_name, shape_file) in shape_map.items():
            for j in range(self.args.objects):
                pos = (5*i, 5*j, -8.9)
                object_index = p.loadURDF("pybullet_data/shapes/{}".format(shape_file), pos, p.getQuaternionFromEuler([0, 0, pi/2]), 
                                          useFixedBase=False, globalScaling = self.args.object_size, physicsClientId=self.physicsClient)
                self.loaded[i].append((object_index, (pos[0], pos[1], -8.9)))
                self.object_indexs.append(object_index)
                                
    def begin(self, objects, goal, parented, set_positions = None):
        self.set_pos()
        self.set_yaw()
        self.set_speeds()
        self.set_shoulder_angle()
        self.goal = goal
        self.parented = parented
        
        self.objects_in_play = {}
        self.watching = {}
        already_in_play = {key : 0 for key in shape_map.keys()}
        if(set_positions == None):
            random_positions = self.generate_positions(len(objects))
        else:
            random_positions = set_positions
        for i, (color_index, shape_index) in enumerate(objects):
            rgba = color_map[color_index][2]
            object_index, old_pos = self.loaded[shape_index][already_in_play[shape_index]]
            already_in_play[shape_index] += 1
            x, y = random_positions[i]
            p.resetBasePositionAndOrientation(object_index, (x, y, 1.1), (0, 0, 0, 1), physicsClientId = self.physicsClient)
            self.object_faces_up(object_index)
            p.changeVisualShape(object_index, -1, rgbaColor = rgba, physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object_index)):
                p.changeVisualShape(object_index, i, rgbaColor=rgba, physicsClientId = self.physicsClient)
            self.objects_in_play[(color_index, shape_index, old_pos)] = object_index
            self.watching[object_index] = 0
        
    def step(self, left_wheel, right_wheel, shoulder, verbose = False, sleep_time = None):
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = []
        for object_index in self.objects_in_play.values():
            pos, _, _ = self.get_pos_yaw_spe(object_index)
            self.objects_start.append(pos)
        if(shoulder < 0): 
            shoulder = -1
        else:
            shoulder = 1
        shoulder_start = opposite_relative_to(self.get_shoulder_angle(), self.args.min_shoulder_angle, self.args.max_shoulder_angle)
        shoulder_step = (shoulder - shoulder_start) / (self.args.steps_per_step - 1)
        self.set_speeds(left_wheel, right_wheel)
        for step in range(self.args.steps_per_step):
            if(verbose): 
                if(step == 0):
                    WAITING = input("WAITING")
            if(sleep_time != None):
                sleep(sleep_time/self.args.steps_per_step)
            self.set_shoulder_angle(shoulder_start + step * shoulder_step)
            p.stepSimulation(physicsClientId = self.physicsClient)
        self.objects_end = []
        for object_index in self.objects_in_play.values():
            pos, _, _ = self.get_pos_yaw_spe(object_index)
            self.objects_end.append(pos)
            
    def set_speeds(self, left_wheel = 0, right_wheel = 0):
        left_wheel = relative_to(left_wheel, self.args.min_speed, self.args.max_speed)
        right_wheel = relative_to(right_wheel, self.args.min_speed, self.args.max_speed)
        linear_velocity = (left_wheel + right_wheel) / 2
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        x = linear_velocity * cos(yaw)
        y = linear_velocity * sin(yaw)
        angular_velocity = (right_wheel - left_wheel) * self.args.angular_scaler
        p.resetBaseVelocity(self.robot_index, linearVelocity=[x, y, 0], angularVelocity=[0, 0, angular_velocity], physicsClientId = self.physicsClient)
        
    def set_shoulder_angle(self, shoulder = -1):
        shoulder = relative_to(shoulder, self.args.min_shoulder_angle, self.args.max_shoulder_angle)
        limb_index = get_joint_index(self.robot_index, 'body_shoulder_joint', physicsClient = self.physicsClient)
        p.resetJointState(self.robot_index, limb_index, shoulder, physicsClientId=self.physicsClient)
                        
    def end(self):
        for (_, _, old_pos), object_index in self.objects_in_play.items():
            p.resetBasePositionAndOrientation(object_index, old_pos, self.default_orn, physicsClientId = self.physicsClient)
            
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
        
    def get_pos_yaw_spe(self, index):
        pos, ors = p.getBasePositionAndOrientation(index, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors, physicsClientId = self.physicsClient)[-1]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
        (vx, vy, _), _ = p.getBaseVelocity(index, physicsClientId=self.physicsClient)
        velocity_vec = np.array([vx, vy])
        spe = float(np.dot(velocity_vec, forward_dir))
        return(pos, yaw, spe)
    
    def get_shoulder_angle(self):
        joint_index = get_joint_index(self.robot_index, 'body_shoulder_joint', physicsClient = self.physicsClient)
        joint_state = p.getJointState(self.robot_index, joint_index, physicsClientId=self.physicsClient)
        return joint_state[0]
    
    def set_pos(self, pos = (0, 0)):
        pos = (pos[0], pos[1], 2.01)
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
        
    def set_yaw(self, yaw = 0):
        orn = p.getQuaternionFromEuler([0, 0, yaw], physicsClientId = self.physicsClient)
        pos, _, _ = self.get_pos_yaw_spe(self.robot_index)
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
            
    def touching_object(self, object_index):
        touching = []
        for sensor_index, _ in self.sensors:
            touching.append(bool(p.getContactPoints(bodyA=self.robot_index, bodyB=object_index, linkIndexA=sensor_index, physicsClientId = self.physicsClient)))
        return(touching)
    
    def touching_anything(self):
        touching = {}
        for object_index in self.objects_in_play.values():
            touching_this_object = self.touching_object(object_index)
            touching[object_index] = touching_this_object
        return(touching)
        
    def rewards(self, verbose = False):
        success = False
        failure = False
        goal_action = self.goal[0]
        goal_shape = self.goal[1]
        goal_color = self.goal[2]
        v_rx = cos(self.robot_start_yaw)
        v_ry = sin(self.robot_start_yaw)
        
        for i, ((color_index, shape_index, _), object_index) in enumerate(self.objects_in_play.items()):
            watching = 0
            pushing = 0
            pulling = 0
            lefting = 0
            righting = 0
            
            # Is the agent watching an object?
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
            watching_now = abs(angle_radians) < pi/6 and not any(touching) and distance <= self.args.watch_distance
            if watching_now: self.watching[object_index] += 1
            else:        self.watching[object_index] = 0 
            if(self.watching[object_index] >= self.args.watch_duration):
                watching = self.watching[object_index]
                
            if(verbose):
                print(self.watching)
            
            # How is the object moving? 
            (x_before, y_before, z_before) = self.objects_start[i]
            (x_after, y_after, z_after) = self.objects_end[i]
            delta_x = x_after - x_before
            delta_y = y_after - y_before
            if(verbose):
                print("\nChange:", delta_x, delta_y)
                print("Angle:", v_rx, v_ry)
            
            # Is the object pushed/pulled away from its starting position, relative to the agent's starting position and angle?
            movement_forward = delta_x * v_rx + delta_y * v_ry
            if(movement_forward >= self.args.push_amount and any(touching)):
                pushing = movement_forward
            if(movement_forward <= -self.args.push_amount and any(touching)):
                pulling = -movement_forward
            # Is the object pushed left/right from its starting position, relative to the agent's starting position and angle?
            movement_left = delta_x * (-v_ry) + delta_y * v_rx
            if(movement_left >= self.args.push_amount and any(touching)):
                lefting = movement_left
            if(movement_left <= self.args.push_amount and any(touching)):
                righting = -movement_left
                
            action_values = {
                "WATCH": watching,
                "PUSH": pushing,
                "PULL": pulling,
                "LEFT": lefting,
                "RIGHT": righting}
            
            if(verbose):
                print("WATCH:", watching)
                print("PUSH:", pushing)
                print("PULL:", pulling)
                print("LEFT:", lefting)
                print("RIGHT:", righting)

            # Find the action with the largest value
            largest_action = max(action_values, key=action_values.get)

            # Check if the largest value is greater than 0 (or another threshold if needed)
            if action_values[largest_action] > 0:
                performing_something = largest_action
            else:
                performing_something = None
            
            if(performing_something != None):
                if(shape_index == goal_shape and color_index == goal_color and performing_something == action_map[goal_action][1]): 
                    success = True
                else:
                    failure = True                
        reward = self.args.reward if success else self.args.punishment if (failure and self.parented) else 0
        return(reward, success)
    
    def photo_from_above(self):
        pos, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        
        def get_photo(temp_yaw):
            x, y = 4 * cos(temp_yaw), 4 * sin(temp_yaw)
            view_matrix = p.computeViewMatrix(
                cameraEyePosition = [pos[0] + x, pos[1] + y, 6], 
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
        temp_yaws = [yaw + pi, yaw + -pi/2, yaw , yaw + pi/2, ]
        rgbas = [get_photo(temp_yaw) for temp_yaw in temp_yaws]
        black_separator = np.zeros((rgbas[0].shape[0], 10, 4), dtype=np.uint8)
        rgbas.insert(3, black_separator)
        rgbas.insert(2, black_separator)
        rgbas.insert(1, black_separator)
        rgba = np.concatenate(rgbas, axis = 1)
        return(rgba)
    
    def photo_for_agent(self):
        pos, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        
        def get_photo(pos, yaw):
            x, y = cos(yaw), sin(yaw)
            view_matrix = p.computeViewMatrix(
                cameraEyePosition = [pos[0], pos[1], 2], 
                cameraTargetPosition = [pos[0] + x*2, pos[1] + y*2, 2],    
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
    physicsClient = get_physics(GUI = True)
    arena = Arena(physicsClient, args = args)
    sleep_time = 3
    
    action, colors_shapes_1, colors_shapes_2 = make_objects_and_action(1, 1, 1, 1)
    
    
    """
    print("\nPUSHING WITH BODY")
    goal = [1, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parented = False, set_positions = [(3,0)])
    for step in range(3):
        arena.step(1, 1, -1, verbose = True, sleep_time = sleep_time)
        above_rgba = arena.photo_from_above()
        agent_rgba = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgba)
        plt.show()
        plt.close()
        print(arena.rewards(verbose = True)[0])
    arena.end()
    """
    
    """
    print("\nPUSHING WITH HAND")
    goal = [1, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parented = False, set_positions = [(8,0)])
    arena.step(0, 0, 1, verbose = True, sleep_time = sleep_time)
    for step in range(3):
        arena.step(1, 1, 1, verbose = True, sleep_time = sleep_time)
        above_rgba = arena.photo_from_above()
        agent_rgba = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgba)
        plt.show()
        plt.close()
        print(arena.rewards(verbose = True)[0])
    arena.end()
    """
    
    """
    print("\nSPIN")
    goal = [0, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parented = False, set_positions = [(3,3)])
    arena.step(0, 0, 1, verbose = True, sleep_time = sleep_time)
    for step in range(5):
        arena.step(1, -1, 1, verbose = True, sleep_time = sleep_time)
        above_rgba = arena.photo_from_above()
        agent_rgba = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgba)
        plt.show()
        plt.close()
        print(arena.rewards(verbose = True)[0])
    arena.end()
    """
    
    #"""
    print("\nWATCH")
    goal = [0, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parented = False, set_positions = [(3,0)])
    for step in range(5):
        arena.step(0, 0, -1, verbose = True, sleep_time = .1)
        above_rgba = arena.photo_from_above()
        agent_rgba = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgba)
        plt.show()
        plt.close()
        print(arena.rewards(verbose = True)[0])
    arena.end()
    #"""
    
    
    """
    print("\nPULLING WITH HAND")
    goal = [2, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parented = False, set_positions = [(3,0)])
    arena.step(0, 0, 1, verbose = True, sleep_time = sleep_time)
    for step in range(3):
        arena.step(-1, -1, 1, verbose = True, sleep_time = sleep_time)
        above_rgba = arena.photo_from_above()
        agent_rgba = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgba)
        plt.show()
        plt.close()
        print(arena.rewards(verbose = True)[0])
    arena.end()
    """
    
    """   
    print("\nPUSHING LEFT")
    goal = [3, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parented = False, set_positions = [(3,0)])
    arena.step(1, -1, -1, verbose = True, sleep_time = sleep_time)
    arena.step(0, 0, 1, verbose = True, sleep_time = sleep_time)
    for step in range(3):
        arena.step(-.5, .5, 1, verbose = True, sleep_time = sleep_time)
        above_rgba = arena.photo_from_above()
        agent_rgba = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgba)
        plt.show()
        plt.close()
        print(arena.rewards(verbose = True)[0])
    arena.end()
    """
    
    """
    print("\nPUSHING RIGHT")
    goal = [4, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parented = False, set_positions = [(3,0)])
    arena.step(-1, 1, -1, verbose = True, sleep_time = sleep_time)
    arena.step(0, 0, 1, verbose = True, sleep_time = sleep_time)
    for step in range(3):
        arena.step(.5, -.5, 1, verbose = True, sleep_time = sleep_time)
        above_rgba = arena.photo_from_above()
        agent_rgba = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgba)
        plt.show()
        plt.close()
        print(arena.rewards(verbose = True)[0])
    arena.end()
    """