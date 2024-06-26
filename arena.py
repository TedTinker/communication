#%%
import os
import matplotlib.pyplot as plt
from random import uniform
import numpy as np
import pybullet as p
from math import pi, sin, cos, tan, radians, degrees, sqrt, isnan
from time import sleep
from skimage.transform import resize

from utils import default_args, shape_map, color_map, action_map, relative_to, opposite_relative_to, make_objects_and_action, duration#, print

def get_physics(GUI, time_step, steps_per_step, w = 10, h = 10):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId = physicsClient)
    p.setAdditionalSearchPath("pybullet_data")
    p.setGravity(0, 0, -100, physicsClientId = physicsClient)
    p.setTimeStep(time_step / steps_per_step, physicsClientId=physicsClient)  # More accurate time step
    p.setPhysicsEngineParameter(numSolverIterations=1, numSubSteps=1, physicsClientId=physicsClient)  # Increased solver iterations for potentially better stability
    return(physicsClient)
    
def get_joint_index(body_id, joint_name, physicsClient):
    num_joints = p.getNumJoints(body_id, physicsClientId = physicsClient)
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i, physicsClientId = physicsClient)
        if info[1].decode() == joint_name:
            return i
    return -1  # Return -1 if no joint with the given name is found

def adjust_sign(number1, number2):
    sign_of_first_number = -1 if number1 < 0 else 1
    min_abs_value = min(abs(number1), abs(number2))
    result = sign_of_first_number * min_abs_value
    return result

# FOV of agent vision.
fov_x_deg = 90
fov_y_deg = 90
fov_x_rad = radians(fov_x_deg)
fov_y_rad = radians(fov_y_deg)
near = 1
far = 9
right = near * tan(fov_x_rad / 2)
left = -right
top = near * tan(fov_y_rad / 2)
bottom = -top

agent_upper_starting_pos = 2.02
object_upper_starting_pos = 1.12
object_lower_start_pos = -8.85

if(__name__ == "__main__"):
    sensor_alpha = .5
else:
    sensor_alpha = 0



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
        self.robot_index = p.loadURDF("pybullet_data/robot.urdf", (0, 0, agent_upper_starting_pos), self.default_orn, useFixedBase=False, globalScaling = self.args.body_size, physicsClientId = self.physicsClient)
        
        p.changeVisualShape(self.robot_index, -1, rgbaColor = (.5,.5,.5,1), physicsClientId = self.physicsClient)
        p.changeDynamics(self.robot_index, -1, maxJointVelocity = 10000)
        self.sensors = []
        for link_index in range(p.getNumJoints(self.robot_index, physicsClientId = self.physicsClient)):
            joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId = self.physicsClient)
            link_name = joint_info[12].decode('utf-8')  # Child link name for the joint
            p.changeDynamics(self.robot_index, link_index, maxJointVelocity = 10000)
            
            if(link_name == "left_wheel"):
                self.left_wheel = link_index 
            if(link_name == "right_wheel"):
                self.right_wheel = link_index
            if("sensor" in link_name):
                self.sensors.append((link_index, link_name))
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (1, 0, 0, sensor_alpha), physicsClientId = self.physicsClient)
            else:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor = (0, 0, 0, 1), physicsClientId = self.physicsClient)
                        
        # Place objects on lower level for future use.
        self.loaded = {key : [] for key in shape_map.keys()}
        self.object_indexs = []
        for i, (shape, shape_name, shape_file) in shape_map.items():
            for j in range(self.args.objects):
                pos = (5*i, 5*j, object_lower_start_pos)
                object_index = p.loadURDF("pybullet_data/shapes/{}".format(shape_file), pos, p.getQuaternionFromEuler([0, 0, pi/2]), 
                                          useFixedBase=False, globalScaling = self.args.object_size, physicsClientId=self.physicsClient)
                p.changeDynamics(object_index, -1, maxJointVelocity = 10000)
                for link_index in range(p.getNumJoints(object_index, physicsClientId = self.physicsClient)):
                    joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId = self.physicsClient)
                    p.changeDynamics(object_index, link_index, maxJointVelocity = 10000)
                self.loaded[i].append((object_index, (pos[0], pos[1], object_lower_start_pos)))
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
        self.set_shoulder_angle()
        self.set_shoulder_speed()
        self.goal = goal
        self.parenting = parenting
        
        self.objects_in_play = {}
        self.watching = {}
        already_in_play = {key : 0 for key in shape_map.keys()}
        if(set_positions == None):
            random_positions = self.generate_positions(len(objects))
        else:
            random_positions = set_positions
        for i, (color_index, shape_index) in enumerate(objects):
            rgba = color_map[color_index][2]
            object_index, idle_pos = self.loaded[shape_index][already_in_play[shape_index]]
            already_in_play[shape_index] += 1
            x, y = random_positions[i]
            p.resetBasePositionAndOrientation(object_index, (x, y, object_upper_starting_pos), (0, 0, 0, 1), physicsClientId = self.physicsClient)
            self.object_faces_up(object_index)
            p.changeVisualShape(object_index, -1, rgbaColor = rgba, physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object_index)):
                p.changeVisualShape(object_index, i, rgbaColor=rgba, physicsClientId = self.physicsClient)
            self.objects_in_play[(color_index, shape_index, idle_pos)] = object_index
            self.watching[object_index] = 0
            
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.object_positions()
        self.objects_end = self.object_positions()
        self.objects_touch = self.touching_any_object()
        for object_index, touch_dict in self.objects_touch.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
        
    def step(self, left_wheel, right_wheel, shoulder, verbose = False, sleep_time = None):
        
        if(sleep_time != None):
            p.setTimeStep(self.args.time_step / self.args.steps_per_step, physicsClientId=self.physicsClient)  # More accurate time step
            
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.object_positions()
        touching = self.touching_any_object()
        for object_index, touch_dict in touching.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
 
        if(shoulder < 0): 
            shoulder = -self.args.max_shoulder_speed
        else:
            shoulder = self.args.max_shoulder_speed
        if(verbose): 
            WAITING = input("WAITING")
        for step in range(self.args.steps_per_step):
            self.set_shoulder_speed(shoulder) 
            self.set_wheel_speeds(left_wheel, right_wheel)
            if(sleep_time != None):
                sleep(sleep_time / self.args.steps_per_step)
            p.stepSimulation(physicsClientId = self.physicsClient)
            if(self.get_shoulder_angle() > self.args.max_shoulder_angle):
                self.set_shoulder_angle(shoulder = self.args.max_shoulder_angle)
                shoulder = 0
            if(self.get_shoulder_angle() < self.args.min_shoulder_angle):
                self.set_shoulder_angle(shoulder = self.args.min_shoulder_angle)
                shoulder = 0
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
            
    def set_pos(self, pos = (0, 0)):
        pos = (pos[0], pos[1], agent_upper_starting_pos)
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
        
    def set_yaw(self, yaw = 0):
        orn = p.getQuaternionFromEuler([0, 0, yaw], physicsClientId = self.physicsClient)
        pos, _, _ = self.get_pos_yaw_spe(self.robot_index)
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
            
    def set_wheel_speeds(self, left_wheel = 0, right_wheel = 0):
        left_wheel = relative_to(left_wheel, -self.args.max_speed, self.args.max_speed)
        right_wheel = relative_to(right_wheel, -self.args.max_speed, self.args.max_speed)
        linear_velocity = (left_wheel + right_wheel) / 2
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        x = linear_velocity * cos(yaw)
        y = linear_velocity * sin(yaw)
        angular_velocity = (right_wheel - left_wheel) * self.args.angular_scaler
        p.resetBaseVelocity(self.robot_index, linearVelocity=[x, y, 0], angularVelocity=[0, 0, angular_velocity], physicsClientId = self.physicsClient)
        
    def get_pos_yaw_spe(self, index):
        pos, ors = p.getBasePositionAndOrientation(index, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors, physicsClientId = self.physicsClient)[-1]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
        (vx, vy, _), _ = p.getBaseVelocity(index, physicsClientId=self.physicsClient)
        velocity_vec = np.array([vx, vy])
        spe = float(np.dot(velocity_vec, forward_dir))
        return(pos, yaw, spe)
    
    def set_shoulder_speed(self, shoulder = 0):
        joint_index = get_joint_index(self.robot_index, 'body_left_shoulder_joint', physicsClient = self.physicsClient)
        shoulder = relative_to(shoulder, -self.args.max_shoulder_speed, self.args.max_shoulder_speed)
        p.setJointMotorControl2(self.robot_index, joint_index, controlMode = p.VELOCITY_CONTROL, targetVelocity = shoulder, physicsClientId=self.physicsClient)
        
    def set_shoulder_angle(self, shoulder = None):
        if(shoulder == None):
            shoulder = self.args.max_shoulder_angle
        limb_index = get_joint_index(self.robot_index, 'body_left_shoulder_joint', physicsClient = self.physicsClient)
        p.resetJointState(self.robot_index, limb_index, shoulder, physicsClientId=self.physicsClient)
        
    def get_shoulder_angle(self):
        joint_index = get_joint_index(self.robot_index, 'body_left_shoulder_joint', physicsClient = self.physicsClient)
        joint_state = p.getJointState(self.robot_index, joint_index, physicsClientId=self.physicsClient)
        return joint_state[0]
        
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
        pos, orn = p.getBasePositionAndOrientation(object_index, physicsClientId = self.physicsClient)
        x = pos[0]
        y = pos[1]
        (a, b, c) = p.getEulerFromQuaternion(orn, physicsClientId = self.physicsClient)
        orn = p.getQuaternionFromEuler([0, 0, c if not isnan(c) else 0])
        p.resetBasePositionAndOrientation(object_index, (x, y, object_upper_starting_pos), orn, physicsClientId = self.physicsClient)
            
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
    
    def object_positions(self):
        object_positions = []
        for object_index in self.objects_in_play.values():
            pos, _, _ = self.get_pos_yaw_spe(object_index)
            object_positions.append(pos)
        return(object_positions)
        
    def rewards(self, verbose = False):
        win = False
        reward = 0
        goal_action = self.goal[0]
        goal_color = self.goal[1]
        goal_shape = self.goal[2]
        v_rx = cos(self.robot_start_yaw)
        v_ry = sin(self.robot_start_yaw)
        
        if(verbose):
            for object_key, object_dict in self.objects_touch.items():
                for link_name, value in object_dict.items():
                    if(value):
                        print(f"Touching {object_key} with {link_name}.")
                        
        objects_goals = {}
                
        for i, ((color_index, shape_index, _), object_index) in enumerate(self.objects_in_play.items()):
            watching = False 
            pushing = False 
            pulling = False 
            lefting = False 
            righting = False
            
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
                print("Object movement (forward):", round(movement_forward, 2))
                print("Object movement (left):", round(movement_left, 2))
                print("Angle of object movement:", round(v_rx, 2), round(v_ry, 2))
            
            # Is the agent watching an object?
            watching_now = abs(angle_radians) < pi/6 and not touching and distance <= self.args.watch_distance
            if watching_now: self.watching[object_index] += 1
            else:        self.watching[object_index] = 0 
            if(self.watching[object_index] >= self.args.watch_duration):
                watching = True
                        
            # Is the object pushed/pulled away from its starting position, relative to the agent's starting position and angle?
            if(movement_forward >= self.args.push_amount and touching):
                pushing = True
            if(movement_forward <= -self.args.pull_amount and touching and abs(angle_radians) < pi/2):
                pulling = True
                        
            # Is the object pushed left/right from its starting position, relative to the agent's starting position and angle?
            if(movement_left >= self.args.left_right_amount and touching):
                lefting = True
            if(movement_left <= -self.args.left_right_amount and touching):
                righting = True
                    
            # Also add reward for mere distance.
            midpoint = (self.args.dist_reward_min + self.args.dist_reward_max) / 2
            if distance <= self.args.dist_reward_min:
                distance_reward = 1
            elif distance <= midpoint:
                proportion = (distance - self.args.dist_reward_min) / (midpoint - self.args.dist_reward_min)
                distance_reward = (1 - proportion)
            elif(distance >= midpoint and distance <= self.args.dist_reward_max):
                proportion = (distance - midpoint) / (self.args.dist_reward_max - midpoint)
                distance_reward = -proportion
            else:
                distance_reward = -1
            distance_reward *= self.args.dist_reward
                
            # Also add reward for mere angle.
            abs_angle = abs(angle_degrees)
            midpoint = (self.args.angle_reward_min + self.args.angle_reward_max) / 2
            if abs_angle <= self.args.angle_reward_min:
                angle_reward = 1
            elif abs_angle <= midpoint:
                proportion = (abs_angle - self.args.angle_reward_min) / (midpoint - self.args.angle_reward_min)
                angle_reward = (1 - proportion)
            elif(abs_angle >= midpoint and abs_angle <= self.args.angle_reward_max):
                proportion = (abs_angle - midpoint) / (self.args.angle_reward_max - midpoint)
                angle_reward = -proportion
            else:
                angle_reward = -1
            if(distance_reward < 0 and angle_reward > 0):
                angle_reward *= 0
            angle_reward *= self.args.angle_reward
                    
            objects_goals[(color_index, shape_index)] = [watching, pushing, pulling, lefting, righting, distance_reward, angle_reward]
                        
        which_goal_message = " " * self.args.max_comm_len
        for (color, shape), (watching, pushing, pulling, lefting, righting, distance_reward, angle_reward) in objects_goals.items():
            if(watching or pushing or pulling or lefting or righting):
                if(watching): action_char = action_map[0][0]
                if(pushing):  action_char = action_map[1][0]
                if(pulling):  action_char = action_map[2][0]
                if(lefting):  action_char = action_map[3][0]
                if(righting): action_char = action_map[4][0]
                color_char = color_map[color][0]
                shape_char = shape_map[shape][0]
                which_goal_message = action_char + color_char + shape_char + "   "
            if(color == goal_color and shape == goal_shape):
                if(sum([watching, pushing, pulling, lefting, righting]) == 1):
                    if((action_map[goal_action][1] == "WATCH" and watching) or 
                    (action_map[goal_action][1] == "PUSH" and pushing) or
                    (action_map[goal_action][1] == "PULL" and pulling) or
                    (action_map[goal_action][1] == "LEFT" and lefting) or
                    (action_map[goal_action][1] == "RIGHT" and righting)):   
                        win = True 
                        reward = self.args.reward
            else:
                if(watching or pushing or pulling or lefting or righting):
                    reward = self.args.wrong_object_punishment
                                           
        [watching, pushing, pulling, lefting, righting, distance_reward, angle_reward] = objects_goals[(goal_color, goal_shape)]
        if(action_map[goal_action][1] == "FREE_PLAY"):
            reward = 0 if which_goal_message == " " * self.args.max_comm_len else self.args.free_play_reward
            if(self.args.free_play_reward_dist):
                distance_reward = max([distance_reward for _, (_, _, _, _, _, distance_reward, _) in objects_goals.items()])
                angle_reward = max([angle_reward for _, (_, _, _, _, _, _, angle_reward) in objects_goals.items()])
            else:
                distance_reward = 0
                angle_reward = 0
            
        if(verbose):
            print("\nWhich goal message:\'" + which_goal_message, "\'")
            print("Raw reward:", round(reward, 2))
            print("Distance:", round(distance, 2))
            print("Distance reward:", round(distance_reward, 2))
            print("Angle:", round(angle_degrees, 2))
            print("Angle reward:", round(angle_reward, 2))
            print("Total reward:", reward + distance_reward + angle_reward)
            print("Win:", win)
            
        return(reward, distance_reward, angle_reward, win, which_goal_message)
    
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
            cameraEyePosition = [pos[0], pos[1], 2], 
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
    args = default_args
    physicsClient = get_physics(GUI = True, time_step = args.time_step, steps_per_step = args.steps_per_step)
    arena = Arena(physicsClient, args = args)
    sleep_time = 1
    
    def show_them():
        above_rgba = arena.photo_from_above()
        agent_rgbd = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(agent_rgbd[:,:,:-1])
        plt.show()
        plt.close()
        
        
        
    action, colors_shapes_1, colors_shapes_2 = make_objects_and_action(
        num_objects = 3,
        allowed_actions = [0],
        allowed_colors = [0, 1, 2, 3, 4, 5, 6],
        allowed_shapes = [0, 1, 2])
        
        
    """
    print("\nFREE PLAY")
    goal = [-1, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False)
    steps = 0
    while(True):
        steps += 1
        p.stepSimulation(physicsClientId = arena.physicsClient)
        if(steps % 50 == 0): 
            print("\n\n")
            arena.rewards(verbose = True)
            
        sleep(.05)
    arena.end()
    """
    
    
    
    action, colors_shapes_1, colors_shapes_2 = make_objects_and_action(
        num_objects = 1,
        allowed_actions = [0],
        allowed_colors = [0],
        allowed_shapes = [1])
    
    
    """    
    print("\nHIT")
    goal = [0, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(4.5,0)])
    show_them()
    reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    for j in range(3):
        arena.step(0, 0, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    arena.end()
    """
        
        
        
    """    
    print("\nAPPROACH")
    goal = [0, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(15,0)])
    show_them()
    reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    for j in range(15):
        arena.step(1, 1, 1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    arena.end()
    """
    
    """
    print("\nAIM")
    goal = [0, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(8,0)])
    show_them()
    reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    for j in range(15):
        arena.step(-.1, .1, 1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    arena.end()
    """
    
    """
    print("\nNO GOAL")
    goal = [0, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(8,0)])
    show_them()
    reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    i = 1
    show_them()
    reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    for j in range(5):
        i *= -1
        arena.step(1, -1, i, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    arena.end()
    """
    
    """
    print("\nWATCH")
    goal = [0, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(6,0)])
    show_them()
    reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
    while(True):
        arena.step(0, 0, 1, verbose = True, sleep_time = .1)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    """
    
    """
    print("\nPUSH")
    goal = [1, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(5,0)])
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(1, 1, 1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    """
    
    #"""
    print("\nPULL")
    goal = [2, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(3,0)])
    show_them()
    arena.rewards(verbose = True)
    arena.step(0, 0, -1, verbose = True, sleep_time = sleep_time)
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(-1, -1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
    
    #"""
    print("\nPULL BACKWARD")
    goal = [2, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(-3,0)])
    show_them()
    arena.rewards(verbose = True)
    for i in range(4):
        arena.step(-1, -1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
        if(win):
            break
    #arena.end()
    """
    
    #"""   
    print("\nLEFT")
    goal = [3, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(3,0)])
    show_them()
    arena.rewards(verbose = True)
    arena.step(0, 0, -1, verbose = True, sleep_time = sleep_time)
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(-1, 1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
    
    #"""   
    print("\nLEFT AGAIN")
    goal = [3, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(3,3)])
    show_them()
    arena.rewards(verbose = True)
    arena.step(0, 0, -1, verbose = True, sleep_time = sleep_time)
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(-1, 1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
    
    #"""
    print("\nRIGHT")
    goal = [4, colors_shapes_1[0][0], colors_shapes_1[0][1]]
    arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(3,0)])
    show_them()
    arena.rewards(verbose = True)
    arena.step(0, 0, -1, verbose = True, sleep_time = sleep_time)
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(1, -1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
# %%