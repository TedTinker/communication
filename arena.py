#%%
import matplotlib.pyplot as plt
from random import uniform
import numpy as np
import pybullet as p
import math
from math import pi, sin, cos, tan, radians, degrees, isnan
from time import sleep
from skimage.transform import resize

from utils import default_args, Object, Goal, action_map, shape_map, relative_to, make_objects_and_action



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



# FOV of robot vision.
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

robot_upper_starting_pos = 2.02
object_upper_starting_pos = 1.12
object_lower_starting_pos = -8.85

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
        p.loadURDF("plane.urdf", [0, 0, 0], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
        p.loadURDF("plane.urdf", [0, 0, -10], globalScaling=2, useFixedBase=True, physicsClientId=self.physicsClient)
            
        # Place robot. 
        self.robot_default_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId = self.physicsClient)
        self.robot_default_pos = (0, 0, robot_upper_starting_pos)
        self.robot_index = p.loadURDF("pybullet_data/robot.urdf", self.robot_default_pos, self.robot_default_orn, useFixedBase=False, globalScaling = self.args.body_size, physicsClientId = self.physicsClient)
        
        # Color robot.
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
                        
        # Place objects on lower level for future use. Save details about objects. 
        self.all_objects = {shape.name : [] for shape in shape_map.values()}
        self.objects_in_play = []
        for i, shape in shape_map.items():
            for j in range(2):
                default_pos = (5*i, 5*j, object_lower_starting_pos)
                object_index = p.loadURDF("pybullet_data/shapes/{}".format(shape.file), default_pos, p.getQuaternionFromEuler([0, 0, pi/2]), 
                                          useFixedBase=False, globalScaling = self.args.object_size, physicsClientId=self.physicsClient)
                p.changeDynamics(object_index, -1, maxJointVelocity = 10000)
                for link_index in range(p.getNumJoints(object_index, physicsClientId = self.physicsClient)):
                    joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId = self.physicsClient)
                    p.changeDynamics(object_index, link_index, maxJointVelocity = 10000)
                self.all_objects[shape.name].append(Object(object_index, default_pos, None, shape))
                
    def end(self):
        for object in self.objects_in_play:
            p.resetBasePositionAndOrientation(object.index, object.default_pos, self.robot_default_orn, physicsClientId = self.physicsClient)
            
    def stop(self):
        p.disconnect(physicsClientId = self.physicsClient)
                                        
    def begin(self, color_shapes, goal, set_positions = None):
        self.set_robot_pos()
        self.set_robot_yaw()
        self.set_wheel_speeds()
        self.set_shoulder_angle(self.args.max_shoulder_angle, self.args.max_shoulder_angle)
        self.set_shoulder_speed()
        self.goal = goal
        
        self.objects_in_play = []
        self.watching = {}
        already_in_play = {shape.name : 0 for shape in shape_map.values()}
        if(set_positions == None):
            random_positions = self.generate_object_positions(len(color_shapes))
        else:
            random_positions = set_positions
        for i, (color, shape) in enumerate(color_shapes):
            rgba = color.rgba
            object = self.all_objects[shape.name][already_in_play[shape.name]]
            object = Object(object.index, object.default_pos, color, object.shape)
            already_in_play[shape.name] += 1
            x, y = random_positions[i]
            p.resetBasePositionAndOrientation(object.index, (x, y, object_upper_starting_pos), (0, 0, 0, 1), physicsClientId = self.physicsClient)
            self.object_faces_up(object)
            
            link_name = p.getBodyInfo(object.index)[0].decode('utf-8')
            if "white" not in link_name.lower():
                p.changeVisualShape(object.index, -1, rgbaColor = rgba, physicsClientId = self.physicsClient)
            else:
                p.changeVisualShape(object.index, -1, rgbaColor = (1, 1, 1, 1), physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object.index)):
                joint_info = p.getJointInfo(object.index, i, physicsClientId=self.physicsClient)
                joint_name = joint_info[1].decode("utf-8") 
                if "white" not in joint_name.lower(): 
                    p.changeVisualShape(object.index, i, rgbaColor=rgba, physicsClientId=self.physicsClient)
            self.objects_in_play.append(object)
            self.watching[object.index] = 0
            
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.object_positions()
        self.objects_end = self.object_positions()
        self.objects_touch = self.touching_any_object()
        for object, touch_dict in self.objects_touch.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
             
    def step(self, left_wheel, right_wheel, left_shoulder, right_shoulder, verbose = False, sleep_time = None):
        
        if(sleep_time != None):
            p.setTimeStep(self.args.time_step / self.args.steps_per_step, physicsClientId=self.physicsClient)  # More accurate time step
            
        self.robot_start_yaw = self.get_pos_yaw_spe(self.robot_index)[1]
        self.objects_start = self.object_positions()
        touching = self.touching_any_object()
        for object, touch_dict in touching.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
 
        if(left_shoulder < 0): 
            left_shoulder = -1
        else:
            left_shoulder = 1
        if(right_shoulder < 0):
            right_shoulder = -1
        else:
            right_shoulder = 1

        if(verbose): 
            WAITING = input("WAITING")
        for step in range(self.args.steps_per_step):
            self.set_wheel_speeds(left_wheel, right_wheel)
            self.set_shoulder_speed(left_shoulder, right_shoulder) 
            if(sleep_time != None):
                sleep(sleep_time / self.args.steps_per_step)
            p.stepSimulation(physicsClientId = self.physicsClient)
            
            left_shoulder_angle, right_shoulder_angle = self.get_shoulder_angle()
            if(left_shoulder_angle > self.args.max_shoulder_angle):
                self.set_shoulder_angle(left_shoulder = self.args.max_shoulder_angle)
                left_shoulder = 0
            if(left_shoulder_angle < self.args.min_shoulder_angle):
                self.set_shoulder_angle(left_shoulder = self.args.min_shoulder_angle)
                left_shoulder = 0
                
            if(right_shoulder_angle > self.args.max_shoulder_angle):
                self.set_shoulder_angle(right_shoulder = self.args.max_shoulder_angle)
                right_shoulder = 0
            if(right_shoulder_angle < self.args.min_shoulder_angle):
                self.set_shoulder_angle(right_shoulder = self.args.min_shoulder_angle)
                right_shoulder = 0
                
            touching_now = self.touching_any_object()
            for object, touch_dict in touching_now.items():
                for body_part, value in touch_dict.items():
                    if(value):
                        touching[object][body_part] += 1/self.args.steps_per_step
                        if(touching[object][body_part]) > 1:
                            touching[object][body_part] = 1
                
        self.objects_end = self.object_positions()
        self.objects_touch = touching
            
        if(sleep_time != None):
            p.setTimeStep(self.args.time_step / self.args.steps_per_step, physicsClientId=self.physicsClient)  # More accurate time step
            p.setPhysicsEngineParameter(numSolverIterations=1, numSubSteps=1, physicsClientId=self.physicsClient)
            
    def get_pos_yaw_spe(self, index):
        pos, ors = p.getBasePositionAndOrientation(index, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors, physicsClientId = self.physicsClient)[-1]
        forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
        (vx, vy, _), _ = p.getBaseVelocity(index, physicsClientId=self.physicsClient)
        velocity_vec = np.array([vx, vy])
        spe = float(np.dot(velocity_vec, forward_dir))
        return(pos, yaw, spe)
            
    def set_robot_pos(self, pos = (0, 0)):
        pos = (pos[0], pos[1], robot_upper_starting_pos)
        _, yaw, _ = self.get_pos_yaw_spe(self.robot_index)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = self.physicsClient)
        
    def set_robot_yaw(self, yaw = 0):
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
    
    def set_shoulder_speed(self, left_shoulder = 0, right_shoulder = 0):
        left_shoulder = relative_to(left_shoulder, -self.args.max_shoulder_speed, self.args.max_shoulder_speed)
        left_joint_index = get_joint_index(self.robot_index, 'body_left_shoulder_joint', physicsClient = self.physicsClient)
        p.setJointMotorControl2(self.robot_index, left_joint_index, controlMode = p.VELOCITY_CONTROL, targetVelocity = left_shoulder, physicsClientId=self.physicsClient)
        right_shoulder = relative_to(right_shoulder, -self.args.max_shoulder_speed, self.args.max_shoulder_speed)
        right_joint_index = get_joint_index(self.robot_index, 'body_right_shoulder_joint', physicsClient = self.physicsClient)
        p.setJointMotorControl2(self.robot_index, right_joint_index, controlMode = p.VELOCITY_CONTROL, targetVelocity = right_shoulder, physicsClientId=self.physicsClient)
        
    def set_shoulder_angle(self, left_shoulder = None, right_shoulder = None):
        if(left_shoulder == None):
            pass 
        else:
            limb_index = get_joint_index(self.robot_index, 'body_left_shoulder_joint', physicsClient = self.physicsClient)
            p.resetJointState(self.robot_index, limb_index, left_shoulder, physicsClientId=self.physicsClient)
        if(right_shoulder == None):
            pass 
        else:
            limb_index = get_joint_index(self.robot_index, 'body_right_shoulder_joint', physicsClient = self.physicsClient)
            p.resetJointState(self.robot_index, limb_index, right_shoulder, physicsClientId=self.physicsClient)
        
    def get_shoulder_angle(self):
        left_joint_index = get_joint_index(self.robot_index, 'body_left_shoulder_joint', physicsClient = self.physicsClient)
        left_joint_state = p.getJointState(self.robot_index, left_joint_index, physicsClientId=self.physicsClient)
        right_joint_index = get_joint_index(self.robot_index, 'body_right_shoulder_joint', physicsClient = self.physicsClient)
        right_joint_state = p.getJointState(self.robot_index, right_joint_index, physicsClientId=self.physicsClient)
        return left_joint_state[0], right_joint_state[0]
        
    def generate_object_positions(self, n = 2, r = 4):
        base_angle = uniform(0, 2 * pi)
        x1 = r * cos(base_angle)
        y1 = r * sin(base_angle)
        angle_step = (2 * pi) / n
        positions = [(x1, y1)]
        for i in range(1, n):
            current_angle = base_angle + (i * angle_step)
            x = r * cos(current_angle)
            y = r * sin(current_angle)
            positions.append((x, y))
        return positions
            
    def object_faces_up(self, object):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object.index, physicsClientId=self.physicsClient)
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_index, physicsClientId=self.physicsClient)
        delta_x = robot_pos[0] - obj_pos[0]
        delta_y = robot_pos[1] - obj_pos[1]
        angle_to_robot = math.atan2(delta_y, delta_x)
        (roll, pitch, _) = p.getEulerFromQuaternion(obj_orn, physicsClientId=self.physicsClient)
        new_orn = p.getQuaternionFromEuler([0, 0, angle_to_robot if not isnan(angle_to_robot) else 0])
        p.resetBasePositionAndOrientation(object.index, (obj_pos[0], obj_pos[1], object_upper_starting_pos), new_orn, physicsClientId=self.physicsClient)
            
    def touching_object(self, object):
        touching = {}
        for sensor_index, link_name in self.sensors:
            touching_this = bool(p.getContactPoints(
                bodyA=self.robot_index, bodyB=object.index, linkIndexA=sensor_index, physicsClientId = self.physicsClient))
            touching[link_name] = 1 if touching_this else 0
        return(touching)
    
    def touching_any_object(self):
        touching = {}
        for object in self.objects_in_play:
            touching_this_object = self.touching_object(object)
            touching[object] = touching_this_object
        return(touching)
    
    def object_positions(self):
        object_positions = []
        for object in self.objects_in_play:
            pos, _, _ = self.get_pos_yaw_spe(object.index)
            object_positions.append(pos)
        return(object_positions)
    
    def rewards(self, verbose = False):
        win = False
        reward = 0
        v_rx = cos(self.robot_start_yaw)
        v_ry = sin(self.robot_start_yaw)
        
        if(verbose):
            for object_key, object_dict in self.objects_touch.items():
                for link_name, value in object_dict.items():
                    if(value):
                        print(f"Touching {object_key} with {link_name}.")
                                                
        objects_goals = {}
                
        for i, object in enumerate(self.objects_in_play):
            watching = False 
            pushing = False 
            pulling = False 
            lefting = False 
            righting = False
            
            # Is the robot touching the object?
            touching = any(self.objects_touch[object].values())
            
            # Distance and angle from robot to object.
            object_pos, _ = p.getBasePositionAndOrientation(object.index, physicsClientId=self.physicsClient)
            robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot_index, physicsClientId = self.physicsClient)
            distance_vector = np.subtract(object_pos[:2], robot_pos[:2])
            distance = np.linalg.norm(distance_vector)
            normalized_distance_vector = distance_vector / distance
            rotation_matrix = p.getMatrixFromQuaternion(robot_ori, physicsClientId = self.physicsClient)
            forward_vector = np.array([rotation_matrix[0], rotation_matrix[3]])
            forward_vector /= np.linalg.norm(forward_vector)
            dot_product = np.dot(forward_vector, normalized_distance_vector)
            angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))  
            angle_degrees = degrees(angle_radians)
            cross_product = np.cross(np.append(forward_vector, 0), np.append(normalized_distance_vector, 0))
            if cross_product[2] < 0:  
                angle_radians = -angle_radians
                
            # How is the object moving in relation to the robot?
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
            
            # Is the robot watching an object?
            watching_now = abs(angle_radians) < pi/6 and not touching and distance <= self.args.watch_distance
            if watching_now: self.watching[object.index] += 1
            else:        self.watching[object.index] = 0 
            if(self.watching[object.index] >= self.args.watch_duration):
                watching = True
                        
            # Is the object pushed/pulled away from its starting position, relative to the robot's starting position and angle?
            if(movement_forward >= self.args.push_amount and touching):
                pushing = True
            if(movement_forward <= -self.args.pull_amount and touching and abs(angle_radians) < pi/2):
                pulling = True
                        
            # Is the object pushed left/right from its starting position, relative to the robot's starting position and angle?
            if(movement_left >= self.args.left_right_amount and touching):
                lefting = True
            if(movement_left <= -self.args.left_right_amount and touching):
                righting = True
                    
            objects_goals[(object.color, object.shape)] = [watching, pushing, pulling, lefting, righting]
        
        
                        
        mother_comm = None
        for (color, shape), (watching, pushing, pulling, lefting, righting) in objects_goals.items():
            #print(f"watching {watching}, pushing {pushing}, pulling {pulling}, lefting {lefting}, righting {righting}")
            num_actions_performed = sum([watching, pushing, pulling, lefting, righting]) 
            # If no action performed, don't bother with this object.
            if(num_actions_performed == 0):
                pass
            else:
                # If multiple actions performed, stop, it's a fail
                #if(num_actions_performed > 1):
                #    win = False 
                #    reward = 0
                #    break
                # If one action is occuring, find if it's the goal.
                if(watching): action = action_map[1]
                if(pushing):  action = action_map[2]
                if(pulling):  action = action_map[3]
                if(lefting):  action = action_map[4]
                if(righting): action = action_map[5]
                
                mother_comm = Goal(action, color, shape, self.goal.parenting)
                # If the goal, reward.
                if(action == self.goal.action and color == self.goal.color and shape == self.goal.shape):
                    win = True 
                    reward = self.args.reward
                ## If an action is occuring to the wrong object, stop, it's a fail.
                #else:
                #    win = False 
                #    reward = 0
                #    break
                
        if(self.goal.action.name == "FREEPLAY"):
            reward = 0
            win = None
        elif(self.args.silent_mother_outside_free_play):
            mother_comm = None
        if(self.args.silent_mother):
            mother_comm = None
                                        
        if(verbose):
            print(f"Mother comm: {mother_comm}")
            print("Raw reward:", round(reward, 2))
            print("Win:", win)
                        
        return(reward, win, mother_comm)
    
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
        proj_matrix = proj_matrix = p.computeProjectionMatrix(left, right, bottom, top, near, far)
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
        
        
    
if __name__ == "__main__":
    args = default_args
    physicsClient = get_physics(GUI = True, time_step = args.time_step, steps_per_step = args.steps_per_step)
    arena = Arena(physicsClient, args = args)
    sleep_time = 1
    
    def show_them():
        above_rgba = arena.photo_from_above()
        robot_rgbd = arena.photo_for_agent()
        plt.imshow(above_rgba)
        plt.show()
        plt.close()
        plt.imshow(robot_rgbd[:,:,:-1])
        plt.show()
        plt.close()
        
        
        
    action, colors_shapes_1, colors_shapes_2 = make_objects_and_action(1, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
        


    print(f"\n\n{action_map.keys()}\n\n")


    
    #"""
    print("\nWATCH")
    goal = Goal(action_map[1], colors_shapes_1[0][0], colors_shapes_1[0][1], True)    
    arena.begin(colors_shapes_1, goal, set_positions = [(6,0)])
    show_them()
    reward, win, mother_comm = arena.rewards(verbose = True)
    while(True):
        arena.step(0, 0, 1, 1, verbose = True, sleep_time = .1)
        show_them()
        reward, win, mother_comm = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
    
    #"""
    print("\nPUSH")
    goal = Goal(action_map[2], colors_shapes_1[0][0], colors_shapes_1[0][1], True)    
    arena.begin(colors_shapes_1, goal, set_positions = [(5,0)])
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(1, 1, 1, 1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, win, mother_comm = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
    
    #"""
    print("\nPULL")
    goal = Goal(action_map[3], colors_shapes_1[0][0], colors_shapes_1[0][1], True)    
    arena.begin(colors_shapes_1, goal, set_positions = [(3,0)])
    show_them()
    arena.rewards(verbose = True)
    arena.step(0, 0, -1, -1, verbose = True, sleep_time = sleep_time)
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(-1, -1, -1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, win, mother_comm = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
    
    #"""
    print("\nPULL BACKWARD")
    goal = Goal(action_map[3], colors_shapes_1[0][0], colors_shapes_1[0][1], True)
    arena.begin(colors_shapes_1, goal, set_positions = [(-3,0)])
    show_them()
    arena.rewards(verbose = True)
    for i in range(4):
        arena.step(-1, -1, -1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, win, mother_comm = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
    
    #"""   
    print("\nLEFT")
    goal = Goal(action_map[4], colors_shapes_1[0][0], colors_shapes_1[0][1], True)
    arena.begin(colors_shapes_1, goal, set_positions = [(3,0)])
    show_them()
    arena.rewards(verbose = True)
    arena.step(0, 0, -1, -1, verbose = True, sleep_time = sleep_time)
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(-1, 1, -1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, win, mother_comm = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
    
    """   
    print("\nLEFT AGAIN")
    goal = Goal(action_map[4], colors_shapes_1[0][0], colors_shapes_1[0][1], True)
    arena.begin(colors_shapes_1, goal, set_positions = [(2,4)])
    show_them()
    arena.rewards(verbose = True)
    arena.step(0, 0, -1, -1, verbose = True, sleep_time = sleep_time)
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(-1, 1, -1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, win, mother_comm = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    """
    
    #"""
    print("\nRIGHT")
    goal = Goal(action_map[5], colors_shapes_1[0][0], colors_shapes_1[0][1], True)
    arena.begin(colors_shapes_1, goal, set_positions = [(3,0)])
    show_them()
    arena.rewards(verbose = True)
    arena.step(0, 0, -1, -1, verbose = True, sleep_time = sleep_time)
    show_them()
    arena.rewards(verbose = True)
    while(True):
        arena.step(1, -1, -1, -1, verbose = True, sleep_time = sleep_time)
        show_them()
        reward, win, mother_comm = arena.rewards(verbose = True)
        if(win):
            break
    arena.end()
    #"""
# %%