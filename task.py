#%%
from random import randint, choice, randrange
from math import cos, sin, pi, degrees
import torch
import pybullet as p
import numpy as np

from utils import default_args, shape_map, color_map, action_map, make_object, pad_zeros,\
    string_to_onehots, onehots_to_string, print
from arena import Arena



class Task:
    
    def __init__(
            self, 
            actions = 1, 
            objects = 1, 
            shapes = 1, 
            colors = 1, 
            parent = True, 
            args = default_args):
        self.actions = actions
        self.objects = objects 
        self.shapes = shapes
        self.colors = colors
        self.parent = parent
        self.args = args
        
    def begin(self, goal_action = None, test = False, verbose = False):
        self.solved = False
        self.goal = []
        self.current_objects_1 = []
        for i in range(self.objects):
            self.make_object(test = test)
        
        index = self.make_goal(goal_action)

        if(not self.parent):
            self.current_objects_2 = []
            for i in range(self.objects):
                self.make_object(test = test, agent_1 = False)
            new_index = randrange(len(self.current_objects_2))
            self.current_objects_2[new_index] = self.current_objects_1[index]
                
        if(verbose):
            print(self)
        
    def make_object(self, test = False, agent_1 = True):
        shape, color = make_object(self.shapes, self.colors, test)
        if(agent_1):
            self.current_objects_1.append((shape, color))
        else:
            self.current_objects_2.append((shape, color))
    
    def make_goal(self, goal_action = None):
        action = randint(0, self.actions - 1)
        index = randrange(len(self.current_objects_1))
        shape, color = self.current_objects_1[index]
        self.goal = (action_map[action] if goal_action == None else goal_action, (shape, color))
        self.goal_text = "{} {} {}.".format(self.goal[0], list(color_map)[color], list(shape_map)[shape])
        self.goal_comm = string_to_onehots(self.goal_text)
        self.goal_comm = pad_zeros(self.goal_comm, self.args.max_comm_len)
        return(index)
    
    def __str__(self):
        to_return = "\n\nSHAPE-COLORS (1):\t{}".format(["{} {}".format(list(color_map)[color], list(shape_map)[shape]) for shape, color in self.current_objects_1])
        if(not self.parent):
            to_return += "\nSHAPE-COLORS (2):\t{}".format(["{} {}".format(list(color_map)[color], list(shape_map)[shape]) for shape, color in self.current_objects_2])
        to_return += "\nGOAL:\t{}".format(onehots_to_string(self.goal_comm))
        return(to_return)



class Task_Runner:
    
    def __init__(self, task, GUI = False, args = default_args):
        self.args = args
        self.task = task
        self.parenting = self.task.parent
        self.arena_1 = Arena(GUI = GUI, args = args)
        if(not self.parenting): self.arena_2 = Arena(args = args)
        
    def begin(self, goal_action = None, test = False, verbose = False):
        self.steps = 0 
        self.task.begin(goal_action, test, verbose)
        self.arena_1.begin(self.task.current_objects_1, self.task.goal)
        if(not self.parenting): self.arena_2.begin(self.task.current_objects_2, self.task.goal)
        
    def obs(self, agent_1 = True):
        if(agent_1): arena = self.arena_1
        else:        
            if(self.parenting):
                return(
                    torch.zeros((1, self.args.image_size, self.args.image_size, 4)),
                    torch.zeros((1, 1)),
                    None)
            else:
                arena = self.arena_2
        pos, yaw, spe = arena.get_pos_yaw_spe()
        x, y = cos(yaw), sin(yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [pos[0] + x, pos[1] + y, 2], 
            cameraTargetPosition = [pos[0] + x*2, pos[1] + y*2, 2],    # Camera / target position very important
            cameraUpVector = [0, 0, 1], physicsClientId = arena.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 10, physicsClientId = arena.physicsClient)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=self.args.image_size, height=self.args.image_size,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = arena.physicsClient)
        
        rgb = np.divide(rgba[:,:,:-1], 255)
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
        if(d.max() == d.min()): pass
        else: d = (d.max() - d)/(d.max()-d.min())
        rgbd = np.concatenate([rgb, d], axis = -1)
        rgbd = torch.from_numpy(rgbd).float().unsqueeze(0)
        spe = torch.tensor(spe).unsqueeze(0).unsqueeze(0)
        
        return(rgbd, spe, self.task.goal_comm)
    
    def change_velocity(self, yaw_change, speed, right_shoulder, right_arm, right_hand, left_shoulder, left_arm, left_hand, agent_1 = True, verbose = False):
        if(agent_1): arena = self.arena_1
        else:        arena = self.arena_2
        pos, yaw, spe = arena.get_pos_yaw_spe()
        
        old_yaw = yaw
        new_yaw = old_yaw + yaw_change
        new_yaw %= 2*pi
        arena.setBasePositionAndOrientation((pos[0], pos[1], 1), new_yaw)
        
        old_speed = spe
        x = cos(new_yaw)*speed
        y = sin(new_yaw)*speed
        arena.setBaseVelocity(x, y)
        
        arena.setArmsAndHands(right_shoulder, right_arm, right_hand, left_shoulder, left_arm, left_hand)
                
        if(verbose):
            print("\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
                round(degrees(old_yaw)) % 360, round(degrees(yaw_change)), round(degrees(new_yaw))))
            print("Old speed:\t{}\nNew speed:\t{}".format(old_speed, speed))
            #self.render(view = "body")  
            
    def step(self, action, agent_1 = True, verbose = False):
        if(agent_1): arena = self.arena_1
        else:        arena = self.arena_2
        yaw, spe, right_shoulder, right_arm, right_hand, left_shoulder, left_arm, left_hand = \
            action[0].item(), action[1].item(), action[2].item(), action[3].item(), action[4].item(), action[5].item(), action[6].item(), action[7].item()
        
        if(verbose): print("\n\nStep {}:.".format(self.steps))
        yaw = -yaw * self.args.max_yaw_change
        yaw = [-self.args.max_yaw_change, self.args.max_yaw_change, yaw] ; yaw.sort() ; yaw = yaw[1]
        spe = self.args.min_speed + ((spe + 1)/2) * \
            (self.args.max_speed - self.args.min_speed)
        spe = [self.args.min_speed, self.args.max_speed, spe] ; spe.sort() ; spe = spe[1]
        right_shoulder *= pi
        right_arm *= pi
        right_hand *= pi
        left_shoulder *= pi
        left_arm *= pi 
        left_hand *= pi
        if(verbose): print("Yaw: {}. Speed: {}. Shoulders: {}. Arms: {}. Hands: {}.".format(
            round(degrees(yaw)), round(spe), (round(degrees(right_shoulder)), round(degrees(left_shoulder))), (round(degrees(right_arm)), round(degrees(left_arm))), (round(degrees(right_hand)), round(degrees(left_hand)))))
        
        for s in range(self.args.steps_per_step):
            self.change_velocity(yaw/self.args.steps_per_step, spe, right_shoulder, right_arm, right_hand, left_shoulder, left_arm, left_hand, verbose = verbose if s == 0 else False)
            arena.step()
        
        reward, win = arena.rewards()
        return(reward, win)
        
    def action(self, action_1, action_2 = None, verbose = False):
        self.steps += 1
        done = False
        
        reward, win = self.step(action_1, verbose = verbose)
        if(not self.parenting): 
            reward_2, win_2 = self.step(action_2, agent_1 = False, verbose = verbose)
            reward = max([reward, reward_2])
            win = win or win_2
                    
        if(reward > 0): 
            reward *= self.args.step_cost ** (self.steps-1)
        end = self.steps >= self.args.max_steps
        if(end and not win): 
            reward += self.args.step_lim_punishment
            done = True
        if(win):
            done = True
            if(verbose):
                print("Correct!", end = " ")
        if(verbose):
            print("Reward:", reward)
            if(done): 
                print("Done.")
        return(reward, done, win)
    
    def done(self):
        self.arena_1.end()
        if(not self.parenting):
            self.arena_2.end()
    
    def get_recommended_action(self, agent_1 = True, verbose = False):
        if(agent_1): arena = self.arena_1
        else:        arena = self.arena_2
        goal = arena.goal
        goal_action = goal[0]
        goal_shape = list(shape_map)[goal[1][0]]
        goal_color = list(color_map.values())[goal[1][1]]
        distance_angles = []
        for i, ((shape, color, old_pos), object_num) in enumerate(arena.objects_in_play.items()):
            if(shape != goal_shape or color != goal_color): pass 
            else:
                object_pos, _ = p.getBasePositionAndOrientation(object_num, physicsClientId=arena.physicsClient)
                agent_pos, agent_ori = p.getBasePositionAndOrientation(arena.body_num)
                distance_vector = np.subtract(object_pos, agent_pos)
                distance = np.linalg.norm(distance_vector)
                normalized_distance_vector = distance_vector / distance
                rotation_matrix = p.getMatrixFromQuaternion(agent_ori)
                forward_vector = np.array([rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]])
                forward_vector /= np.linalg.norm(forward_vector)

                dot_product = np.dot(forward_vector, normalized_distance_vector)
                angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))  
                cross_product = np.cross(forward_vector, normalized_distance_vector)
                if cross_product[2] < 0:  
                    angle_radians = -angle_radians
                distance_angles.append((distance, angle_radians))
        distance, angle = min(distance_angles, key=lambda t: abs(t[1]))
        
        if(verbose):
            print(distance, angle)

        angle /= self.args.max_yaw_change
        yaw_change = [-1, 1, -angle]
        yaw_change.sort() 
        yaw_change = yaw_change[1]
        
        speed = 0
        right_shoulder = 0 if goal_action.upper() == "TOUCH" else -.5
        right_arm = 0
        right_hand = 0
        left_shoulder = 0 if goal_action.upper() == "TOUCH" else -.5
        left_arm = 0
        left_hand = 0
        
        if(abs(angle) < pi/8):
            if(goal_action.upper() == "WATCH"):
                if(distance < self.args.watch_distance):
                    speed = -1
            elif(goal_action.upper() == "TOUCH"):
                speed = 1
            else:
                if(distance > 3):
                    speed = .5
                else:
                    if(goal_action.upper() == "LIFT"):
                        pass
                    if(goal_action.upper() == "PULL"):
                        pass
                    if(goal_action.upper() == "SPIN"):
                        pass
                
        return(torch.tensor([
            yaw_change,
            speed,
            right_shoulder,
            right_arm,
            right_hand,
            left_shoulder,
            left_arm,
            left_hand]).float())
    
    
    
if __name__ == "__main__":        
    from time import sleep
    import matplotlib.pyplot as plt
    args = default_args

    task_runner = Task_Runner(Task(actions = 2, objects = 2, shapes = 2, colors = 2), GUI = True)
    task_runner.begin(goal_action = "WATCH", verbose = True)
    done = False
    while(done == False):
        rgba = task_runner.arena_1.photo_from_above()
        plt.imshow(rgba)
        plt.show()
        plt.close()
        recommendation = task_runner.get_recommended_action(verbose = True)
        action = torch.tensor([.3, 1, 1, 1, 1, 1, 1, 1]) 
        
        reward, done, win = task_runner.action(recommendation, verbose = True)
        sleep(.1)
# %%