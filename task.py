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
        
    def begin(self, test = False, verbose = False):
        self.solved = False
        self.goal = []
        self.current_objects_1 = []
        for i in range(self.objects):
            self.make_object(test = test)
        
        index = self.make_goal()

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
    
    def make_goal(self):
        action = randint(0, self.actions - 1)
        index = randrange(len(self.current_objects_1))
        shape, color = self.current_objects_1[index]
        self.goal = (action_map[action], (shape, color))
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
        
    def begin(self, test = False, verbose = False):
        self.steps = 0 
        self.task.begin(test, verbose)
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
    
    def change_velocity(self, yaw_change, speed, right_arm, right_hand, left_arm, left_hand, agent_1 = True, verbose = False):
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
        
        arena.setArmsAndHands(right_arm, right_hand, left_arm, left_hand)
                
        if(verbose):
            print("\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
                round(degrees(old_yaw)) % 360, round(degrees(yaw_change)), round(degrees(new_yaw))))
            print("Old speed:\t{}\nNew speed:\t{}".format(old_speed, speed))
            #self.render(view = "body")  
            
    def step(self, action, agent_1 = True, verbose = False):
        if(agent_1): arena = self.arena_1
        else:        arena = self.arena_2
        yaw, spe, right_arm, right_hand, left_arm, left_hand = \
            action[0], action[1], action[2], action[3], action[4], action[5]
        
        if(verbose): print("\n\nStep {}:.".format(self.steps))
        yaw = -yaw * self.args.max_yaw_change
        yaw = [-self.args.max_yaw_change, self.args.max_yaw_change, yaw] ; yaw.sort() ; yaw = yaw[1]
        spe = self.args.min_speed + ((spe + 1)/2) * \
            (self.args.max_speed - self.args.min_speed)
        spe = [self.args.min_speed, self.args.max_speed, spe] ; spe.sort() ; spe = spe[1]
        right_arm *= pi ; right_hand *= pi
        left_arm *= pi ;  left_hand *= pi
        if(verbose): print("Yaw: {}. Speed: {}. Arms: {}. Hands: {}.".format(
            round(degrees(yaw)), round(spe), (round(degrees(right_arm)), round(degrees(left_arm))), (round(degrees(right_hand), round(degrees(left_hand))))))
        
        for s in range(self.args.steps_per_step):
            self.change_velocity(yaw/self.args.steps_per_step, spe, right_arm, right_hand, left_arm, left_hand, verbose = verbose if s == 0 else False)
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
    
    def get_recommended_action(self, agent_1 = True):
        # Find angle between agent and the first correct object.
        # If angle is large, just turn.
        # If angle is small...
        #   If action is watch, wait.
        #   Otherwise, find distance.
        #   If distance is large, move closer.
        #   If distance is small...
        #       If action is touch, move closer.
        #       If action is pull, find arm angle.
        #           If arms are close together, separate them.
        #           If arms are far apart, move closer.
        #           Then bring arms together.
        #           Then move backward.
        # ETC.
        return(torch.tensor([0,0,0,0,0,0]))
    
    
    
if __name__ == "__main__":        
    from time import sleep
    args = default_args

    task_runner = Task_Runner(Task(actions = 5, objects = 3, shapes = 5, colors = 6))
    task_runner.begin(verbose = True)
    done = False
    while(done == False):
        action = [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)]
        reward, done, win = task_runner.action(action, verbose = True)
        sleep(.1)
# %%