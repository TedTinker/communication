#%%
import torch
import pybullet as p
import numpy as np
from time import sleep
from random import uniform, choice

from utils import default_args, task_map, shape_map, color_map, make_objects_and_task,\
    string_to_onehots, onehots_to_string, print, agent_to_english, comm_map
from utils_submodule import pad_zeros
from arena import Arena, get_physics



class Processor:
    
    def __init__(
            self, 
            arena_1, arena_2,
            tasks = [-1], 
            objects = 1, 
            colors = [0], 
            shapes = [0], 
            parenting = True, 
            args = default_args):
        
        self.arena_1 = arena_1 
        self.arena_2 = arena_2
        self.tasks = tasks
        self.objects = objects 
        self.shapes = shapes
        self.colors = colors
        self.parenting = parenting
        self.args = args
                
    def begin(self, test = False, verbose = False):
        self.steps = 0
        self.solved = False
        self.goal = []
        self.current_objects_1 = []
            
        task_num, self.current_objects_1, self.current_objects_2 = make_objects_and_task(
            num_objects = self.objects, allowed_tasks = self.tasks, allowed_colors = self.colors, allowed_shapes = self.shapes, test = test)
        goal_color, goal_shape = self.current_objects_1[0]
        self.goal = (task_num, goal_color, goal_shape)
        
        self.goal_text = "{}{}{}".format(task_map[task_num][0], color_map[goal_color][0], shape_map[goal_shape][0])
        if(task_num == -1):
            self.goal_text = self.goal_text[0]            
        self.goal_human_text = "Given "
        for i, (c, s) in enumerate(self.current_objects_1):
            self.goal_human_text += color_map[c][1] + " " + shape_map[s][1]
            if i < len(self.current_objects_1) - 1:
                if len(self.current_objects_1) > 2: 
                    self.goal_human_text += ", "
                else:
                    self.goal_human_text += " "
            if i == len(self.current_objects_1) - 2: 
                self.goal_human_text += "and "
            elif i == len(self.current_objects_1) - 1:
                self.goal_human_text += ": "
        self.goal_human_text += agent_to_english(self.goal_text) + "."
        self.goal_comm = string_to_onehots(self.goal_text)
        self.goal_comm = pad_zeros(self.goal_comm, self.args.max_comm_len)
        
        self.arena_1.begin(self.current_objects_1, self.goal, self.parenting)
        if(not self.parenting): self.arena_2.begin(self.current_objects_2, self.goal, self.parenting)
                                
        if(verbose):
            print(self)
    
    def __str__(self):
        to_return = "\n\nSHAPE-COLORS (1):\t{}".format(["{} {}".format(list(color_map)[color], list(shape_map)[shape]) for color, shape in self.current_objects_1])
        if(not self.parenting):
            to_return += "\nSHAPE-COLORS (2):\t{}".format(["{} {}".format(list(color_map)[color], list(shape_map)[shape]) for color, shape in self.current_objects_2])
        to_return += "\nGOAL:\t{} ({})".format(onehots_to_string(self.goal_comm), self.goal_human_text)
        return(to_return)
    
    def obs(self, agent_1 = True):
        if(agent_1): arena = self.arena_1
        else:        
            if(self.parenting):
                return(torch.zeros((1, self.args.image_size, self.args.image_size, 4)), None, None)
            else:
                arena = self.arena_2
                
        rgbd = arena.photo_for_agent()
        rgbd = torch.from_numpy(rgbd).float().unsqueeze(0)
        
        touched = [0] * self.args.sensors_shape
        for object_key, object_dict in arena.objects_touch.items(): 
            for i, (link_name, value) in enumerate(object_dict.items()):
                touched[i] += value
                
        #_, _, speed = arena.get_pos_yaw_spe()
        #speed = opposite_relative_to(speed, self.args.min_speed, self.args.max_speed)
        sensors = torch.tensor([touched]).float()
                
        return(rgbd, sensors, self.goal_comm.unsqueeze(0))
            
    def act(self, action, agent_1 = True, verbose = False, sleep_time = None):
        if(agent_1): arena = self.arena_1
        else:        arena = self.arena_2
        left_wheel, right_wheel, left_shoulder = \
            action[0].item(), action[1].item(), action[2].item()
        if(self.args.two_arms):
            right_shoulder = action[3].item()
        else: 
            right_shoulder = None
                  
        if(verbose): 
            print("\n\nStep {}:".format(self.steps))
            print("Left Wheel: {}. Right Wheel: {}. Shoulders: {}, {}.".format(
            round(left_wheel, 2), round(right_wheel, 2), round(left_shoulder, 2), round(right_shoulder, 2)))
            
        arena.step(left_wheel, right_wheel, left_shoulder, right_shoulder, verbose = verbose, sleep_time = sleep_time)
        raw_reward, distance_reward, angle_reward, win, which_goal_message = arena.rewards()
        return(raw_reward, distance_reward, angle_reward, win, which_goal_message)
        
    def step(self, action_1, action_2 = None, verbose = False, sleep_time = None):
        self.steps += 1
        done = False
        
        raw_reward, distance_reward, angle_reward, win, which_goal_message_1 = self.act(action_1, verbose = verbose, sleep_time = sleep_time)
        if(self.parenting): 
            distance_reward_2 = 0
            angle_reward_2 = 0
            which_goal_message_2 = " " * self.args.max_comm_len
        else:
            raw_reward_2, distance_reward_2, angle_reward_2, win_2, which_goal_message_2 = self.act(action_2, agent_1 = False, verbose = verbose, sleep_time = sleep_time)
            raw_reward = max([raw_reward, raw_reward_2])
            win = win or win_2
                    
        if(raw_reward > 0): 
            raw_reward *= self.args.step_cost ** (self.steps-1)
        if(distance_reward > 0):
            distance_reward *= self.args.step_cost ** (self.steps-1)
        if(angle_reward > 0):
            angle_reward *= self.args.step_cost ** (self.steps-1)
        end = self.steps >= self.args.max_steps
                                
        if(end and not win): 
            done = True
            goal_task = self.arena_1.goal[0]
            if(task_map[goal_task][1].upper() != "FREE_PLAY"):
                raw_reward += self.args.step_lim_punishment # STOP THIS IN FREE PLAY!
            if(verbose):
                print("Episode end!", end = " ")
        if(win):
            done = True
            if(verbose):
                print("Correct!", end = " ")
        if(verbose):
            print("Raw reward:", raw_reward)
            print("Distance reward:", distance_reward)
            print("Angle reward:", angle_reward)
            if(done): 
                print("Done.")
                                
        return(raw_reward, distance_reward, angle_reward, distance_reward_2, angle_reward_2, done, win, which_goal_message_1, which_goal_message_2)
    
    def done(self):
        self.arena_1.end()
        if(not self.parenting):
            self.arena_2.end()
    
    
    
if __name__ == "__main__":        
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    args = default_args
    
    physicsClient = get_physics(GUI = True, time_step = args.time_step, steps_per_step = args.steps_per_step)
    arena_1 = Arena(physicsClient)
    arena_2 = None
    processor = Processor(arena_1, arena_2, tasks = [2], objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0], parenting = True, args = args)
    
    def get_images():
        rgba = processor.arena_1.photo_from_above()
        rgbd, _, _ = processor.obs()
        rgb = rgbd[0,:,:,0:3]
        d = rgbd[0,:,:,-1]
        return(rgba, rgb, d)
        
    def example_images(images):
        rgba, rgb, d = images
        
        plt.imshow(rgba)
        plt.axis('off') 
        plt.show()
        plt.close()

        plt.imshow(rgb)
        plt.axis('off') 
        plt.show()
        plt.close()
        
        plt.imshow(d, cmap='gray',interpolation='none')
        plt.axis('off') 
        plt.show()
        plt.close()
        
    i = 0
    while(True):
        i += 1
        
        print("episode", i)
        processor.begin(verbose = True)
        done = False
        j = 0
        while(done == False):
            j += 1 
            print("step", j)
            example_images(get_images())
            recommendation = torch.zeros((4,)) 
            print("Got recommendation:", recommendation)
            raw_reward, distance_reward, angle_reward, distance_reward_2, angle_reward_2, done, win, which_goal_message_1, which_goal_message_2 = processor.step(recommendation, verbose = True)
            print("Done:", done)
            rgbd, _, _ = processor.obs()
            rgb = rgbd[0,:,:,0:3]
            plt.imshow(rgb)
            plt.axis('off') 
            plt.show()
            plt.close()
            sleep(.1)
        print("Win:", win)
        example_images(get_images())
        processor.done()
# %%