#%%
import torch
import pybullet as p
import numpy as np
from time import sleep
from random import uniform, choice

from utils import default_args, task_map, shape_map, color_map, make_objects_and_task,\
    onehots_to_string, print, Goal, Obs, empty_goal
from utils_submodule import pad_zeros
from arena import Arena, get_physics



class Processor:
    
    def __init__(self, arena_1, arena_2, tasks = [0], objects = 1, colors = [0], shapes = [0], parenting = True, args = default_args):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
        
                
    def begin(self, test = False, verbose = False):
        self.steps = 0            
        goal_task, self.current_objects_1, self.current_objects_2 = make_objects_and_task(
            num_objects = self.objects, allowed_tasks = self.tasks, allowed_colors = self.colors, allowed_shapes = self.shapes, test = test)
        goal_color, goal_shape = self.current_objects_1[0]
        self.goal = Goal(goal_task, goal_color, goal_shape, self.parenting)
        self.arena_1.begin(self.current_objects_1, self.goal, self.parenting)
        if(not self.parenting):
            self.arena_2.begin(self.current_objects_2, self.goal, self.parenting)
                                
        if(verbose):
            print(self)
            
            
    
    def __str__(self):
        to_return = "\n\nSHAPE-COLORS (1):\t{}".format(["{} {}".format(color, shape) for color, shape in self.current_objects_1])
        if(not self.parenting):
            to_return += "\nSHAPE-COLORS (2):\t{}".format(["{} {}".format(color, shape) for color, shape in self.current_objects_2])
        to_return += "\nGOAL:\t{} ({})".format(self.goal.char_text, self.goal.human_text)
        return(to_return)
    
    
    
    def get_arena(self, agent_1 = True):
        if(agent_1): 
            arena = self.arena_1
        else:        
            if(self.parenting):
                return(None)
            else:
                arena = self.arena_2
        return(arena)
    
    
        
    def obs(self, agent_1 = True):
        arena = self.get_arena(agent_1)
        if(arena == None):
            return(torch.zeros((1, self.args.image_size, self.args.image_size, 4)), None, None)
                
        rgbd = arena.photo_for_agent()
        rgbd = torch.from_numpy(rgbd).float().unsqueeze(0)
        
        touched = [0] * self.args.sensors_shape
        for object_key, object_dict in arena.objects_touch.items(): 
            for i, (link_name, value) in enumerate(object_dict.items()):
                touched[i] += value
                
        sensors = torch.tensor([touched]).float()
                
        return(rgbd, sensors, self.goal.one_hots.unsqueeze(0))
    
    
            
    def act(self, wheels_shoulders, agent_1 = True, verbose = False, sleep_time = None):
        arena = self.get_arena(agent_1)
        
        left_wheel, right_wheel, left_shoulder, right_shoulder = \
            wheels_shoulders[0].item(), wheels_shoulders[1].item(), wheels_shoulders[2].item(), wheels_shoulders[3].item()
                  
        if(verbose): 
            print("\n\nStep {}:".format(self.steps))
            print("Wheels: {}, {}. Shoulders: {}, {}.".format(
            round(left_wheel, 2), round(right_wheel, 2), round(left_shoulder, 2), round(right_shoulder, 2)))
            
        arena.step(left_wheel, right_wheel, left_shoulder, right_shoulder, verbose = verbose, sleep_time = sleep_time)
        raw_reward, win, mother_comm = arena.rewards()
        return(raw_reward, win, mother_comm)
    
    
        
    def step(self, wheels_shoulders_1, wheels_shoulders_2 = None, verbose = False, sleep_time = None):
        self.steps += 1
        done = False
        
        raw_reward, win, mother_comm_1 = self.act(wheels_shoulders_1, verbose = verbose, sleep_time = sleep_time)
        if(self.parenting): 
            mother_comm_2 = empty_goal
        else:
            raw_reward_2, win_2, mother_comm_2 = self.act(wheels_shoulders_2, agent_1 = False, verbose = verbose, sleep_time = sleep_time)
            raw_reward = max([raw_reward, raw_reward_2])
            win = win or win_2
                    
        if(raw_reward > 0): 
            raw_reward *= self.args.step_cost ** (self.steps-1)
        end = self.steps >= self.args.max_steps
                                
        if(end and not win): 
            done = True
            goal_task = self.arena_1.goal.task
            if(goal_task.name != "FREEPLAY"):
                raw_reward += self.args.step_lim_punishment 
            if(verbose):
                print("Episode end!", end = " ")
        if(win):
            done = True
            if(verbose):
                print("Correct!", end = " ")
        if(verbose):
            print("Raw reward:", raw_reward)
            if(done): 
                print("Done.")
                                
        return(raw_reward, done, win, mother_comm_1, mother_comm_2)
    
    
    
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
            raw_reward, done, win, mother_comm_1, mother_comm_2 = processor.step(recommendation, verbose = True)
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