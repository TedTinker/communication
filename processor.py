#%%
import torch
import pybullet as p
import numpy as np
from time import sleep
from random import uniform, choice

from utils import task_map, shape_map, color_map, make_objects_and_task, print, Goal, Obs, empty_goal
from utils_submodule import pad_zeros
from arena import Arena, get_physics



class Processor:
    
    def __init__(self, args, arena_1, arena_2, tasks_and_weights = [(0, 1)], objects = 1, colors = [0], shapes = [0], parenting = True, linestyle = '-', full_name = ""):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
        
                
    def begin(self, test = False, verbose = False):
        self.steps = 0            
        goal_task, self.current_objects_1, self.current_objects_2 = make_objects_and_task(
            num_objects = self.objects, allowed_tasks_and_weights = self.tasks_and_weights, allowed_colors = self.colors, allowed_shapes = self.shapes, test = test)
        goal_color, goal_shape = self.current_objects_1[0]
        if(goal_task.name == "FREEPLAY"):
            goal_color = goal_task
            goal_shape = goal_task
        self.goal = Goal(goal_task, goal_color, goal_shape, self.parenting)
        self.arena_1.begin(self.current_objects_1, self.goal, self.parenting)
        if(not self.parenting):
            self.arena_2.begin(self.current_objects_2, self.goal, self.parenting)
        self.mother_voice_1 = empty_goal
        self.mother_voice_2 = empty_goal
                                
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
            return(Obs(torch.zeros((1, self.args.image_size, self.args.image_size, 4)), None, None, None))
                
        rgbd = arena.photo_for_agent()
        rgbd = torch.from_numpy(rgbd).float().unsqueeze(0)
        
        touched = [0] * self.args.sensors_shape
        for object_key, object_dict in arena.objects_touch.items(): 
            for i, (link_name, value) in enumerate(object_dict.items()):
                touched[i] += value
                
        sensors = torch.tensor([touched]).float()
        
        mother_voice = self.mother_voice_1 if agent_1 else self.mother_voice_2
                        
        return(Obs(rgbd, sensors, self.goal, mother_voice))
    
    
            
    def act(self, wheels_joints, agent_1 = True, verbose = False, sleep_time = None):
        arena = self.get_arena(agent_1)
        if(arena == None):
            return(None, None)
        
        left_wheel_speed, right_wheel_speed, joint_1_speed, joint_2_speed = \
            wheels_joints[0].item(), wheels_joints[1].item(), wheels_joints[2].item(), wheels_joints[3].item() if self.args.robot_name.startswith("two") else 0
                  
        if(verbose): 
            print("\n\nStep {}:".format(self.steps))
            print("Wheels: {}, {}. Joints: {}, {}.".format(
            round(left_wheel_speed, 2), round(right_wheel_speed, 2), round(joint_1_speed, 2), round(joint_2_speed, 2)))
            
        arena.step(left_wheel_speed, right_wheel_speed, joint_1_speed, joint_2_speed, verbose = verbose, sleep_time = sleep_time)
        reward, win, mother_voice = arena.rewards()
        if(agent_1): 
            self.mother_voice_1 = mother_voice
        else:
            self.mother_voice_2 = mother_voice
        return(reward, win)
    
    
        
    def step(self, wheels_joints_1, wheels_joints_2 = None, verbose = False, sleep_time = None):
        self.steps += 1
        done = False
        
        reward, win = self.act(wheels_joints_1, verbose = verbose, sleep_time = sleep_time)
        if(not self.parenting): 
            reward_2, win_2 = self.act(wheels_joints_2, agent_1 = False, verbose = verbose, sleep_time = sleep_time)
            reward = max([reward, reward_2])
            win = win or win_2
                    
        if(reward > 0): 
            reward *= self.args.step_cost ** (self.steps-1)
        end = self.steps >= self.args.max_steps
                                
        if(end and not win): 
            done = True
            goal_task = self.arena_1.goal.task
            if(goal_task.name != "FREEPLAY"):
                reward += self.args.step_lim_punishment 
            if(verbose):
                print("Episode end!", end = " ")
        if(win):
            done = True
            if(verbose):
                print("Correct!", end = " ")
        if(verbose):
            print("Raw reward:", reward)
            if(done): 
                print("Done.")
                                
        return(reward, done, win)
    
    
    
    def done(self):
        self.arena_1.end()
        if(not self.parenting):
            self.arena_2.end()
            
        self.mother_voice_1 = empty_goal
        self.mother_voice_2 = empty_goal
    
    
    
if __name__ == "__main__":        
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    from utils import args
    
    physicsClient = get_physics(GUI = True, args = args)
    arena_1 = Arena(args, physicsClient)
    arena_2 = None
    processor = Processor(args, arena_1, arena_2, tasks_and_weights = [(0, 1)], objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0], parenting = True)
    
    def get_images():
        rgba = processor.arena_1.photo_from_above()
        obs = processor.obs()
        rgb = obs.rgbd[0,:,:,0:3]
        d = obs.rgbd[0,:,:,-1]
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
            reward, done, win = processor.step(recommendation, verbose = True)
            print("Done:", done)
            obs = processor.obs()
            rgb = obs.rgbd[0,:,:,0:3]
            plt.imshow(rgb)
            plt.axis('off') 
            plt.show()
            plt.close()
            sleep(.1)
        print("Win:", win)
        example_images(get_images())
        processor.done()
# %%