#%%
import torch
import pybullet as p
import numpy as np
from time import sleep
from random import uniform, choice

from utils import default_args, action_map, shape_map, color_map, make_objects_and_action,\
    string_to_onehots, onehots_to_string, print, agent_to_english, comm_map
from submodule_utils import pad_zeros
from arena import Arena, get_physics



class Task:
    
    def __init__(
            self, 
            actions = [-1], 
            objects = 1, 
            colors = [0], 
            shapes = [0], 
            parenting = True, 
            args = default_args):
        self.actions = actions
        self.objects = objects 
        self.shapes = shapes
        self.colors = colors
        self.parenting = parenting
        self.args = args
        
        self.agent_to_english = agent_to_english
        
    def begin(self, test = False, verbose = False):
        self.solved = False
        self.goal = []
        self.current_objects_1 = []
            
        action_num, self.current_objects_1, self.current_objects_2 = make_objects_and_action(
            num_objects = self.objects, allowed_actions = self.actions, allowed_colors = self.colors, allowed_shapes = self.shapes, test = test)
        goal_color, goal_shape = self.current_objects_1[0]
        self.goal = (action_num, goal_color, goal_shape)
        
        self.goal_text = "{}{}{}".format(action_map[action_num][0], color_map[goal_color][0], shape_map[goal_shape][0])
        if(action_num == -1):
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
        self.goal_human_text += self.agent_to_english(self.goal_text) + "."
        self.goal_comm = string_to_onehots(self.goal_text)
        self.goal_comm = pad_zeros(self.goal_comm, self.args.max_comm_len)
                                
        if(verbose):
            print(self)
    
    def __str__(self):
        to_return = "\n\nSHAPE-COLORS (1):\t{}".format(["{} {}".format(list(color_map)[color], list(shape_map)[shape]) for color, shape in self.current_objects_1])
        if(not self.parenting):
            to_return += "\nSHAPE-COLORS (2):\t{}".format(["{} {}".format(list(color_map)[color], list(shape_map)[shape]) for color, shape in self.current_objects_2])
        to_return += "\nGOAL:\t{} ({})".format(onehots_to_string(self.goal_comm), self.goal_human_text)
        return(to_return)
    


class Task_Runner:
    
    def __init__(self, task, arena_1, arena_2, args = default_args):
        self.args = args
        self.task = task
        self.parenting = self.task.parenting
        self.arena_1 = arena_1 
        self.arena_2 = arena_2
        
    def begin(self, test = False, verbose = False):
        self.steps = 0
        self.task.begin(test, verbose)
        self.arena_1.begin(self.task.current_objects_1, self.task.goal, self.parenting)
        if(not self.parenting): self.arena_2.begin(self.task.current_objects_2, self.task.goal, self.parenting)
        
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
                
        return(rgbd, self.task.goal_comm.unsqueeze(0), sensors)
            
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
    
    def get_recommended_action(self, agent_1 = True, verbose = False):
        if(agent_1): arena = self.arena_1
        else:        
            if(self.parenting):
                return(None)
            else:
                arena = self.arena_2
        goal = arena.goal
        goal_action = goal[0]
        goal_color = goal[1]
        goal_shape = goal[2]
                
        distances = []
        angles = []
        shapes = []
        colors = []
        for i, ((shape, color, old_pos), object_index) in enumerate(arena.objects_in_play.items()):
            object_pos, _ = p.getBasePositionAndOrientation(object_index, physicsClientId=arena.physicsClient)
            agent_pos, agent_ori = p.getBasePositionAndOrientation(arena.robot_index)
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
            distances.append(distance)
            angles.append(angle_radians)
            shapes.append(shape)
            colors.append(color)
        
        relevant_distances_and_angles = [(distances[i], angles[i]) for i in range(len(distances)) if colors[i] == goal_color and shapes[i] == goal_shape]
        #relevant_distance, relevant_angle = min(relevant_distances_and_angles, key=lambda t: abs(t[1]))
        
        left_wheel = uniform(-1, 1)
        right_wheel = uniform(-1, 1)
        left_shoulder = 1 # uniform(-.1, .1)
        right_shoulder = 1 # uniform(-.1, .1)
        
        if(action_map[goal_action][1].upper() == "PUSH"):
            pass
                
        if(action_map[goal_action][1].upper() == "PULL"):
            pass
                
        if(action_map[goal_action][1].upper()== "LEFT"):
            pass
                
        if(action_map[goal_action][1].upper() == "RIGHT"):
            pass
        
        recommendation = torch.tensor([left_wheel, right_wheel, left_shoulder] + ([right_shoulder] if self.args.two_arms else [])).float()
                
        return(recommendation)
    
    
    
if __name__ == "__main__":        
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    args = default_args
    
    physicsClient = get_physics(GUI = True, time_step = args.time_step, steps_per_step = args.steps_per_step)
    arena_1 = Arena(physicsClient)
    arena_2 = None
    task_runner = Task_Runner(Task(actions = [2], objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0]), arena_1, arena_2)
    
    def get_images():
        rgba = task_runner.arena_1.photo_from_above()
        rgbd, _, _ = task_runner.obs()
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
        task_runner.begin(verbose = True)
        done = False
        j = 0
        while(done == False):
            j += 1 
            print("step", j)
            example_images(get_images())
            recommendation = task_runner.get_recommended_action(verbose = False)#True)
            print("Got recommendation:", recommendation)
            raw_reward, distance_reward, angle_reward, distance_reward_2, angle_reward_2, done, win, which_goal_message_1, which_goal_message_2 = task_runner.step(recommendation, verbose = True)
            print("Done:", done)
            rgbd, _, _ = task_runner.obs()
            rgb = rgbd[0,:,:,0:3]
            plt.imshow(rgb)
            plt.axis('off') 
            plt.show()
            plt.close()
            sleep(.1)
        print("Win:", win)
        example_images(get_images())
        task_runner.done()
# %%