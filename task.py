#%%
from random import randint, choice, randrange
from math import cos, sin, pi, degrees
import torch
import pybullet as p
import numpy as np
from time import sleep

from utils import default_args, shape_map, color_map, action_map, make_object, pad_zeros,\
    string_to_onehots, onehots_to_string, print, relative_to, opposite_relative_to
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
        self.goal = (action_map[action].upper() if goal_action == None else goal_action.upper(), (shape, color))
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
                    None)
            else:
                arena = self.arena_2
        rgbd = arena.photo_for_agent()
        rgbd = torch.from_numpy(rgbd).float().unsqueeze(0)
        return(rgbd, self.task.goal_comm)
    
    def change_velocity(self, yaw_change, shoulder, arm, agent_1 = True, verbose = False):
        if(agent_1): arena = self.arena_1
        else:        arena = self.arena_2
        pos, yaw, spe = arena.get_pos_yaw_spe()
        
        old_yaw = yaw
        new_yaw = old_yaw + yaw_change
        new_yaw %= 2*pi
        arena.setBasePositionAndOrientation((pos[0], pos[1], 1), new_yaw)
        
        arena.setArmsAndHands(shoulder, arm)
                
        if(verbose):
            print("\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
                round(degrees(old_yaw)) % 360, round(degrees(yaw_change)), round(degrees(new_yaw))))
            #self.render(view = "body")  
            
    def step(self, action, agent_1 = True, verbose = False):
        if(agent_1): arena = self.arena_1
        else:        arena = self.arena_2
        yaw, shoulder, arm = \
            action[0].item(), action[1].item(), action[2].item()
        
        yaw = -yaw * self.args.max_yaw_change
        yaw = [-self.args.max_yaw_change, self.args.max_yaw_change, yaw] 
        yaw.sort() 
        yaw = yaw[1]
      
        if(verbose): 
            print("\n\nStep {}:".format(self.steps))
            print("Yaw: {}. Shoulders: {}. Arms: {}. Hands: {}.".format(
            round(degrees(yaw)), round(degrees(shoulder)), round(degrees(arm))))
        
        shoulder_before, arm_before = arena.get_arm_angles()
        shoulder_before = -opposite_relative_to(shoulder_before, self.args.min_shoulder, self.args.max_shoulder)
        arm_before = opposite_relative_to(arm_before, self.args.min_arm, self.args.max_arm)
        #print("\n\nSTART: {}, to {}.".format(shoulder_before, shoulder))
        for s in range(self.args.steps_per_step):
            portion = (s+1)/self.args.steps_per_step
            current_shoulder = (shoulder * portion) + (shoulder_before * (1 - portion))
            current_arm = (arm * portion) + (arm_before * (1 - portion))
            self.change_velocity(
                yaw/self.args.steps_per_step, 
                current_shoulder, 
                current_arm,  
                verbose = verbose if s == 0 else False)
            #print("{} + {} = {}".format(shoulder * portion, shoulder_before * (1 - portion), current_shoulder))
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
        else:        
            if(self.parenting):
                return(None)
            else:
                arena = self.arena_2
        goal = arena.goal
        goal_action = goal[0]
        goal_shape = list(shape_map)[goal[1][0]]
        goal_color = list(color_map.values())[goal[1][1]]
                
        distances = []
        hand_distances = []
        angles = []
        shapes = []
        colors = []
        for i, ((shape, color, old_pos), object_num) in enumerate(arena.objects_in_play.items()):
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
            distances.append(distance)
            angles.append(angle_radians)
            shapes.append(shape)
            colors.append(color)
        
        relevant_distances_and_angles = [(distances[i], angles[i]) for i in range(len(distances)) if shapes[i] == goal_shape and colors[i] == goal_color]
        relevant_distance, relevant_angle = min(relevant_distances_and_angles, key=lambda t: abs(t[1]))
                        
        if(verbose):
            print("\nDISTANCE:", relevant_distance, "ANGLE:", relevant_angle, "\n")

        relevant_angle /= self.args.max_yaw_change
        relevant_angle += (-.5 if goal_action == "LEFT" else .2 if goal_action == "RIGHT" else 0)
        
        # By default: Turn toward closest relevent object, shoulder up, arm halfway out. 
        # For 'watch,' that's all needed!
        yaw_change = relative_to(-relevant_angle, -1, 1)
        shoulder = 1
        arm = 0
        
        shoulder_before, arm_before = arena.get_arm_angles()
        shoulder_before = -opposite_relative_to(shoulder_before, self.args.min_shoulder, self.args.max_shoulder)
        arm_before = opposite_relative_to(arm_before, self.args.min_arm, self.args.max_arm)
        
        #print("shoulder before:", round(shoulder_before,2), "\tarm before:", round(arm_before,2))
        
        # If pointed at valid object, move arm based on goal.
        if(goal_action in ["PUSH", "PULL"] and abs(relevant_angle) < pi/8):
            push = goal_action == "PUSH"
            pull = goal_action == "PULL"
            arm = -1 if push else 1
            if((push and arm_before < -.8) or
                (pull and arm_before > .8)):
                shoulder = -1
            if(shoulder_before <= 0 and 
                ((push and arm_before < 0) or 
                (pull and arm_before > 0))):
                arm = arm_before + (.5 if push else -.5)
                shoulder = -1
                    
        if(goal_action in ["LEFT", "RIGHT"]):
            arm = 1
            if(abs(relevant_angle) < pi/8):
                left = goal_action == "LEFT"
                right = goal_action == "RIGHT"
                shoulder = -1
                if(shoulder_before <= 0):
                    yaw_change += -.2 if left else .2
            
        #print("shoulder after:", shoulder, "\tarm after:", arm)
                
        return(torch.tensor([
            yaw_change,
            shoulder,
            arm]).float())
    
    
    
if __name__ == "__main__":        
    import matplotlib.pyplot as plt
    args = default_args
    
    task_runner = Task_Runner(Task(actions = 5, objects = 2, shapes = 5, colors = 6), GUI = True)
    
    def get_images():
        rgba = task_runner.arena_1.photo_from_above()
        rgbd, _ = task_runner.obs()
        rgb = rgbd[0,:,:,0:3]
        return(rgba, rgb)
        
    def example_images(images):
        num_images = len(images)
        fig, axs = plt.subplots(2, num_images, figsize=(num_images * 5, 10), gridspec_kw={'wspace':0.1, 'hspace':0.1})
        for i, (rgba, rgb) in enumerate(images):
            if num_images > 1:
                ax1 = axs[0, i]
                ax2 = axs[1, i]
            else:
                ax1 = axs[0]
                ax2 = axs[1]
            ax1.imshow(rgba)
            ax1.axis('off') 
            ax2.imshow(rgb)
            ax2.axis('off') 
        plt.tight_layout()
        plt.show()

    while(True):
        images = []
        task_runner.begin(goal_action = "LEFT", verbose = True)
        done = False
        while(done == False):
            images.append(get_images())
            recommendation = task_runner.get_recommended_action(verbose = False)#True)
            reward, done, win = task_runner.action(recommendation, verbose = False)#True)
            sleep(.1)
        images.append(get_images())
        print("Win:", win)
        example_images(images)
        task_runner.done()
# %%