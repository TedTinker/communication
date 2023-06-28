#%%
import os
import numpy as np
import pybullet as p
from math import pi, degrees, sin, cos
from time import sleep
from random import shuffle, choices

from utils import default_args, print, shapes, colors, goals
from arena import Arena



import torch
from torchvision.transforms.functional import resize

class Scenario:
    
    def __init__(self, num_objects = 1, num_agents = 1, many_goals = True, revealed_goals = False, GUI = False, args = default_args):
        self.num_objects = num_objects
        self.num_agents = num_agents
        self.many_goals = many_goals
        self.revealed_goals = revealed_goals
        self.GUI = GUI
        self.args = args
        self.begin()
        
    def begin(self):
        self.steps = 0 
        all_pairs = [(shape, color) for shape in shapes for color in colors]
        shuffle(all_pairs)
        if(self.many_goals):
            gs = [choices(goals)[0] for _ in all_pairs]
        else:
            gs = ["watch"] + ["none"]*(len(all_pairs)-1)
        all_pairs = [(shape, color, goal) for (shape, color), goal in zip(all_pairs, gs)]
        self.objects = all_pairs[:self.num_objects]
        
        self.arenas = []
        self.agent_poses = []
        self.agent_yaws = []
        self.agent_spes = []
        self.agent_comms = []
        self.new_agent_comms = []
        for i in range(self.num_agents):
            self.arenas.append(Arena(self.objects, self.GUI, self.args))
            self.arenas[-1].begin()
            pos, yaw, spe = self.arenas[-1].get_pos_yaw_spe()
            self.agent_poses.append(pos)
            self.agent_yaws.append(yaw)
            self.agent_spes.append(spe)
            self.agent_comms.append(torch.zeros((1,self.args.symbols)))
            self.new_agent_comms.append(torch.zeros((1,self.args.symbols)))
            
    def replace_comms(self):
        self.agent_comms = self.new_agent_comms
        
    def obs(self, i):
        arena = self.arenas[-1]
        pos, yaw, spe = self.agent_poses[i], self.agent_yaws[i], self.agent_spes[i]
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
        
        goal_communication = torch.zeros((1,len(shapes) + len(colors) + len(goals)))
        if(self.revealed_goals): 
            if(self.steps >= len(self.objects)): pass
            else:
                shape, color, goal =  self.objects[self.steps]
                shape_num = shapes.index(shape)
                color_num = colors.index(color)
                goal_num = goals.index(goal)
                goal_communication[0,shape_num] = 1
                goal_communication[0,len(shapes) + color_num] = 1
                goal_communication[0,len(shapes) + len(colors) + goal_num] = 1
        
        #if(self.arena.in_random() and self.args.randomness > 0):
        #    rgbd = torch.randint(2, size = rgbd.size(), dtype = rgbd.dtype)
        #    spe = torch.randint(2, size = spe.size(), dtype = spe.dtype)
        #    spe[spe == 0] = 0 ; spe[spe == 1] = self.args.max_speed
        if(self.num_agents == 1):
            comms = torch.zeros((1, self.args.symbols))
        else:
            comms = self.agent_comms[:i] + self.agent_comms[i+1:]
            comms = comms[0]
        return(rgbd, spe, comms, goal_communication)
    
    def change_velocity(self, i, yaw_change, speed, arms, hands, verbose = False):
        arena = self.arenas[i]
        pos, yaw, spe = self.agent_poses[i], self.agent_yaws[i], self.agent_spes[i]
        
        old_yaw = yaw
        new_yaw = old_yaw + yaw_change
        new_yaw %= 2*pi
        arena.resetBasePositionAndOrientation((pos[0], pos[1], 1), new_yaw)
        
        old_speed = spe
        x = cos(new_yaw)*speed
        y = sin(new_yaw)*speed
        arena.resetBaseVelocity(x, y)
        _, self.agent_yaws[i], _ = arena.get_pos_yaw_spe()
        
        arena.resetArmsAndHands(arms, hands)
                
        if(verbose):
            print("\n\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
                round(degrees(old_yaw)) % 360, round(degrees(yaw_change)), round(degrees(new_yaw))))
            print("Old speed:\t{}\nNew speed:\t{}".format(old_speed, speed))
            #self.render(view = "body")  
            print("\n")
        
    def action(self, i, action, verbose = True):
        arena = self.arenas[i]
        self.steps += 1
        
        yaw, spe, arms, hands, comm = action[0], action[1], action[2], action[3], action[4:]
        
        if(verbose): print("\n\nStep {}: yaw {}, spe {}.".format(self.steps, yaw, spe))
        yaw = -yaw * self.args.max_yaw_change
        yaw = [-self.args.max_yaw_change, self.args.max_yaw_change, yaw] ; yaw.sort() ; yaw = yaw[1]
        spe = self.args.min_speed + ((spe + 1)/2) * \
            (self.args.max_speed - self.args.min_speed)
        spe = [self.args.min_speed, self.args.max_speed, spe] ; spe.sort() ; spe = spe[1]
        arms *= pi ; hands *= pi
        if(verbose): print("updated: yaw {}, spe {}, arms {}, hands {}.".format(yaw, spe, arms, hands))
        action_name = "Yaw: {}. Speed: {}. Arms: {}. Hands: {}.".format(round(degrees(yaw)), round(spe), round(degrees(arms)), round(degrees(hands)))
        
        for _ in range(self.args.steps_per_step):
            self.change_velocity(i, yaw/self.args.steps_per_step, spe/self.args.steps_per_step, arms, hands, verbose = verbose)
            p.stepSimulation(physicsClientId = arena.physicsClient)
            self.agent_poses[i], self.agent_yaws[i], self.agent_spes[i] = arena.get_pos_yaw_spe()
            self.new_agent_comms[i] = torch.tensor(comm).unsqueeze(0)
            
        pos, yaw, spe = self.agent_poses[i], self.agent_yaws[i], self.agent_spes[i]
        if(verbose): print("agent: pos {}, yaw {}, spe {}.".format(pos, yaw, spe))
        
        reward, end = arena.rewards()
        if(reward > 0): reward *= self.args.step_cost ** self.steps
        if(verbose): print("end {}, reward {}".format(end, reward))
        
        if(not end): end = self.steps >= self.args.max_steps
        if(self.steps >= self.args.max_steps): 
            failures = 0
            for (shape, color, goal), object in arena.objects.items():
                if(goal != "none"): failures += 1
            reward += self.args.step_lim_punishment * failures
        if(verbose): print("end {}, reward {}\n\n".format(end, reward))

        return(reward, end, action_name)
    
    
    
if __name__ == "__main__":        
    from random import random
    from time import sleep
    import matplotlib.pyplot as plt

    default_args.randomness = 1
    scenario = Scenario(3, 1, goals, True, default_args)
    done = False
    yaws = [0] * 20
    speeds = [-1] * 20
    arms = [i/10 for i in range(-10,10)]
    hands = [i/10 for i in range(-10,10)]
    i = 0
    while(done == False):
        if(yaws == []):
            print("Stopped!")
            break
        #reward, done, action_name = scenario.action(0, random(), random(), random(), random(), verbose = True)
        reward, done, action_name = scenario.action(0, (yaws[i], speeds[i], arms[i], hands[i], "hi"), verbose = True)
        rgbd, spe, comms, goal_comm = scenario.obs(0)
        rgbd = rgbd.squeeze(0)[:,:,0:3]
        plt.imshow(rgbd)
        plt.show()
        plt.close()
        sleep(.1)
        i += 1
        if(i >= len(yaws)): i = 0
# %%
