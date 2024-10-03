#%%
import torch
from collections import namedtuple

from utils import default_args, make_objects_and_task, print, Goal, goal_to_onehots, string_to_onehots, Whole_Obs, Action
from arena import Arena, get_physics



Step_Results = namedtuple('Step_Results', [
    "reward", "done", "win", 
    "whole_obs_1", "whole_obs_2"])



class Processor:
    
    def __init__(self, arena_1, arena_2, tasks = [0, 1, 2, 3, 4, 5], colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, num_objects = 2, args = default_args): 
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
    def begin(self, test = False):
        self.steps = 0
        task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(
            self.num_objects, self.tasks, self.colors, self.shapes, test = test)
        self.goal = Goal(task, colors_shapes_1[0][0], colors_shapes_1[0][1], self.parenting)   
        self.arena_1.begin(colors_shapes_1, self.goal)
        if(not self.parenting): 
            self.arena_2.begin(colors_shapes_2, self.goal)
        
    def choose_arena(self, agent_1 = True):
        if(agent_1): arena = self.arena_1
        else:        
            if(self.parenting): return(None)
            else:               arena = self.arena_2
        return(arena)
    
    def obs(self, agent_1 = True):
        arena = self.choose_arena(agent_1)
        if(arena == None): return(None, None, None)
        rgbd = arena.photo_for_agent()
        rgbd = torch.from_numpy(rgbd).float().unsqueeze(0)
        touched = [0] * self.args.sensors_shape
        for object, object_dict in arena.objects_touch.items(): 
            for i, (link_name, value) in enumerate(object_dict.items()):
                touched[i] += value
        sensors = torch.tensor([touched]).float()
        father_comm = goal_to_onehots(self.goal)
        reward, win, mother_comm = arena.rewards()
        if(mother_comm == None):
            mother_comm = string_to_onehots("AAA")
        else:
            mother_comm = goal_to_onehots(mother_comm)
                            
        whole_obs = Whole_Obs(rgbd, sensors, father_comm, mother_comm)
        return(whole_obs, reward, win)
    
    def act(self, action, agent_1 = True, verbose = False, sleep_time = None):
        arena = self.choose_arena(agent_1)
        if(arena == None): return
        
        wheels_shoulders = action.wheels_shoulders
        while(len(wheels_shoulders.shape) != 1):
            wheels_shoulders = wheels_shoulders.squeeze(0)
            
        left_wheel, right_wheel, left_shoulder, right_shoulder = \
            wheels_shoulders[0].item(), wheels_shoulders[1].item(), wheels_shoulders[2].item(), wheels_shoulders[3].item() 
            
        arena.step(left_wheel, right_wheel, left_shoulder, right_shoulder, verbose = verbose, sleep_time = sleep_time)
        if(verbose): 
            print("\n\nStep {}, agent {}:".format(self.steps, 1 if agent_1 else 2))
            print("Left Wheel: {}. Right Wheel: {}. Left Shoulder: {}. Right Shoulder: {}.".format(
            round(left_wheel, 2), round(right_wheel, 2), round(left_shoulder, 2), round(right_shoulder, 2)))
    
    def step(self, action_1, action_2, verbose = False, sleep_time = None):
        self.steps += 1
        done = False
                
        self.act(action_1, verbose = verbose, sleep_time = sleep_time)
        whole_obs_1, reward_1, win_1 = self.obs()
        if(self.parenting): 
            whole_obs_2 = None
            reward_2 = reward_1
            win_2 = win_1
        else:
            self.act(action_2, agent_1 = False, verbose = verbose, sleep_time = sleep_time)
            whole_obs_2, reward_2, win_2 = self.obs(agent_1 = False)
        reward = max([reward_1, reward_2])
        win = win_1 or win_2
        if(reward > 0): 
            reward *= self.args.step_cost ** (self.steps-1)
            
        end = self.steps >= self.args.max_steps     
        if(end and not win): 
            done = True
            reward += self.args.step_lim_punishment 
            if(verbose):
                print("Episode failed!", end = " ")
        if(win):
            done = True
            if(verbose):
                print("Episode succeeded!", end = " ")
        if(verbose):
            print("Raw reward:", reward)
            if(done): 
                print("Done.")
                            
        return(Step_Results(
            reward, done, win, 
            whole_obs_1, whole_obs_2))
    
    def done(self):
        self.arena_1.end()
        if(not self.parenting):
            self.arena_2.end()
    
    
    
if __name__ == "__main__":
    args = default_args
    physicsClient_1 = get_physics(GUI = False, time_step = args.time_step, steps_per_step = args.steps_per_step)
    physicsClient_2 = get_physics(GUI = True, time_step = args.time_step, steps_per_step = args.steps_per_step)
    processor = Processor(Arena(physicsClient_1), Arena(physicsClient_2), parenting = False)
    processor.begin()
    for i in range(10):
        action_1 = Action(torch.rand(4), torch.rand(args.max_comm_len, args.comm_shape))
        action_2 = Action(torch.rand(4), torch.rand(args.max_comm_len, args.comm_shape))
        step_results = processor.step(action_1, action_2, verbose = True, sleep_time = 1)
    processor.done()