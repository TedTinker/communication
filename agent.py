#%%

import os 
import psutil
from time import sleep
import numpy as np
from math import log
from itertools import accumulate, product
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

from utils import default_args, onehots_to_string, wheels_shoulders_to_string, cpu_memory_usage, \
    task_map, color_map, shape_map, task_name_list, print, To_Push, empty_goal, rolling_average, Obs, Action
from utils_submodule import model_start
from arena import Arena, get_physics
from processor import Processor
from buffer import RecurrentReplayBuffer
from pvrnn import PVRNN
from models import Actor, Critic
from plotting_episodes import plot_step



def print_cpu_usage(string, num):
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_affinity = process.cpu_affinity()
    print(f"{string}: {num} CPU affinity: {cpu_affinity}")
    current_cpu = psutil.cpu_percent(interval=1, percpu=True)
    print(f"{string}: {num} Current CPU usage per core: {current_cpu}")



class Agent:
    
    def __init__(self, i = -1, GUI = False, args = default_args):
        
        self.agent_num = i
        self.args = args
        self.total_steps = 0
        self.total_episodes = 0
        self.total_epochs = 0
        
        if self.args.device.type == "cuda":
            print(f"\nIN AGENT: {i} DEVICE: {self.args.device} ({torch.cuda.current_device()} out of {[j for j in range(torch.cuda.device_count())]}, {torch.cuda.get_device_name(torch.cuda.current_device())})\n")
        else:
            print(f"\nIN AGENT: {i} DEVICE: {self.args.device}\n")
            
        self.start_physics(GUI) 
        
        #os.sched_setaffinity(0, {self.args.cpu})
        
        self.processors = {
            "fp" :          Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 1)],                                             objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, args = self.args,
                                      full_name = "Free Play"),
            "w" :           Processor(self.arena_1, self.arena_2, tasks_and_weights = [(1, 1)],                                             objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, args = self.args,
                                      full_name = "Watch"),
            "wpulr" :       Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, args = self.args,
                                      full_name = "All Tasks"),
            
            "wpulr_1" :     Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (1)"),
            "wpulr_2" :     Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (2)"),
            "wpulr_3" :     Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 3), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (3)"),
            "wpulr_4" :     Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 4), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (4)"),
            "wpulr_5" :     Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 5), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (5)"),
            "wpulr_6" :     Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 6), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (6)"),
            "wpulr_7" :     Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 7), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (7)"),
            "wpulr_8" :     Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 8), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (8)"),
            "wpulr_9" :     Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 9), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],     objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (9)"),
            "wpulr_10" :    Processor(self.arena_1, self.arena_2, tasks_and_weights = [(0, 10), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],    objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, linestyle = '--', args = self.args,
                                      full_name = "All Tasks (10)"),
            
            
            
            }
        
        self.all_processors = {f"{task_map[task].name}_{color_map[color].name}_{shape_map[shape].name}" : 
            Processor(self.arena_1, self.arena_2, tasks_and_weights = [(task, 1)], objects = 1, colors = [color], shapes = [shape], parenting = True, args = self.args) for task, color, shape in \
                product([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4])}
                #product([0, 1], [0, 1], [0, 1])}
        all_processor_names = list(self.all_processors.keys())
        self.all_processor_names = all_processor_names
        
        self.target_entropy = self.args.target_entropy
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=self.args.lr, weight_decay = self.args.weight_decay) 
        if(self.args.half):
            self.log_alpha = self.log_alpha.to(dtype=torch.float16)
        
        self.target_entropy_text = self.args.target_entropy_text
        self.alpha_text = 1
        self.log_alpha_text = torch.tensor([0.0], requires_grad=True)
        self.alpha_text_opt = optim.Adam(params=[self.log_alpha_text], lr=self.args.lr, weight_decay = self.args.weight_decay) 
        if(self.args.half):
            self.log_alpha_text = self.log_alpha_text.to(dtype=torch.float16)

        self.forward = PVRNN(self.args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=self.args.lr, weight_decay = self.args.weight_decay)
                           
        self.actor = Actor(self.args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.args.lr, weight_decay = self.args.weight_decay) 
        
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []
        for _ in range(self.args.critics):
            self.critics.append(Critic(self.args))
            self.critic_targets.append(Critic(self.args))
            self.critic_targets[-1].load_state_dict(self.critics[-1].state_dict())
            self.critic_opts.append(optim.Adam(self.critics[-1].parameters(), lr=self.args.lr, weight_decay = self.args.weight_decay))
            
        all_params = list(self.forward.parameters())
        all_params += list(self.actor.parameters())
        for critic in self.critics:
            all_params += list(critic.parameters())
        self.complete_opt = optim.Adam(all_params, lr=self.args.lr, weight_decay=self.args.weight_decay)       
        
        self.memory = RecurrentReplayBuffer(self.args)
        self.old_memories = []
        
        self.plot_dict = {
            "args" : self.args,
            "arg_title" : args.arg_title,
            "arg_name" : args.arg_name,
            "division_epochs" : [],
            "episode_dicts" : {}, 
            "all_processor_names" : self.all_processor_names,
            "component_data" : {},
            "agent_lists" : {} if (self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list) else {"forward" : PVRNN(self.args), "actor" : Actor(self.args), "critic" : Critic(self.args)},
            "reward" : [], 
            "gen_reward" : [], 
            "steps" : [],
            "accuracy" : [], 
            "rgbd_loss" : [], 
            "sensors_loss" : [], 
            "father_comm_loss" : [], 
            "complexity" : [],
            "alpha" : [], 
            "alpha_text" : [],
            "actor" : [], 
            "critics" : [[] for _ in range(self.args.critics)], 
            "extrinsic" : [], 
            "q" : [], 
            "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            "intrinsic_imitation" : [],
            "rgbd_prediction_error_curiosity" : [], 
            "sensors_prediction_error_curiosity" : [], 
            "father_comm_prediction_error_curiosity" : [], 
            "prediction_error_curiosity" : [], 
            "rgbd_hidden_state_curiosity" : [],
            "sensors_hidden_state_curiosity" : [],
            "father_comm_hidden_state_curiosity" : [],
            "hidden_state_curiosity" : [],
            "behavior" : {},
            "wins_all" : [],
            "gen_wins_all" : []}
        for t in task_map.values():
            self.plot_dict[f"wins_{t.name}"] = []
            self.plot_dict[f"gen_wins_{t.name}"] = []
            
            
            
    def start_physics(self, GUI = False):
        self.steps = 0
        self.episodes = 0 
        self.epochs = 0 
        physicsClient_1 = get_physics(GUI = GUI, time_step = self.args.time_step, steps_per_step = self.args.steps_per_step)
        self.arena_1 = Arena(physicsClient_1, args = self.args)
        physicsClient_2 = get_physics(GUI = False, time_step = self.args.time_step, steps_per_step = self.args.steps_per_step)
        self.arena_2 = Arena(physicsClient_2, args = self.args)
        self.processor_name = self.args.processor_list[0]
        
    def give_actor_voice(self):
        self.actor.comm_out.load_state_dict(self.forward.predict_obs.comm_out.state_dict())
        
        
        
    def regular_checks(self, force = False, swapping = False):
        if(force):
            self.gen_test()
            self.get_component_data()
            self.save_episodes(swapping = swapping)
            self.save_agent()
            return
        if(self.epochs % self.args.epochs_per_gen_test == 0):
            self.gen_test()  
        else:
            self.plot_dict["gen_wins_all"].append(None)
            win_dict_list = [self.plot_dict["gen_wins_" + task.name] for task in task_map.values()]
            for i, win_dict in enumerate(win_dict_list):
                win_dict.append(None)
        if(self.epochs % self.args.epochs_per_component_data == 0):
            self.get_component_data()  
        if(self.epochs % self.args.epochs_per_episode_dict == 0):
            self.save_episodes(swapping = swapping)
        if(self.epochs % self.args.epochs_per_agent_list == 0):
            self.save_agent()
        
        
        
    def training(self, q = None):      
        self.regular_checks(force = True)
        while(True):
            cumulative_epochs = 0
            for i, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs == cumulative_epochs):                     
                    self.regular_checks(force = True)
                    linestyle = self.processors[self.processor_name].linestyle
                    full_name = self.processors[self.processor_name].full_name
                    self.processor_name = self.args.processor_list[i+1] 
                    self.old_memories.append(deepcopy(self.memory))
                    self.plot_dict["division_epochs"].append((self.total_epochs, linestyle, full_name))
                    self.regular_checks(force = True, swapping = True)
            step = self.training_episode()

            percent_done = str(self.epochs / sum(self.args.epochs))
            if(q != None):
                q.put((self.agent_num, percent_done))
            if(self.epochs >= sum(self.args.epochs)): 
                linestyle = self.processors[self.processor_name].linestyle
                full_name = self.processors[self.processor_name].full_name
                self.plot_dict["division_epochs"].append((self.total_epochs, linestyle, full_name))
                break
                        
            self.regular_checks()
        
        self.regular_checks(force = True)
        
        self.plot_dict["accumulated_reward"] = list(accumulate(self.plot_dict["reward"]))
        self.plot_dict["accumulated_gen_reward"] = list(accumulate(self.plot_dict["gen_reward"]))
        
        for task_name in task_name_list + ["all"]:
            self.plot_dict["rolled_wins_" + task_name] = rolling_average(self.plot_dict["wins_" + task_name])
            self.plot_dict["rolled_gen_wins_" + task_name] = rolling_average(self.plot_dict["gen_wins_" + task_name])
            
        self.min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in self.min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "all_processor_names", "component_data", "episode_dicts", "agent_lists", "spot_names", "steps", "behavior"]):
                if(key == "hidden_state"):
                    min_maxes = []
                    hidden_state = deepcopy(self.plot_dict[key])
                    for l in hidden_state:
                        minimum = None ; maximum = None 
                        l = [_ for _ in l if _ != None]
                        if(l != []):
                            if(  minimum == None):  minimum = min(l)
                            elif(minimum > min(l)): minimum = min(l)
                            if(  maximum == None):  maximum = max(l) 
                            elif(maximum < max(l)): maximum = max(l)
                        min_maxes.append((minimum, maximum))
                    self.min_max_dict[key] = min_maxes
                else:
                    minimum = None ; maximum = None 
                    l = self.plot_dict[key]
                    l = deepcopy(l)
                    l = [_ for _ in l if _ != None]
                    if(l != []):
                        if(  minimum == None):  minimum = min(l)
                        elif(minimum > min(l)): minimum = min(l)
                        if(  maximum == None):  maximum = max(l) 
                        elif(maximum < max(l)): maximum = max(l)
                    self.min_max_dict[key] = (minimum, maximum)
                
    
    
    def step_in_episode(self, 
                        prev_action_1, hq_1, mother_comm_1,
                        prev_action_2, hq_2, mother_comm_2, sleep_time = None):

        with torch.no_grad():
            self.eval()
            parenting = self.processor.parenting
                        
                        
                        
            def agent_step(agent_1 = True):
                
                if(parenting and not agent_1):
                    return(None, None, hq_2, hq_2, None, None, None, None, )
                
                prev_action = prev_action_1 if agent_1 else prev_action_2
                partner_prev_comm_out = prev_action_2.comm_out if (agent_1 and not parenting) else torch.zeros((1, 1, self.args.max_comm_len, self.args.comm_shape)) if agent_1 else prev_action_1.comm_out
                hq = hq_1 if agent_1 else hq_2
                mother_comm = mother_comm_1 if agent_1 else mother_comm_2
                obs = self.processor.obs(agent_1)
                
                comm_in = mother_comm.one_hots.unsqueeze(0).unsqueeze(0) if self.processor.goal.task.name == "FREEPLAY" else obs.father_comm.one_hots.unsqueeze(0).unsqueeze(0) if parenting else partner_prev_comm_out
                obs.father_comm = comm_in
                hp, hq, rgbd_is, sensors_is, father_comm_is = self.forward.bottom_to_top_step(
                    hq_1, self.forward.obs_in(obs), self.forward.action_in(prev_action))

                action, _, _ = self.actor(hq.detach(), parenting) 
                values = []
                for i in range(self.args.critics):
                    value = self.critics[i](action, hq.detach()) 
                    values.append(round(value.item(), 3))
                
                return(obs, action, hp, hq, values, rgbd_is, sensors_is, father_comm_is)
            
            obs_1, action_1, hp_1, hq_1, values_1, rgbd_is_1, sensors_is_1, father_comm_is_1 = agent_step()
            obs_2, action_2, hp_2, hq_2, values_2, rgbd_is_2, sensors_is_2, father_comm_is_2 = agent_step(agent_1 = False)

            reward, done, win, mother_comm_1, mother_comm_2 = self.processor.step(action_1.wheels_shoulders[0,0].clone(), None if action_2 == None else action_2.wheels_shoulders[0,0].clone(), sleep_time = sleep_time)
            
            def next_agent_step(agent_1 = True):
                
                if(parenting and not agent_1):
                    return(None, None)
                
                next_obs = self.processor.obs(agent_1)
                mother_comm = mother_comm_1 if agent_1 else mother_comm_2 
                obs = obs_1 if agent_1 else obs_2 
                action = action_1 if agent_1 else action_2
                partner_comm_out = None if parenting else action_2.comm_out if agent_1 else action_1.comm_out
                next_comm_in = mother_comm.one_hots.unsqueeze(0).unsqueeze(0) if self.processor.goal.task.name == "FREEPLAY" else next_obs.father_comm.one_hots.unsqueeze(0).unsqueeze(0) if parenting else partner_comm_out
                next_obs.father_comm = next_comm_in
                to_push = To_Push(obs, action, reward, next_obs, done)     
                return(next_obs, to_push)
            
            next_obs_1, to_push_1 = next_agent_step()
            next_obs_2, to_push_2 = next_agent_step(agent_1 = False)
            
        torch.cuda.empty_cache()
        
        return(action_1, values_1, hp_1.squeeze(1), hq_1.squeeze(1), rgbd_is_1, sensors_is_1, father_comm_is_1, mother_comm_1,
               action_2, values_2, hp_2.squeeze(1), hq_2.squeeze(1), rgbd_is_2, sensors_is_2, father_comm_is_2, mother_comm_2,
               reward, done, win, to_push_1, to_push_2)
            
           
           
    def start_episode(self):
        done = False
        complete_reward = 0
        steps = 0
        
        def start_agent(agent_1 = True):
            to_push_list = []
            prev_action = Action(torch.zeros((1, 1, self.args.wheels_shoulders_shape)), torch.zeros((1, 1, self.args.max_comm_len, self.args.comm_shape)))
            hq = torch.zeros((1, 1, self.args.pvrnn_mtrnn_size)) 
            mother_comm = empty_goal
            return(to_push_list, prev_action, hq, mother_comm)
                
        return(done, complete_reward, steps, start_agent(), start_agent(agent_1 = False))
           
           
    
    def training_episode(self):        
        done, complete_reward, steps, \
            (to_push_list_1, prev_action_1, hq_1, mother_comm_1), \
            (to_push_list_2, prev_action_2, hq_2, mother_comm_2) = self.start_episode()
                    
        self.episodes += 1 
        self.total_episodes += 1
        self.plot_dict["behavior"][self.episodes] = []
        self.processor = self.processors[self.processor_name]
        self.processor.begin()    
                        
        for step in range(self.args.max_steps):
            self.steps += 1                           
            self.total_steps += 1                                                                  
            if(not done):
                steps += 1
                prev_action_1, values_1, hp_1, hq_1, rgbd_is_1, sensors_is_1, father_comm_is_1, mother_comm_1, \
                    prev_action_2, values_2, hp_2, hq_2, rgbd_is_2, sensors_is_2, father_comm_is_2, mother_comm_2, \
                        reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                            prev_action_1, hq_1, mother_comm_1,
                            prev_action_2, hq_2, mother_comm_2)
                        
                if(self.args.agents_per_behavior_analysis == -1 or self.agent_num <= self.args.agents_per_behavior_analysis):  
                    self.plot_dict["behavior"][self.episodes].append(mother_comm_1)
                    
                to_push_list_1.append(to_push_1)
                to_push_list_2.append(to_push_2)
                complete_reward += reward
            if(self.steps % self.args.steps_per_epoch == 0):
                self.epoch(self.args.batch_size)
                                                    
        self.processor.done()
        self.plot_dict["steps"].append(steps)
        self.plot_dict["reward"].append(complete_reward)
        goal_task = self.processor.goal.task.name
        self.plot_dict["wins_all"].append(win)
        for task_name in task_name_list:
            if(task_name == goal_task): 
                self.plot_dict["wins_" + task_name].append(win)
            else:                
                self.plot_dict["wins_" + task_name].append(None)
                             
        for to_push in to_push_list_1:
            to_push.push(self.memory)
            
        for to_push in to_push_list_2:
            if(to_push != None):
                to_push.push(self.memory)
                        
        return(step)
        
        
        
    def gen_test(self):
        done, complete_reward, steps, \
            (to_push_list_1, prev_action_1, hq_1, mother_comm_1), \
            (to_push_list_2, prev_action_2, hq_2, mother_comm_2) = self.start_episode()
                
        try:
            self.processor = self.processors[self.processor_name]
            self.processor.begin(test = True)        
            for step in range(self.args.max_steps):
                #print("Step", step)
                if(not done):
                    prev_action_1, values_1, hp_1, hq_1, rgbd_is_1, sensors_is_1, father_comm_is_1, mother_comm_1, \
                        prev_action_2, values_2, hp_2, hq_2, rgbd_is_2, sensors_is_2, father_comm_is_2, mother_comm_2, \
                            reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                                prev_action_1, hq_1, mother_comm_1,
                                prev_action_2, hq_2, mother_comm_2)
                    complete_reward += reward
                #print("DONE")
            self.processor.done()
            goal_task = self.processor.goal.task.name
            self.plot_dict["gen_wins_all"].append(win)
            for task_name in task_name_list:
                if(task_name == goal_task): 
                    self.plot_dict["gen_wins_" + task_name].append(win)
                else:                       
                    self.plot_dict["gen_wins_" + task_name].append(None)
        except:
            complete_reward = 0
            win = False
            self.plot_dict["gen_wins_all"].append(None)
            for task_name in task_name_list:
                self.plot_dict["gen_wins_" + task_name].append(win)
        self.plot_dict["gen_reward"].append(complete_reward)
        return(win)
        
        
        
    def save_episodes(self, swapping = False, test = False, sleep_time = None, for_display = False):        
        with torch.no_grad():
            self.processor = self.processors[self.processor_name]
            self.processor.begin(test = test)       
            parenting = self.processor.parenting
            if(self.args.agents_per_episode_dict != -1 and self.agent_num > self.args.agents_per_episode_dict): 
                return
            for episode_num in range(self.args.episodes_in_episode_dict):
                common_keys = [
                    "rgbd", "sensors", "father_comm", "wheels_shoulders", "wheels_shoulders_text", "comm_out", 
                    "birds_eye", "reward", "critic_predictions", "prior_predicted_rgbd", 
                    "prior_predicted_sensors", "prior_predicted_father_comm", "posterior_predicted_rgbd", 
                    "posterior_predicted_sensors", "posterior_predicted_father_comm", 
                    "rgbd_dkl", "sensors_dkl", "father_comm_dkl", "mother_comm"]
                episode_dict = {}
                for agent_id in [0, 1]:
                    for key in common_keys:
                        episode_dict[f"{key}_{agent_id}"] = []
                episode_dict["reward"] = []
                episode_dict["processor"] = self.processor
                episode_dict["goal"] = "'{}' ({})".format(self.processor.goal.char_text, self.processor.goal.human_text)
                
                done = False
                
                done, complete_reward, steps, \
                    (to_push_list_1, prev_action_1, hq_1, mother_comm_1), \
                    (to_push_list_2, prev_action_2, hq_2, mother_comm_2) = self.start_episode()
                        
                hp_1 = torch.zeros((1, 1, self.args.pvrnn_mtrnn_size)) 
                hq_1 = torch.zeros((1, 1, self.args.pvrnn_mtrnn_size)) 

                hp_2 = torch.zeros((1, 1, self.args.pvrnn_mtrnn_size)) 
                hq_2 = torch.zeros((1, 1, self.args.pvrnn_mtrnn_size)) 
                                
                def save_step(step, hp, hq, wheels_shoulders, agent_1 = True):
                    agent_num = 1 if agent_1 else 2
                    
                    birds_eye = self.processor.arena_1.photo_from_above() if agent_1 else self.processor.arena_2.photo_from_above()
                    obs = self.processor.obs(agent_1 = agent_1)
                    
                    episode_dict[f"birds_eye_{agent_num}"].append(birds_eye[:,:,0:3])
                    episode_dict[f"rgbd_{agent_num}"].append(obs.rgbd[0,:,:,0:3])        
                    episode_dict[f"sensors_{agent_num}"].append(obs.sensors.tolist()[0])
                    
                    if(agent_1):
                        comm_in = mother_comm_1 if self.processor.goal.task.name == "FREEPLAY" else obs.father_comm.char_text if parenting else prev_action_2.comm_out[0,0]
                    else:
                        comm_in = mother_comm_2 if self.processor.goal.task.name == "FREEPLAY" else obs.father_comm.char_text if parenting else prev_action_1.comm_out[0,0]
                    episode_dict[f"father_comm_{agent_num}"].append("'{}' ({})".format(comm_in, comm_in))
                    
                    if(agent_1):
                        mother_comm = mother_comm_1
                    else:
                        mother_comm = mother_comm_2
                    episode_dict[f"mother_comm_{agent_num}"].append(mother_comm)
                    
                    if(step != 0):
                        
                        pred_obs_p = self.forward.predict(hp.unsqueeze(1), self.forward.wheels_shoulders_in(wheels_shoulders)) 
                        pred_obs_q= self.forward.predict(hq.unsqueeze(1), self.forward.wheels_shoulders_in(wheels_shoulders))

                        episode_dict[f"prior_predicted_rgbd_{agent_num}"].append(pred_obs_p.rgbd[0,0][:,:,0:3])
                        episode_dict[f"prior_predicted_sensors_{agent_num}"].append([round(o.item(), 2) for o in pred_obs_p.sensors[0,0]])
                        prior_predicted_father_comm = onehots_to_string(pred_obs_p.father_comm[0,0])
                        episode_dict[f"prior_predicted_father_comm_{agent_num}"].append("'{}' ({})".format(prior_predicted_father_comm, prior_predicted_father_comm))
                        
                        episode_dict[f"posterior_predicted_rgbd_{agent_num}"].append(pred_obs_q.rgbd[0,0][:,:,0:3])
                        episode_dict[f"posterior_predicted_sensors_{agent_num}"].append([round(o.item(), 2) for o in pred_obs_q.sensors[0,0]])
                        posterior_predicted_father_comm = onehots_to_string(pred_obs_q.father_comm[0,0])
                        episode_dict[f"posterior_predicted_father_comm_{agent_num}"].append("'{}' ({})".format(posterior_predicted_father_comm, posterior_predicted_father_comm))
                    
                def display(step, agent_1 = True, done = False, stopping = False, wait = True):
                    if(for_display):
                        print(f"\n{self.processor.goal.human_text}", end = " ")
                        print("STEP:", step)
                        plot_step(step, episode_dict, agent_1 = agent_1, last_step = done, saving = False)
                        if(not self.processor.parenting and not stopping):
                            display(step, agent_1 = False, stopping = True)
                        if(wait):
                            WAITING = input("WAITING")
                
                for step in range(self.args.max_steps + 1):
                    save_step(step, hp_1, hq_1, wheels_shoulders = prev_action_1.wheels_shoulders, agent_1 = True)    
                    if(not self.processor.parenting):
                        save_step(step, hp_2, hq_2, wheels_shoulders = prev_action_2.wheels_shoulders, agent_1 = False)  
                        
                    display(step)
                    
                    prev_action_1, values_1, hp_1, hq_1, rgbd_is_1, sensors_is_1, father_comm_is_1, mother_comm_1, \
                        prev_action_2, values_2, hp_2, hq_2, rgbd_is_2, sensors_is_2, father_comm_is_2, mother_comm_2, \
                            reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                                prev_action_1, hq_1, mother_comm_1,
                                prev_action_2, hq_2, mother_comm_2, sleep_time) 
                            
                    episode_dict["reward"].append(str(round(reward, 3)))
                    
                    episode_dict["wheels_shoulders_1"].append(prev_action_1.wheels_shoulders)
                    episode_dict["wheels_shoulders_text_1"].append(wheels_shoulders_to_string(prev_action_1.wheels_shoulders))
                    comm_out_1 = onehots_to_string(prev_action_1.comm_out)
                    episode_dict["comm_out_1"].append("{} ({})".format(prev_action_1.comm_out, prev_action_1.comm_out))
                    episode_dict["rgbd_dkl_1"].append(rgbd_is_1.dkl.sum().item())
                    episode_dict["sensors_dkl_1"].append(sensors_is_1.dkl.sum().item())
                    episode_dict["father_comm_dkl_1"].append(father_comm_is_1.dkl.sum().item())
                    episode_dict["critic_predictions_1"].append(values_1)
                    episode_dict["reward_1"].append(str(round(reward, 3)))
                        
                    if(not self.processor.parenting):
                        episode_dict["wheels_shoulders_2"].append(prev_action_2.wheels_shoulders)
                        episode_dict["wheels_shoulders_text_2"].append(wheels_shoulders_to_string(prev_action_2.wheels_shoulders))
                        comm_out_2 = onehots_to_string(prev_action_2.comm_out)
                        episode_dict["comm_out_2"].append("{} ({})".format(prev_action_2.comm_out, prev_action_2.comm_out))
                        episode_dict["rgbd_dkl_2"].append(rgbd_is_2.dkl.sum().item())
                        episode_dict["sensors_dkl_2"].append(sensors_is_2.dkl.sum().item())
                        episode_dict["father_comm_dkl_2"].append(father_comm_is_2.dkl.sum().item())
                        episode_dict["critic_predictions_2"].append(values_2)
                        episode_dict["reward_2"].append(str(round(reward_2, 3)))
                    
                    if(done):
                        save_step(step, hp_1, hq_1, wheels_shoulders = prev_action_1.wheels_shoulders, agent_1 = True)    
                        if(not self.processor.parenting):
                            save_step(step, hp_2, hq_2, wheels_shoulders = prev_action_2.wheels_shoulders, agent_1 = False) 
                        display(step + 1, done = True, wait = False)
                        self.processor.done()
                        break
                
                if(for_display):
                    return(win)
                else:
                    self.plot_dict["episode_dicts"]["{}_{}_{}_{}".format(self.agent_num, self.epochs, episode_num, 1 if swapping else 0)] = episode_dict
                    
                    
                    
    def get_component_data(self):
        if(self.args.agents_per_component_data != -1 and self.agent_num > self.args.agents_per_component_data): 
            return
        adjusted_args = deepcopy(self.args)
        adjusted_args.capacity = len(self.all_processors)
        temp_memory = RecurrentReplayBuffer(adjusted_args)
        processor_lens = []
        for processor_name in self.all_processor_names:
            self.processor = self.all_processors[processor_name]
            #print(self.processor.processor)
            self.processor.begin(test = None)    
            done, complete_reward, steps, \
                (to_push_list_1, prev_action_1, hq_1, mother_comm_1), \
                (to_push_list_2, prev_action_2, hq_2, mother_comm_2) = self.start_episode()
                     
            for step in range(self.args.max_steps):
                #print("Step", step)
                if(not done):
                    prev_action_1, values_1, hp_1, hq_1, rgbd_is_1, sensors_is_1, father_comm_is_1, mother_comm_1, \
                        prev_action_2, values_2, hp_2, hq_2, rgbd_is_2, sensors_is_2, father_comm_is_2, mother_comm_2, \
                            reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                                prev_action_1, hq_1, mother_comm_1,
                                prev_action_2, hq_2, mother_comm_2)
                to_push_list_1.append(to_push_1)
                if(done): break
            #print("DONE")
            self.processor.done()
            processor_lens.append(step)           
            for to_push in to_push_list_1:
                to_push.push(temp_memory)
                
        batch = self.get_batch(temp_memory, len(self.all_processors), random_sample = False)
        rgbd, sensors, father_comm, mother_comm, wheels_shoulders, comm_out, reward, done, mask, all_mask, episodes, steps = batch
        
        hps, hqs, rgbd_is, sensors_is, father_comm_is, pred_obs_q, labels = self.forward(
            torch.zeros((episodes, 1, self.args.pvrnn_mtrnn_size)), 
            Obs(rgbd, sensors, father_comm, father_comm), Action(wheels_shoulders, comm_out))
        
        father_comm_zq = father_comm_is.zq.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        all_mask = all_mask.detach().cpu().numpy()   
        
        non_zero_mask = labels[:, 0, 0] != 0  # This checks if the first element of each sequence is not 0
        father_comm_zq_filtered = father_comm_zq[non_zero_mask]
        labels_filtered = labels[non_zero_mask]
        all_mask_filtered = all_mask[non_zero_mask]
                
        self.plot_dict["component_data"][self.epochs] = (father_comm_zq, labels, all_mask, father_comm_zq_filtered, labels_filtered, all_mask_filtered)
                        
        
        
    def save_agent(self):
        if(self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list): return
        self.plot_dict["agent_lists"]["{}_{}".format(self.agent_num, self.epochs)] = deepcopy(self.state_dict())
        
        
        
    def get_batch(self, memory, batch_size, random_sample = True):
        batch = memory.sample(batch_size, random_sample = random_sample)
        if(batch == False): return(False)
        
        rgbd, sensors, father_comm, mother_comm, wheels_shoulders, comm_out, reward, done, mask = batch
        rgbd = torch.from_numpy(rgbd).to(self.args.device)
        sensors = torch.from_numpy(sensors).to(self.args.device)
        father_comm = torch.from_numpy(father_comm).to(self.args.device)
        mother_comm = torch.from_numpy(mother_comm).to(self.args.device)
        wheels_shoulders = torch.from_numpy(wheels_shoulders)
        comm_out = torch.from_numpy(comm_out)
        reward = torch.from_numpy(reward).to(self.args.device)
        done = torch.from_numpy(done).to(self.args.device)
        mask = torch.from_numpy(mask)
        wheels_shoulders = torch.cat([torch.zeros(wheels_shoulders[:,0].unsqueeze(1).shape), wheels_shoulders], dim = 1).to(self.args.device)
        comm_out = torch.cat([torch.zeros(comm_out[:,0].unsqueeze(1).shape), comm_out], dim = 1).to(self.args.device)
        all_mask = torch.cat([torch.ones(mask.shape[0], 1, 1), mask], dim = 1).to(self.args.device)
        mask = mask.to(self.args.device)
        episodes = reward.shape[0]
        steps = reward.shape[1]
        
        if(self.args.half):
            rgbd, sensors, father_comm, wheels_shoulders, comm_out, reward, done, mask, all_mask, mask = \
                rgbd.to(dtype=torch.float16), sensors.to(dtype=torch.float16), father_comm.to(dtype=torch.float16), wheels_shoulders.to(dtype=torch.float16), \
                comm_out.to(dtype=torch.float16), reward.to(dtype=torch.float16), done.to(dtype=torch.float16), \
                mask.to(dtype=torch.float16), wheels_shoulders.to(dtype=torch.float16), comm_out.to(dtype=torch.float16), all_mask.to(dtype=torch.float16), mask.to(dtype=torch.float16)
        
        #print("\n\n")
        #print("Agent {}, epoch {}. rgbd: {}. comm in: {}. wheels_shoulders: {}. comm out: {}. reward: {}.  done: {}. mask: {}.".format(
        #    self.agent_num, self.epochs, rgbd.shape, comm_in.shape, wheels_shoulders.shape, comm_out.shape, reward.shape, done.shape, mask.shape))
        #print("\n\n")
        
        return(rgbd, sensors, father_comm, mother_comm, wheels_shoulders, comm_out, reward, done, mask, all_mask, episodes, steps)
        
    
    
    def epoch(self, batch_size):
        self.epochs += 1
        self.total_epochs += 1
        self.train()
        parenting = self.processor.parenting
                                
        batch = self.get_batch(self.memory, batch_size)
        if(batch == False):
            return(False)
        batch_size = batch[0].shape[0]
        
        rgbd, sensors, father_comm, mother_comm, wheels_shoulders, comm_out, reward, done, mask, all_mask, episodes, steps = batch
        # Also sample from old memories, and use them to keep actor and critic outputs the same.
        
        
                
        # Train forward
        hps, hqs, rgbd_is, sensors_is, father_comm_is, pred_obs_q, labels = self.forward(
            torch.zeros((episodes, 1, self.args.pvrnn_mtrnn_size)), 
            Obs(rgbd, sensors, father_comm, mother_comm), Action(wheels_shoulders, comm_out))
                                
        rgbd_loss = F.binary_cross_entropy(pred_obs_q.rgbd, rgbd[:,1:], reduction = "none").mean((-1,-2,-3)).unsqueeze(-1) * mask * self.args.rgbd_scaler
                        
        sensors_loss = F.mse_loss(pred_obs_q.sensors, sensors[:,1:], reduction = "none")
        sensors_loss = sensors_loss.mean(-1).unsqueeze(-1) * mask * self.args.sensors_scaler
                        
        real_father_comm = father_comm[:,1:].reshape((episodes * steps, self.args.max_comm_len, self.args.comm_shape))
        real_father_comm = torch.argmax(real_father_comm, dim = -1)
        pred_father_comm = pred_obs_q.father_comm.reshape((pred_obs_q.father_comm.shape[0] * pred_obs_q.father_comm.shape[1], self.args.max_comm_len, self.args.comm_shape))
        pred_father_comm = pred_father_comm.transpose(1,2)
    
        father_comm_loss = F.cross_entropy(pred_father_comm, real_father_comm, reduction = "none")
        father_comm_loss = father_comm_loss.reshape(episodes, steps, self.args.max_comm_len)
        father_comm_loss = father_comm_loss.mean(dim=2).unsqueeze(-1) * mask * self.args.father_comm_scaler
        
        accuracy = (rgbd_loss + sensors_loss + father_comm_loss).mean()
        
        rgbd_complexity_for_hidden_state = rgbd_is.dkl.mean(-1).unsqueeze(-1) * all_mask
        sensors_complexity_for_hidden_state = sensors_is.dkl.mean(-1).unsqueeze(-1) * all_mask
        father_comm_complexity_for_hidden_state = father_comm_is.dkl.mean(-1).unsqueeze(-1) * all_mask
                
        complexity = sum([
            self.args.beta_rgbd * rgbd_complexity_for_hidden_state.mean(),
            self.args.beta_sensors * sensors_complexity_for_hidden_state.mean(),
            self.args.beta_father_comm * father_comm_complexity_for_hidden_state.mean()])       
                                
        self.forward_opt.zero_grad()
        (accuracy + complexity).backward()
        self.forward_opt.step()
        
        torch.cuda.empty_cache()
                        
                        
        
        # Get curiosity                 
        rgbd_prediction_error_curiosity     = self.args.prediction_error_eta_rgbd       * rgbd_loss
        sensors_prediction_error_curiosity  = self.args.prediction_error_eta_sensors    * sensors_loss
        father_comm_prediction_error_curiosity     = self.args.prediction_error_eta_father_comm       * father_comm_loss
        prediction_error_curiosity = rgbd_prediction_error_curiosity + father_comm_prediction_error_curiosity + sensors_prediction_error_curiosity
        
        rgbd_hidden_state_curiosity    = self.args.hidden_state_eta_rgbd       * torch.clamp(rgbd_complexity_for_hidden_state[:,1:], min = 0, max = self.args.dkl_max)  # Or tanh? sigmoid? Or just clamp?
        sensors_hidden_state_curiosity = self.args.hidden_state_eta_sensors    * torch.clamp(sensors_complexity_for_hidden_state[:,1:], min = 0, max = self.args.dkl_max)
        father_comm_hidden_state_curiosity    = self.args.hidden_state_eta_father_comm       * torch.clamp(father_comm_complexity_for_hidden_state[:,1:], min = 0, max = self.args.dkl_max)
        hidden_state_curiosity = rgbd_hidden_state_curiosity + father_comm_hidden_state_curiosity + sensors_hidden_state_curiosity
        
        if(self.args.curiosity == "prediction_error"):  curiosity = prediction_error_curiosity
        elif(self.args.curiosity == "hidden_state"):    curiosity = hidden_state_curiosity
        else:                                           curiosity = torch.zeros(reward.shape).to(self.args.device)
        extrinsic = torch.mean(reward).item()
        intrinsic_curiosity = curiosity.mean().item()
        reward += curiosity


                
        # Train critics
        with torch.no_grad():
            new_action, log_pis_next, log_pis_next_text = \
                self.actor(hqs.detach(), parenting)
            Q_target_nexts = []
            for i in range(self.args.critics):
                Q_target_next = self.critic_targets[i](new_action, hqs.detach())
                Q_target_next[:,1:]
                Q_target_nexts.append(Q_target_next)                
            log_pis_next = log_pis_next[:,1:]
            log_pis_next_text = log_pis_next_text[:,1:]
                        
            Q_target_nexts_stacked = torch.stack(Q_target_nexts, dim=0)
            Q_target_next, _ = torch.min(Q_target_nexts_stacked, dim=0)
            Q_target_next = Q_target_next[:,1:]
            if self.args.alpha == None:      alpha = self.alpha 
            else:                            alpha = self.args.alpha
            if self.args.alpha_text == None: alpha_text = self.alpha_text 
            else:                            alpha_text = self.args.alpha_text
            Q_targets = reward + (self.args.GAMMA * (1 - done) * (Q_target_next - (alpha * log_pis_next) - (alpha_text * log_pis_next_text)))
        
        critic_losses = []
        Qs = []
        for i in range(self.args.critics):
            Q = self.critics[i](Action(wheels_shoulders[:,1:], comm_out[:,1:]), hqs[:,:-1].detach())
            critic_loss = 0.5*F.mse_loss(Q*mask, Q_targets*mask)
            critic_losses.append(critic_loss)
            Qs.append(Q[0,0].item())
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
        
            self.soft_update(self.critics[i], self.critic_targets[i], self.args.tau)
        
        torch.cuda.empty_cache()
                                    
            
        
        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None:      alpha = self.alpha 
            else:                            alpha = self.args.alpha
            if self.args.alpha_text == None: alpha_text = self.alpha_text 
            else:                            alpha_text = self.args.alpha_text
            new_action, log_pis, log_pis_text = self.actor(hqs[:,:-1].detach(), parenting)
            
            loc = torch.zeros(self.args.wheels_shoulders_shape, dtype=torch.float64).to(self.args.device).float()
            n = self.args.wheels_shoulders_shape
            scale_tril = torch.tril(torch.ones(n, n)).to(self.args.device).float()
            policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
            policy_prior_log_prrgbd = self.args.normal_alpha * policy_prior.log_prob(new_action.wheels_shoulders).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis - policy_prior_log_prrgbd)*mask).item()
                
            Qs = []
            for i in range(self.args.critics):
                Q = self.critics[i](new_action, hqs[:,:-1].detach())
                Qs.append(Q)
            Qs_stacked = torch.stack(Qs, dim=0)
            Q, _ = torch.min(Qs_stacked, dim=0)
            Q = Q.mean(-1).unsqueeze(-1)
            
            actor_loss = ((alpha * log_pis - policy_prior_log_prrgbd) + (alpha_text * log_pis_text) - Q)*mask
            actor_loss = actor_loss.mean() / mask.mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        else:
            Q = None
            intrinsic_entropy = None
            intrinsic_imitation = None
            actor_loss = None
        
            
            
        # Train alpha
        if self.args.alpha == None:
            _, log_pis, _ = self.actor(hqs[:,:-1].detach(), parenting)
            alpha_loss = -(self.log_alpha.to(self.args.device) * (log_pis + self.target_entropy))*mask
            alpha_loss = alpha_loss.mean() / mask.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha.to(dtype=torch.float32)).to(self.args.device)
            torch.cuda.empty_cache()
        else:
            alpha_loss = None
            
        if self.args.alpha_text == None:
            _, _, log_pis_text = self.actor(hqs[:,:-1].detach(), parenting)
            alpha_text_loss = -(self.log_alpha_text.to(self.args.device) * (log_pis_text + self.target_entropy_text))*mask
            alpha_text_loss = alpha_text_loss.mean() / mask.mean()
            self.alpha_text_opt.zero_grad()
            alpha_text_loss.backward()
            self.alpha_text_opt.step()
            self.alpha_text = torch.exp(self.log_alpha_text.to(dtype=torch.float32)).to(self.args.device)
            torch.cuda.empty_cache()
        else:
            alpha_text_loss = None
                                
                                
                                
        if(accuracy != None):       accuracy = accuracy.item()
        if(rgbd_loss != None):      rgbd_loss = rgbd_loss.mean().item()
        if(sensors_loss != None):   sensors_loss = sensors_loss.mean().item()
        if(father_comm_loss != None):      father_comm_loss = father_comm_loss.mean().item()
        if(complexity != None):     complexity = complexity.item()
        if(alpha_loss != None):     alpha_loss = alpha_loss.item()
        if(alpha_text_loss != None): alpha_text_loss = alpha_text_loss.item()
        if(actor_loss != None):     actor_loss = actor_loss.item()
        if(Q != None):              Q = -Q.mean().item()
        for i in range(self.args.critics):
            if(critic_losses[i] != None): 
                critic_losses[i] = critic_losses[i].item()
                critic_losses[i] = log(critic_losses[i]) if critic_losses[i] > 0 else critic_losses[i]
                
        rgbd_prediction_error_curiosity = rgbd_prediction_error_curiosity.mean().item()
        sensors_prediction_error_curiosity = sensors_prediction_error_curiosity.mean().item()
        father_comm_prediction_error_curiosity = father_comm_prediction_error_curiosity.mean().item()
        
        rgbd_hidden_state_curiosity = rgbd_hidden_state_curiosity.mean().item()
        sensors_hidden_state_curiosity = sensors_hidden_state_curiosity.mean().item()
        father_comm_hidden_state_curiosity = father_comm_hidden_state_curiosity.mean().item()
        
        prediction_error_curiosity = prediction_error_curiosity.mean().item()
        hidden_state_curiosity = hidden_state_curiosity.mean().item()
        


        if(self.epochs == 1 or self.epochs >= sum(self.args.epochs) or self.epochs % self.args.keep_data == 0):
            self.plot_dict["accuracy"].append(accuracy)
            self.plot_dict["rgbd_loss"].append(rgbd_loss)
            self.plot_dict["sensors_loss"].append(sensors_loss)
            self.plot_dict["father_comm_loss"].append(father_comm_loss)
            self.plot_dict["complexity"].append(complexity)                                                                             
            self.plot_dict["alpha"].append(alpha_loss)
            self.plot_dict["alpha_text"].append(alpha_text_loss)
            self.plot_dict["actor"].append(actor_loss)
            for layer, f in enumerate(critic_losses):
                self.plot_dict["critics"][layer].append(f)    
            self.plot_dict["critics"].append(critic_losses)
            self.plot_dict["extrinsic"].append(extrinsic)
            self.plot_dict["q"].append(Q)
            self.plot_dict["intrinsic_curiosity"].append(intrinsic_curiosity)
            self.plot_dict["intrinsic_entropy"].append(intrinsic_entropy)
            self.plot_dict["rgbd_prediction_error_curiosity"].append(rgbd_prediction_error_curiosity)
            self.plot_dict["sensors_prediction_error_curiosity"].append(sensors_prediction_error_curiosity)
            self.plot_dict["father_comm_prediction_error_curiosity"].append(father_comm_prediction_error_curiosity)
            self.plot_dict["prediction_error_curiosity"].append(prediction_error_curiosity)
            self.plot_dict["rgbd_hidden_state_curiosity"].append(rgbd_hidden_state_curiosity)    
            self.plot_dict["sensors_hidden_state_curiosity"].append(sensors_hidden_state_curiosity)  
            self.plot_dict["father_comm_hidden_state_curiosity"].append(father_comm_hidden_state_curiosity)    
            self.plot_dict["hidden_state_curiosity"].append(hidden_state_curiosity)    
        
    
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        to_return = [self.forward.state_dict(), self.actor.state_dict()]
        for i in range(self.args.critics):
            to_return.append(self.critics[i].state_dict())
            to_return.append(self.critic_targets[i].state_dict())
        return(to_return)

    def load_state_dict(self, state_dict):
        self.forward.load_state_dict(state_dict = state_dict[0])
        self.actor.load_state_dict(state_dict = state_dict[1])
        for i in range(self.args.critics):
            self.critics[i].load_state_dict(state_dict = state_dict[2+2*i])
            self.critic_targets[i].load_state_dict(state_dict = state_dict[3+2*i])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.forward.eval()
        self.actor.eval()
        for i in range(self.args.critics):
            self.critics[i].eval()
            self.critic_targets[i].eval()

    def train(self):
        self.forward.train()
        self.actor.train()
        for i in range(self.args.critics):
            self.critics[i].train()
            self.critic_targets[i].train()
        
        
        
if __name__ == "__main__":
    agent = Agent(args = default_args)
    agent.save_episodes()
# %%