#%%

import os 
import psutil
from time import sleep
import numpy as np
from math import log
from itertools import accumulate
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

from utils import default_args, duration, dkl, calculate_similarity, onehots_to_string, many_onehots_to_strings, action_to_string, cpu_memory_usage, action_map, action_name_list, print, string_to_onehots, agent_to_english
from submodule_utils import model_start
from arena import Arena, get_physics
from task import Task, Task_Runner
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
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        
        if self.args.device.type == "cuda":
            print(f"\nIN AGENT: {i} DEVICE: {self.args.device} ({torch.cuda.current_device()} out of {[j for j in range(torch.cuda.device_count())]}, {torch.cuda.get_device_name(torch.cuda.current_device())})\n")
        else:
            print(f"\nIN AGENT: {i} DEVICE: {self.args.device}\n")
        
        #os.sched_setaffinity(0, {self.args.cpu})
        
        self.tasks = {
            "fp" :      Task(actions = [-1],                objects = 2, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2],         parenting = True, args = self.args),
            "w" :       Task(actions = [0],                 objects = 2, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2],         parenting = True, args = self.args),
            "wpulr" :   Task(actions = [0, 1, 2, 3, 4],     objects = 2, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2],         parenting = True, args = self.args),
            
            "fp5" :      Task(actions = [-1],                objects = 2, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2, 3, 4],         parenting = True, args = self.args),
            "w5" :       Task(actions = [0],                 objects = 2, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2, 3, 4],         parenting = True, args = self.args),
            "wpulr5" :   Task(actions = [0, 1, 2, 3, 4],     objects = 2, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2, 3, 4],         parenting = True, args = self.args)}
            
        physicsClient_1 = get_physics(GUI = GUI, time_step = self.args.time_step, steps_per_step = self.args.steps_per_step)
        self.arena_1 = Arena(physicsClient_1, args = self.args)
        physicsClient_2 = get_physics(GUI = False, time_step = self.args.time_step, steps_per_step = self.args.steps_per_step)
        self.arena_2 = Arena(physicsClient_2, args = self.args)
        self.task_runners = {task_name : Task_Runner(task, self.arena_1, self.arena_2, self.args) for i, (task_name, task) in enumerate(self.tasks.items())}
        self.task_name = self.args.task_list[0]
        
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
            "episode_dicts" : {}, 
            "agent_lists" : {} if (self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list) else {"forward" : PVRNN(self.args), "actor" : Actor(self.args), "critic" : Critic(self.args)},
            "rewards" : [], 
            "gen_rewards" : [], 
            "steps" : [],
            "accuracy" : [], 
            "rgbd_loss" : [], 
            "comm_loss" : [], 
            "sensors_loss" : [], 
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
            "comm_prediction_error_curiosity" : [], 
            "sensors_prediction_error_curiosity" : [], 
            "prediction_error_curiosity" : [], 
            "rgbd_hidden_state_curiosity" : [],
            "comm_hidden_state_curiosity" : [],
            "sensors_hidden_state_curiosity" : [],
            "hidden_state_curiosity" : [],
            "wins_all" : [],
            "gen_wins_all" : []}
        for a in action_map.values():
            self.plot_dict[f"wins_{a[1].lower()}"] = []
            self.plot_dict[f"gen_wins_{a[1].lower()}"] = []
                        
        
        
    def training(self, q = None):      
        start_time = duration()
        prev_time = duration()
        
        self.gen_test()  
        self.save_episodes()
        self.save_agent()
        while(True):
            cumulative_epochs = 0
            for i, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs == cumulative_epochs): 
                    
                    prev_time = duration()
                    
                    self.gen_test()  
                    self.save_episodes(swapping = False)
                    self.save_agent()
                    
                    parenting_before = self.tasks[self.task_name].parenting
                    # I think this line is why epochs of length 0 don't work the way I want.
                    self.task_name = self.args.task_list[i+1] 
                    # Try replacing it with something like this.
                    # while(epochs in task == 0 and tasks left > 0):
                    #   self.task_name = self.args.task_list[i+1] 
                    #   i += 1
                    self.old_memories.append(deepcopy(self.memory))
                    parenting_after = self.tasks[self.task_name].parenting
                    
                    if(parenting_before and not parenting_after):
                        self.actor.comm_out.load_state_dict(self.forward.predict_obs.comm_out.state_dict())
                    
                    self.gen_test()  
                    self.save_episodes(swapping = True)
                    self.save_agent()
                    
                    time = duration()
                    if(self.args.show_duration): print("AFTER GEN, SAVE EPISODE, SAVE AGENT:", time - prev_time)
                    prev_time = time
                
            prev_time = duration()
                
            step = self.training_episode()
            
            time = duration()
            if(self.args.show_duration): 
                print(f"TRAINING EPISODE ({step+1} steps):", time - prev_time)
                print(f"{self.epochs} EPOCHS:", time - start_time, "\n")
            prev_time = time
            
            percent_done = str(self.epochs / sum(self.args.epochs))
            if(q != None):
                q.put((self.agent_num, percent_done))
            if(self.epochs >= sum(self.args.epochs)): 
                break
            
            prev_time = duration()
            
            if(self.epochs % self.args.epochs_per_gen_test == 0):
                self.gen_test()  
                time = duration()
                if(self.args.show_duration): print("AFTER GEN:", time - prev_time)
            else:
                self.plot_dict["gen_wins_all"].append(None)
                win_dict_list = [self.plot_dict["gen_wins_" + action_name.lower()] for action_name in action_name_list]
                for i, win_dict in enumerate(win_dict_list):
                    win_dict.append(None)
            if(self.epochs % self.args.epochs_per_episode_dict == 0):
                self.save_episodes(swapping = False)
                time = duration()
                if(self.args.show_duration): print("AFTER SAVE EPISODE:", time - prev_time)
            if(self.epochs % self.args.epochs_per_agent_list == 0):
                self.save_agent()
                time = duration()
                if(self.args.show_duration): print("AFTER SAVE AGENT:", time - prev_time)
            
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        self.plot_dict["gen_rewards"] = list(accumulate(self.plot_dict["gen_rewards"]))
        
        prev_time = duration()
        
        self.gen_test()  
        self.save_episodes(swapping = False)
        self.save_agent()
        
        time = duration()
        if(self.args.show_duration): print("AFTER GEN, SAVE EPISODE, SAVE AGENT (END):", time - prev_time)
        prev_time = time
                
        self.min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in self.min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "episode_dicts", "agent_lists", "spot_names", "steps"]):
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
                        prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1, which_goal_message_1,
                        prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, which_goal_message_2, sleep_time = None):
                    
        start_time = duration()
        prev_time = duration()
        with torch.no_grad():
            self.eval()
            parenting = self.task.task.parenting
            rgbd_1, parent_comm, sensors_1 = self.task.obs()
            rgbd_2, _, sensors_2 = self.task.obs(agent_1 = False)
            recommended_action_1 = self.task.get_recommended_action() 
            comm_in_1 = string_to_onehots(which_goal_message_1).unsqueeze(0).unsqueeze(0) if self.task.task.goal[0] == -1 else parent_comm.unsqueeze(0) if parenting else prev_comm_out_2
            
            time = duration()
            if(self.args.show_duration): print("STEP A:", time - start_time)

            hp_1, hq_1, rgbd_dkls_1, comm_dkls_1, sensors_dkls_1 = self.forward.bottom_to_top_step(
                hq_1, 
                self.forward.rgbd_in(rgbd_1), self.forward.comm_in(comm_in_1), self.forward.sensors_in(sensors_1), 
                self.forward.action_in(prev_action_1), self.forward.comm_out_in(prev_comm_out_1)) 
            
            time = duration()
            if(self.args.show_duration): print("STEP B:", time - start_time)

            action_1, comm_out_1, _, _, ha_1 = self.actor(rgbd_1, comm_in_1, sensors_1, prev_action_1, prev_comm_out_1, hq_1.detach(), ha_1, parenting) 
            values_1 = []
            new_hcs_1 = []
            for i in range(self.args.critics):
                value, hc = self.critics[i](rgbd_1, comm_in_1, sensors_1, action_1, comm_out_1, hq_1.detach(), hcs_1[i]) 
                values_1.append(round(value.item(), 3))
                new_hcs_1.append(hc)
                
            time = duration()
            if(self.args.show_duration): print("STEP C:", time - start_time)
            
            if(parenting): 
                action_2 = torch.zeros_like(action_1)
                comm_out_2 = None
                hp_2 = None
                hq_2 = None
                values_2 = None
                new_hcs_2 = None
                rgbd_dkls_2 = None 
                comm_dkls_2 = None
                sensors_dkls_2 = None
            else:
                recommended_action_2 = self.task.get_recommended_action(agent_1 = False)
                comm_in_2 = string_to_onehots(which_goal_message_2).unsqueeze(0).unsqueeze(0) if self.task.task.goal[0] == -1 else parent_comm.unsqueeze(0) if parenting else prev_comm_out_1
                hp_2, hq_2, rgbd_dkls_2, comm_dkls_2, sensors_dkls_2 = self.forward.bottom_to_top(
                    hq_2, 
                    self.forward.rgbd_in(rgbd_2), self.forward.comm_in(comm_in_2), self.forward.sensors_in(sensors_2), 
                    self.forward.action_in(prev_action_2), self.forward.comm_out_in(prev_comm_out_2))
                action_2, comm_out_2, _, _, ha_2 = self.actor(rgbd_2, prev_comm_out_1, sensors_2, prev_action_2, prev_comm_out_2, hq_2.detach(), ha_2, parenting) 
                values_2 = []
                new_hcs_2 = []
                for i in range(self.args.critics):
                    value, hc = self.critics[i](rgbd_2, parent_comm if parenting else prev_comm_out_1, sensors_2, action_2, comm_out_2, hq_2.detach(), hcs_2[i]) 
                    values_2.append(round(value.item(), 3))
                    new_hcs_2.append(hc)
                                    
            time = duration()
            if(self.args.show_duration): print("\nUP TO STEP:", time - prev_time)
            prev_time = time
            
            raw_reward, distance_reward, angle_reward, distance_reward_2, angle_reward_2, done, win, which_goal_message_1, which_goal_message_2 = self.task.step(action_1[0,0].clone(), action_2[0,0].clone(), sleep_time = sleep_time)
            
            time = duration()
            if(self.args.show_duration): print("AFTER STEP:", time - prev_time)
            prev_time = time
            
            total_reward = raw_reward + distance_reward + angle_reward 
            total_reward_2 = raw_reward + distance_reward_2 + angle_reward_2
            next_rgbd_1, next_parent_comm, next_sensors_1 = self.task.obs()
            next_rgbd_2, _, next_sensors_2 = self.task.obs(agent_1 = False)
            
            next_comm_in_1 = string_to_onehots(which_goal_message_1).unsqueeze(0).unsqueeze(0) if self.task.task.goal[0] == -1 else next_parent_comm.unsqueeze(0) if parenting else comm_out_2 
            next_comm_in_2 = string_to_onehots(which_goal_message_2).unsqueeze(0).unsqueeze(0) if self.task.task.goal[0] == -1 else next_parent_comm.unsqueeze(0) if parenting else comm_out_1
            
            comm_curious = 1 if (not parenting or self.task.task.goal[0] == -1) else 0 # True in free play or unparented
            
            to_push_1 = [
                rgbd_1,
                comm_in_1,
                sensors_1,
                action_1,
                comm_out_1,
                recommended_action_1,
                total_reward,
                comm_curious,
                next_rgbd_1,
                next_comm_in_1,
                next_sensors_1,
                done]
            
            if(parenting): 
                to_push_2 = None
            else:
                to_push_2 = [
                    rgbd_2,
                    comm_in_2,
                    sensors_2,
                    action_2,
                    comm_out_2,
                    recommended_action_2,
                    total_reward,
                    comm_curious,
                    next_rgbd_2,
                    next_comm_in_2,
                    next_sensors_2,
                    done]
        torch.cuda.empty_cache()
        
        time = duration()
        if(self.args.show_duration): print("COMPLETE STEP:", time - start_time, "\n")
        prev_time = time
        
        return(action_1, comm_out_1, values_1, hp_1.squeeze(1), hq_1.squeeze(1), ha_1, new_hcs_1, rgbd_dkls_1, comm_dkls_1, sensors_dkls_1, which_goal_message_1,
               action_2, comm_out_2, values_2, None if hp_2 == None else hp_2.squeeze(1), None if hq_2 == None else hq_2.squeeze(1), ha_2, new_hcs_2, rgbd_dkls_2, comm_dkls_2, sensors_dkls_2, which_goal_message_2,
               raw_reward, total_reward, distance_reward, angle_reward, total_reward_2, distance_reward_2, angle_reward_2, done, win, to_push_1, to_push_2)
            
           
           
    def start_episode(self):
        start_time = duration()
        prev_time = duration()
        
        done = False
        complete_reward = 0
        steps = 0
                
        to_push_list_1 = []
        prev_action_1 = torch.zeros((1, 1, self.args.action_shape))
        prev_comm_out_1 = torch.zeros((1, 1, self.args.max_comm_len, self.args.comm_shape))
        hq_1 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
        ha_1 = torch.zeros((1, 1, self.args.hidden_size)) 
        hcs_1 = [torch.zeros((1, 1, self.args.hidden_size))] * self.args.critics
        which_goal_message_1 = " " * self.args.max_comm_len
        
        to_push_list_2 = []
        prev_action_2 = torch.zeros((1, 1, self.args.action_shape))
        prev_comm_out_2 = torch.zeros((1, 1, self.args.max_comm_len, self.args.comm_shape))
        hq_2 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
        ha_2 = torch.zeros((1, 1, self.args.hidden_size)) 
        hcs_2 = [torch.zeros((1, 1, self.args.hidden_size))] * self.args.critics
        which_goal_message_2 = " " * self.args.max_comm_len
        
        return(start_time, prev_time, done, complete_reward, steps, 
               to_push_list_1, prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1, which_goal_message_1,
               to_push_list_2, prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, which_goal_message_2)
           
           
    
    def training_episode(self):        
        start_time, prev_time, done, complete_reward, steps, \
            to_push_list_1, prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1, which_goal_message_1, \
            to_push_list_2, prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, which_goal_message_2 = self.start_episode()
                    
        self.task = self.task_runners[self.task_name]
        self.task.begin()    
                        
        for step in range(self.args.max_steps):
            self.steps += 1                                                                                             
            if(not done):
                steps += 1
                prev_action_1, prev_comm_out_1, values_1, hp_1, hq_1, ha_1, hcs_1, rgbd_dkls_1, comm_dkls_1, sensors_dkls_1, which_goal_message_1, \
                    prev_action_2, prev_comm_out_2, values_2, hp_2, hq_2, ha_2, hcs_2, rgbd_dkls_2, comm_dkls_2, sensors_dkls_2, which_goal_message_2, \
                        raw_reward, total_reward, distance_reward, angle_reward, total_reward_2, distance_reward_2, angle_reward_2, done, win, to_push_1, to_push_2 = self.step_in_episode(
                            prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1, which_goal_message_1,
                            prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, which_goal_message_2)
                        
                to_push_list_1.append(to_push_1)
                to_push_list_2.append(to_push_2)
                complete_reward += total_reward
            if(self.steps % self.args.steps_per_epoch == 0):
                prev_time = duration()
                plot_data = self.epoch(self.args.batch_size)
                time = duration()
                if(self.args.show_duration): print("\nEPOCH:", time - prev_time)
                prev_time = time
                if(plot_data == False): pass
                else:
                    accuracy, rgbd_loss, comm_loss, sensors_loss, complexity, \
                        alpha_loss, alpha_text_loss, actor_loss, critic_losses, \
                        e, q, ic, ie, ii, \
                        rgbd_prediction_error_curiosity, comm_prediction_error_curiosity, sensors_prediction_error_curiosity, prediction_error_curiosity,\
                        rgbd_hidden_state_curiosity, comm_hidden_state_curiosity, sensors_hidden_state_curiosity, hidden_state_curiosity = plot_data
                    if(self.epochs == 1 or self.epochs >= sum(self.args.epochs) or self.epochs % self.args.keep_data == 0):
                        self.plot_dict["accuracy"].append(accuracy)
                        self.plot_dict["rgbd_loss"].append(rgbd_loss)
                        self.plot_dict["comm_loss"].append(comm_loss)
                        self.plot_dict["sensors_loss"].append(sensors_loss)
                        self.plot_dict["complexity"].append(complexity)                                                                             
                        self.plot_dict["alpha"].append(alpha_loss)
                        self.plot_dict["alpha_text"].append(alpha_text_loss)
                        self.plot_dict["actor"].append(actor_loss)
                        for layer, f in enumerate(critic_losses):
                            self.plot_dict["critics"][layer].append(f)    
                        self.plot_dict["critics"].append(critic_losses)
                        self.plot_dict["extrinsic"].append(e)
                        self.plot_dict["q"].append(q)
                        self.plot_dict["intrinsic_curiosity"].append(ic)
                        self.plot_dict["intrinsic_entropy"].append(ie)
                        self.plot_dict["intrinsic_imitation"].append(ii)
                        self.plot_dict["rgbd_prediction_error_curiosity"].append(rgbd_prediction_error_curiosity)
                        self.plot_dict["comm_prediction_error_curiosity"].append(comm_prediction_error_curiosity)
                        self.plot_dict["sensors_prediction_error_curiosity"].append(sensors_prediction_error_curiosity)
                        self.plot_dict["prediction_error_curiosity"].append(prediction_error_curiosity)
                        self.plot_dict["rgbd_hidden_state_curiosity"].append(rgbd_hidden_state_curiosity)    
                        self.plot_dict["comm_hidden_state_curiosity"].append(comm_hidden_state_curiosity)    
                        self.plot_dict["sensors_hidden_state_curiosity"].append(sensors_hidden_state_curiosity)  
                        self.plot_dict["hidden_state_curiosity"].append(hidden_state_curiosity)    
                                                    
        self.task.done()
        self.plot_dict["steps"].append(steps)
        self.plot_dict["rewards"].append(complete_reward)
        goal_action = self.task.task.goal[0]
        self.plot_dict["wins_all"].append(win)
        win_dict_list = [self.plot_dict["wins_" + action_name.lower()] for action_name in action_name_list]
        for i, win_dict in enumerate(win_dict_list):
            if(i-1 == goal_action): win_dict.append(win)
            else:                   win_dict.append(None)
                             
        for to_push in to_push_list_1:
            rgbd, comm_in, sensors, action, comm_out, recommended_action, total_reward, comm_curious, next_rgbd, next_comm_in, next_sensors, done = to_push
            self.memory.push(
                rgbd.to("cpu"),
                comm_in.to("cpu"), 
                sensors.to("cpu"),
                action.to("cpu"), 
                comm_out.to("cpu"),
                recommended_action.to("cpu"),
                total_reward, 
                comm_curious,
                next_rgbd.to("cpu"),
                next_comm_in.to("cpu"), 
                next_sensors.to("cpu"),
                done)
            
        for to_push in to_push_list_2:
            if(to_push != None):
                rgbd, comm_in, sensors, action, comm_out, recommended_action, total_reward, comm_curious, next_rgbd, next_comm_in, next_sensors, done = to_push
                self.memory.push(
                    rgbd.to("cpu"),
                    comm_in.to("cpu"), 
                    sensors.to("cpu"),
                    action.to("cpu"), 
                    comm_out.to("cpu"),
                    recommended_action.to("cpu"),
                    total_reward, 
                    comm_curious,
                    next_rgbd.to("cpu"),
                    next_comm_in.to("cpu"), 
                    next_sensors.to("cpu"),
                    done)
                
        self.episodes += 1
        
        return(step)
        
        
        
    def gen_test(self):
        start_time, prev_time, done, complete_reward, steps, \
            to_push_list_1, prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1, which_goal_message_1, \
            to_push_list_2, prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, which_goal_message_2 = self.start_episode()
                
        try:
            self.task = self.task_runners[self.task_name]
            self.task.begin(test = True)        
            for step in range(self.args.max_steps):
                #print("Step", step)
                if(not done):
                    prev_action_1, prev_comm_out_1, values_1, hp_1, hq_1, ha_1, hcs_1, rgbd_dkls_1, comm_dkls_1, sensors_dkls_1, which_goal_message_1, \
                        prev_action_2, prev_comm_out_2, values_2, hp_2, hq_2, ha_2, hcs_2, rgbd_dkls_2, comm_dkls_2, sensors_dkls_2, which_goal_message_2, \
                            raw_reward, total_reward, distance_reward, angle_reward, total_reward_2, distance_reward_2, angle_reward_2, done, win, to_push_1, to_push_2 = self.step_in_episode(
                                prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1, which_goal_message_1,
                                prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, which_goal_message_2)
                    complete_reward += total_reward
                #print("DONE")
            time = duration()
            if(self.args.show_duration): print(f"FULL GEN TEST EPISODE ({step + 1} steps):", time - start_time, "\n")
            self.task.done()
            goal_action = self.task.task.goal[0]
            self.plot_dict["gen_wins_all"].append(win)
            win_dict_list = [self.plot_dict["gen_wins_" + action_name.lower()] for action_name in action_name_list]
            for i, win_dict in enumerate(win_dict_list):
                if(i-1 == goal_action): 
                    win_dict.append(win)
                else: win_dict.append(None)
        except:
            complete_reward = 0
            win = False
            self.plot_dict["gen_wins_all"].append(win)
            win_dict_list = [self.plot_dict["gen_wins_" + action_name.lower()] for action_name in action_name_list]
            for i, win_dict in enumerate(win_dict_list):
                win_dict.append(None)
        self.plot_dict["gen_rewards"].append(complete_reward)
        return(win)
        
        
        
    def save_episodes(self, swapping = False, test = False, sleep_time = None, for_display = False):
        with torch.no_grad():
            self.task = self.task_runners[self.task_name]
            self.task.begin(test = test)       
            comm_from_parent = self.task.task.parenting
            if(self.args.agents_per_episode_dict != -1 and self.agent_num > self.args.agents_per_episode_dict): 
                return
            for episode_num in range(self.args.episodes_in_episode_dict):
                episode_dict = {
                    "rgbds_1" : [],
                    "comms_in_1" : [],
                    "sensors_1" : [],
                    "recommended_1" : [],
                    "actions_1" : [],
                    "action_texts_1" : [],
                    "comms_out_1" : [],
                    "birds_eye_1" : [],
                    "raw_rewards" : [],
                    "total_rewards_1" : [],
                    "distance_rewards_1" : [],
                    "angle_rewards_1" : [],
                    "critic_predictions_1" : [],
                    "prior_predicted_rgbds_1" : [],
                    "prior_predicted_comms_in_1" : [],
                    "prior_predicted_sensors_1" : [],
                    "posterior_predicted_rgbds_1" : [],
                    "posterior_predicted_comms_in_1" : [],
                    "posterior_predicted_sensors_1" : [], 
                    "rgbd_dkls_1" : [],
                    "comm_dkls_1" : [],
                    "sensors_dkls_1" : [],
                    "which_goal_message_1" : [],
                    
                    "rgbds_2" : [],
                    "comms_in_2" : [],
                    "sensors_2" : [],
                    "recommended_2" : [],
                    "actions_2" : [],
                    "action_texts_2" : [],
                    "comms_out_2" : [],
                    "birds_eye_2" : [],
                    "total_rewards_2" : [],
                    "distance_rewards_2" : [],
                    "angle_rewards_2" : [],
                    "critic_predictions_2" : [],
                    "prior_predicted_rgbds_2" : [],
                    "prior_predicted_comms_in_2" : [],
                    "prior_predicted_sensors_2" : [],
                    "posterior_predicted_rgbds_2" : [],
                    "posterior_predicted_comms_in_2" : [],
                    "posterior_predicted_sensors_2" : [],
                    "rgbd_dkls_2" : [],
                    "comm_dkls_2" : [],
                    "sensors_dkls_2" : [],
                    "which_goal_message_2" : []}
                
                episode_dict["task"] = self.task.task
                episode_dict["goal"] = "'{}' ({})".format(self.task.task.goal_text, self.task.task.goal_human_text)
                
                done = False
                
                start_time, prev_time, done, complete_reward, steps, \
                    to_push_list_1, prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1, which_goal_message_1, \
                    to_push_list_2, prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, which_goal_message_2 = self.start_episode()
                        
                hp_1 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
                hq_1 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 

                hp_2 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
                hq_2 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
                                
                def save_step(step, hp, hq, action, agent_1 = True):
                    agent_num = 1 if agent_1 else 2
                    
                    recommended = self.task.get_recommended_action(agent_1 = agent_1)
                    episode_dict[f"recommended_{agent_num}"].append(None if recommended == None else action_to_string(recommended))
                    
                    birds_eye = self.task.arena_1.photo_from_above() if agent_1 else self.task.arena_2.photo_from_above()
                    rgbd, parent_comm, sensors = self.task.obs(agent_1 = agent_1)
                    
                    episode_dict[f"birds_eye_{agent_num}"].append(birds_eye[:,:,0:3])
                    episode_dict[f"rgbds_{agent_num}"].append(rgbd[0,:,:,0:3])        
                    episode_dict[f"sensors_{agent_num}"].append(sensors.tolist()[0])
                    
                    if(agent_1):
                        comm_in = which_goal_message_1 if self.task.task.goal[0] == -1 else onehots_to_string(parent_comm[0]) if comm_from_parent else prev_comm_out_2[0,0]
                    else:
                        comm_in = which_goal_message_2 if self.task.task.goal[0] == -1 else onehots_to_string(parent_comm[0]) if comm_from_parent else prev_comm_out_1[0,0]
                    episode_dict[f"comms_in_{agent_num}"].append("'{}' ({})".format(comm_in, self.task.task.agent_to_english(comm_in)))
                    
                    if(agent_1):
                        which_goal_message = which_goal_message_1
                    else:
                        which_goal_message = which_goal_message_2
                    episode_dict[f"which_goal_message_{agent_num}"].append(agent_to_english(which_goal_message))
                    
                    if(step != 0):
                        
                        pred_rgbds_p, pred_comm_in_p, pred_sensors_p = self.forward.predict(hp.unsqueeze(1), self.forward.action_in(action)) 
                        pred_rgbds_q, pred_comm_in_q, pred_sensors_q = self.forward.predict(hq.unsqueeze(1), self.forward.action_in(action))

                        episode_dict[f"prior_predicted_rgbds_{agent_num}"].append(pred_rgbds_p[0,0][:,:,0:3])
                        prior_predicted_comms_in = onehots_to_string(pred_comm_in_p[0,0])
                        episode_dict[f"prior_predicted_comms_in_{agent_num}"].append("'{}' ({})".format(prior_predicted_comms_in, self.task.task.agent_to_english(prior_predicted_comms_in)))
                        episode_dict[f"prior_predicted_sensors_{agent_num}"].append([round(o.item(), 2) for o in pred_sensors_p[0,0]])
                        
                        episode_dict[f"posterior_predicted_rgbds_{agent_num}"].append(pred_rgbds_q[0,0][:,:,0:3])
                        posterior_predicted_comms_in = onehots_to_string(pred_comm_in_q[0,0])
                        episode_dict[f"posterior_predicted_comms_in_{agent_num}"].append("'{}' ({})".format(posterior_predicted_comms_in, self.task.task.agent_to_english(posterior_predicted_comms_in)))
                        episode_dict[f"posterior_predicted_sensors_{agent_num}"].append([round(o.item(), 2) for o in pred_sensors_q[0,0]])
                    
                def display(step, agent_1 = True, done = False, stopping = False, wait = True):
                    if(for_display):
                        print(f"\n{self.task.task.goal_human_text}", end = " ")
                        print("STEP:", step)
                        plot_step(step, episode_dict, agent_1 = agent_1, last_step = done, saving = False)
                        if(not self.task.task.parenting and not stopping):
                            display(step, agent_1 = False, stopping = True)
                        if(wait):
                            WAITING = input("WAITING")
                
                for step in range(self.args.max_steps + 1):
                    save_step(step, hp_1, hq_1, action = prev_action_1, agent_1 = True)    
                    if(not self.task.task.parenting):
                        save_step(step, hp_2, hq_2, action = prev_action_2, agent_1 = False)  
                        
                    display(step)
                    
                    prev_action_1, prev_comm_out_1, values_1, hp_1, hq_1, ha_1, hcs_1, rgbd_dkls_1, comm_dkls_1, sensors_dkls_1, which_goal_message_1, \
                        prev_action_2, prev_comm_out_2, values_2, hp_2, hq_2, ha_2, hcs_2, rgbd_dkls_2, comm_dkls_2, sensors_dkls_2, which_goal_message_2, \
                            raw_reward, total_reward, distance_reward, angle_reward, total_reward_2, distance_reward_2, angle_reward_2, done, win, to_push_1, to_push_2 = self.step_in_episode(
                                prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1, which_goal_message_1,
                                prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, which_goal_message_2, sleep_time) 
                            
                    episode_dict["raw_rewards"].append(str(round(raw_reward, 3)))
                    
                    episode_dict["actions_1"].append(prev_action_1)
                    episode_dict["action_texts_1"].append(action_to_string(prev_action_1))
                    comm_out_1 = onehots_to_string(prev_comm_out_1)
                    episode_dict["comms_out_1"].append("{} ({})".format(comm_out_1, self.task.task.agent_to_english(comm_out_1)))
                    episode_dict["rgbd_dkls_1"].append(rgbd_dkls_1.sum().item())
                    episode_dict["comm_dkls_1"].append(comm_dkls_1.sum().item())
                    episode_dict["sensors_dkls_1"].append(sensors_dkls_1.sum().item())
                    episode_dict["critic_predictions_1"].append(values_1)
                    episode_dict["total_rewards_1"].append(str(round(total_reward, 3)))
                    episode_dict["distance_rewards_1"].append(str(round(distance_reward, 3)))
                    episode_dict["angle_rewards_1"].append(str(round(angle_reward, 3)))
                        
                    if(not self.task.task.parenting):
                        episode_dict["actions_2"].append(prev_action_2)
                        episode_dict["action_texts_2"].append(action_to_string(prev_action_2))
                        comm_out_2 = onehots_to_string(prev_comm_out_2)
                        episode_dict["comms_out_2"].append("{} ({})".format(comm_out_1, self.task.task.agent_to_english(comm_out_2)))
                        episode_dict["rgbd_dkls_2"].append(rgbd_dkls_2.sum().item())
                        episode_dict["comm_dkls_2"].append(comm_dkls_2.sum().item())
                        episode_dict["sensors_dkls_2"].append(sensors_dkls_2.sum().item())
                        episode_dict["critic_predictions_2"].append(values_2)
                        episode_dict["total_rewards_2"].append(str(round(total_reward_2, 3)))
                        episode_dict["distance_rewards_2"].append(str(round(distance_reward_2, 3)))
                        episode_dict["angle_rewards_2"].append(str(round(angle_reward_2, 3)))
                    
                    if(done):
                        save_step(step, hp_1, hq_1, action = prev_action_1, agent_1 = True)    
                        if(not self.task.task.parenting):
                            save_step(step, hp_2, hq_2, action = prev_action_2, agent_1 = False) 
                        display(step + 1, done = True, wait = False)
                        self.task.done()
                        time = duration()
                        if(self.args.show_duration): print(f"\nFULL SAVE EPISODE EPISODE ({step + 1} steps):", time - start_time, "\n")
                        break
                
                if(for_display):
                    return(win)
                else:
                    self.plot_dict["episode_dicts"]["{}_{}_{}_{}".format(self.agent_num, self.epochs, episode_num, 1 if swapping else 0)] = episode_dict
                            
        
        
    def save_agent(self):
        if(self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list): return
        self.plot_dict["agent_lists"]["{}_{}".format(self.agent_num, self.epochs)] = deepcopy(self.state_dict())
        
        
        
    def get_batch(self, memory, batch_size):
        batch = memory.sample(batch_size)
        if(batch == False): return(False)
        
        prev_time = duration()
                        
        self.epochs += 1

        rgbds, comms_in, sensors, actions, comms_out, recommended_actions, rewards, comm_curious, dones, masks = batch
        rgbds = torch.from_numpy(rgbds).to(self.args.device)
        comms_in = torch.from_numpy(comms_in).to(self.args.device)
        sensors = torch.from_numpy(sensors).to(self.args.device)
        actions = torch.from_numpy(actions)
        comms_out = torch.from_numpy(comms_out)
        recommended_actions = torch.from_numpy(recommended_actions).to(self.args.device)
        rewards = torch.from_numpy(rewards).to(self.args.device)
        comm_curious = torch.from_numpy(comm_curious).to(self.args.device)
        dones = torch.from_numpy(dones).to(self.args.device)
        masks = torch.from_numpy(masks)
        actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions], dim = 1).to(self.args.device)
        comms_out = torch.cat([torch.zeros(comms_out[:,0].unsqueeze(1).shape), comms_out], dim = 1).to(self.args.device)
        all_masks = torch.cat([torch.ones(masks.shape[0], 1, 1), masks], dim = 1).to(self.args.device)
        masks = masks.to(self.args.device)
        episodes = rewards.shape[0]
        steps = rewards.shape[1]
        
        if(self.args.half):
            rgbds, comms_in, sensors, actions, comms_out, recommended_actions, rewards, comm_curious, dones, masks, actions, comms_out, all_masks, masks = \
                rgbds.to(dtype=torch.float16), comms_in.to(dtype=torch.float16), sensors.to(dtype=torch.float16), actions.to(dtype=torch.float16), \
                comms_out.to(dtype=torch.float16), recommended_actions.to(dtype=torch.float16), rewards.to(dtype=torch.float16), comm_curious.to(dtype=torch.float16), dones.to(dtype=torch.float16), \
                masks.to(dtype=torch.float16), actions.to(dtype=torch.float16), comms_out.to(dtype=torch.float16), all_masks.to(dtype=torch.float16), masks.to(dtype=torch.float16)
        
        #print("\n\n")
        #print("Agent {}, epoch {}. rgbds: {}. comms in: {}. actions: {}. comms out: {}. recommended actions: {}. rewards: {}. comm_curious: {}. dones: {}. masks: {}.".format(
        #    self.agent_num, self.epochs, rgbds.shape, comms_in.shape, actions.shape, comms_out.shape, recommended_actions.shape, rewards.shape, comm_curious.shape, dones.shape, masks.shape))
        #print("\n\n")
        
        return(rgbds, comms_in, sensors, actions, comms_out, recommended_actions, rewards, comm_curious, dones, actions, comms_out, masks, all_masks, episodes, steps)
        
    
    
    def epoch(self, batch_size):
        self.train()
        parenting = self.task.task.parenting
                                
        batch = self.get_batch(self.memory, batch_size)
        if(batch == False):
            return(False)
        
        rgbds, comms_in, sensors, actions, comms_out, recommended_actions, rewards, comm_curious, dones, actions, comms_out, masks, all_masks, episodes, steps = batch
        # Also sample from old memories, and use them to keep actor and critic outputs the same.
        
        time = duration()
        start_time = duration()
        prev_time = time
                
        # Train forward
        hps, hqs, rgbd_dkl, comm_dkl, sensors_dkl, pred_rgbd_q, pred_comms_q, pred_sensors_q = self.forward(
            torch.zeros((episodes, self.args.layers, self.args.pvrnn_mtrnn_size)), 
            rgbds, comms_in, sensors, actions, comms_out)
                        
        rgbd_loss = F.binary_cross_entropy(pred_rgbd_q, rgbds[:,1:], reduction = "none").mean((-1,-2,-3)).unsqueeze(-1) * masks * self.args.rgbd_scaler
                        
        real_comms = comms_in[:,1:].reshape((episodes * steps, self.args.max_comm_len, self.args.comm_shape))
        real_comms = torch.argmax(real_comms, dim = -1)
        pred_comms = pred_comms_q.reshape((pred_comms_q.shape[0] * pred_comms_q.shape[1], self.args.max_comm_len, self.args.comm_shape))
        pred_comms = pred_comms.transpose(1,2)
    
        comm_loss = F.cross_entropy(pred_comms, real_comms, reduction = "none")
        comm_loss = comm_loss.reshape(episodes, steps, self.args.max_comm_len)
        comm_loss = comm_loss.mean(dim=2).unsqueeze(-1) * masks * self.args.comm_scaler
        
        sensors_loss = F.mse_loss(pred_sensors_q, sensors[:,1:], reduction = "none")
        sensors_loss = sensors_loss.mean(-1).unsqueeze(-1) * masks * self.args.sensors_scaler
        
        accuracy = (rgbd_loss + comm_loss + sensors_loss).mean()
        
        rgbd_complexity_for_hidden_state = rgbd_dkl.mean(-1).unsqueeze(-1) * all_masks
        comm_complexity_for_hidden_state = comm_dkl.mean(-1).unsqueeze(-1) * all_masks
        sensors_complexity_for_hidden_state = sensors_dkl.mean(-1).unsqueeze(-1) * all_masks
                
        complexity = sum([
            self.args.beta_rgbd * rgbd_complexity_for_hidden_state.mean(),
            self.args.beta_comm * comm_complexity_for_hidden_state.mean(),
            self.args.beta_sensors * sensors_complexity_for_hidden_state.mean()])       
                                
        time = duration()
        if(self.args.show_duration): print("USED FORWARD:", time - prev_time)
        prev_time = time
                                
        if(self.args.train_together):
            complete_loss = self.args.forward_scaler * (accuracy + complexity)
        else:
            self.forward_opt.zero_grad()
            (accuracy + complexity).backward()
            self.forward_opt.step()
        
        if(self.args.beta == 0): complexity = None
        torch.cuda.empty_cache()
        
        time = duration()
        if(self.args.show_duration): print("TRAINED FORWARD:", time - prev_time)
        prev_time = time
                        
                        
        
        # Get curiosity                 
        rgbd_prediction_error_curiosity     = self.args.prediction_error_eta_rgbd       * rgbd_loss
        comm_prediction_error_curiosity     = self.args.prediction_error_eta_comm       * comm_loss
        if(self.args.selective_comm_curiosity): comm_prediction_error_curiosity         *= comm_curious
        sensors_prediction_error_curiosity  = self.args.prediction_error_eta_sensors    * sensors_loss
        prediction_error_curiosity = rgbd_prediction_error_curiosity + comm_prediction_error_curiosity + sensors_prediction_error_curiosity
        
        rgbd_hidden_state_curiosity    = self.args.hidden_state_eta_rgbd       * torch.clamp(rgbd_complexity_for_hidden_state[:,1:], min = 0, max = self.args.dkl_max)  # Or tanh? sigmoid? Or just clamp?
        comm_hidden_state_curiosity    = self.args.hidden_state_eta_comm       * torch.clamp(comm_complexity_for_hidden_state[:,1:], min = 0, max = self.args.dkl_max)
        if(self.args.selective_comm_curiosity): comm_hidden_state_curiosity    *= comm_curious
        sensors_hidden_state_curiosity = self.args.hidden_state_eta_sensors    * torch.clamp(sensors_complexity_for_hidden_state[:,1:], min = 0, max = self.args.dkl_max)
        hidden_state_curiosity = rgbd_hidden_state_curiosity + comm_hidden_state_curiosity + sensors_hidden_state_curiosity
        
        if(self.args.curiosity == "prediction_error"):  curiosity = prediction_error_curiosity
        elif(self.args.curiosity == "hidden_state"):    curiosity = hidden_state_curiosity
        else:                                           curiosity = torch.zeros(rewards.shape).to(self.args.device)
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        rewards += curiosity
        
        time = duration()
        if(self.args.show_duration): print("CURIOSITY:", time - prev_time)
        prev_time = time
                        
        
                
        # Train critics
        with torch.no_grad():
            new_actions, new_comms_out, log_pis_next, log_pis_next_text, _ = \
                self.actor(rgbds, comms_in, sensors, actions, comms_out, hqs.detach(), 
                           torch.zeros((episodes, steps + 1, self.args.hidden_size)), parenting)
            Q_target_nexts = []
            for i in range(self.args.critics):
                Q_target_next, _ = self.critic_targets[i](rgbds, comms_in, sensors, new_actions, new_comms_out, hqs.detach(), torch.zeros((episodes, steps + 1, self.args.hidden_size)))
                Q_target_next[:,1:]
                Q_target_nexts.append(Q_target_next)                
            log_pis_next = log_pis_next[:,1:]
            log_pis_next_text = log_pis_next_text[:,1:]
            recommendation_value = calculate_similarity(recommended_actions, new_actions[:,:-1]).unsqueeze(-1)
                        
            Q_target_nexts_stacked = torch.stack(Q_target_nexts, dim=0)
            Q_target_next, _ = torch.min(Q_target_nexts_stacked, dim=0)
            Q_target_next = Q_target_next[:,1:]
            if self.args.alpha == None:      alpha = self.alpha 
            else:                            alpha = self.args.alpha
            if self.args.alpha_text == None: alpha_text = self.alpha_text 
            else:                            alpha_text = self.args.alpha_text
            Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - (alpha * log_pis_next) - (alpha_text * log_pis_next_text) + (self.args.delta * recommendation_value)))
            
        time = duration()
        #if(self.args.show_duration): print("USED TARGET CRITICS:", time - prev_time)
        prev_time = time
        
        critic_losses = []
        Qs = []
        for i in range(self.args.critics):
            Q, _ = self.critics[i](rgbds[:,:-1], comms_in[:,:-1], sensors[:,:-1], actions[:,1:], comms_out[:,1:], hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)))
            critic_loss = 0.5*F.mse_loss(Q*masks, Q_targets*masks)
            critic_losses.append(critic_loss)
            Qs.append(Q[0,0].item())
            if(self.args.train_together):
                complete_loss += self.args.critic_scaler * critic_loss
            else:
                self.critic_opts[i].zero_grad()
                critic_loss.backward()
                self.critic_opts[i].step()
        
            self.soft_update(self.critics[i], self.critic_targets[i], self.args.tau)
        
        torch.cuda.empty_cache()
        
        time = duration()
        if(self.args.show_duration): print("TRAINED CRITICS:", time - prev_time)
        prev_time = time
                                    
            
        
        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None:      alpha = self.alpha 
            else:                            alpha = self.args.alpha
            if self.args.alpha_text == None: alpha_text = self.alpha_text 
            else:                            alpha_text = self.args.alpha_text
            new_actions, new_comms_out, log_pis, log_pis_text, _ = self.actor(rgbds[:,:-1], comms_in[:,:-1], sensors[:,:-1], actions[:,:-1], comms_out[:,:-1], hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)), parenting)
            recommendation_value = calculate_similarity(recommended_actions, new_actions).unsqueeze(-1)
            intrinsic_imitation = -torch.mean((self.args.delta * recommendation_value)*masks).item() 
            
            if self.args.action_prior == "normal":
                loc = torch.zeros(self.args.action_shape, dtype=torch.float64).to(self.args.device).float()
                n = self.args.action_shape
                scale_tril = torch.tril(torch.ones(n, n)).to(self.args.device).float()
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_prrgbd = self.args.normal_alpha * policy_prior.log_prob(new_actions).unsqueeze(-1)
            elif self.args.action_prior == "uniform":
                policy_prior_log_prrgbd = 0.0
            intrinsic_entropy = torch.mean((alpha * log_pis - policy_prior_log_prrgbd)*masks).item()
                
            Qs = []
            for i in range(self.args.critics):
                Q, _ = self.critics[i](rgbds[:,:-1], comms_in[:,:-1], sensors[:,:-1], new_actions, new_comms_out, hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)))
                Qs.append(Q)
            Qs_stacked = torch.stack(Qs, dim=0)
            Q, _ = torch.min(Qs_stacked, dim=0)
            Q = Q.mean(-1).unsqueeze(-1)
            
            actor_loss = ((alpha * log_pis - policy_prior_log_prrgbd) + (alpha_text * log_pis_text) - (self.args.delta * recommendation_value) - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()
            
            time = duration()
            if(self.args.show_duration): print("USED ACTOR:", time - prev_time)
            prev_time = time
            
            if(self.args.train_together):
                complete_loss += self.args.actor_scaler * actor_loss 
            else:
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
            
            time = duration()
            if(self.args.show_duration): print("TRAINED ACTOR:", time - prev_time)
            prev_time = time
        else:
            Q = None
            intrinsic_entropy = None
            intrinsic_imitation = None
            actor_loss = None
            
        if(self.args.train_together):
            self.complete_opt.zero_grad()
            complete_loss.backward()
            self.complete_opt.step()
        
            
            
        # Train alpha
        if self.args.alpha == None:
            _, _, log_pis, _, _ = self.actor(rgbds[:,:-1], comms_in[:,:-1], sensors[:,:-1], actions[:,:-1], comms_out[:,:-1], hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)), parenting)
            alpha_loss = -(self.log_alpha.to(self.args.device) * (log_pis + self.target_entropy))*masks
            alpha_loss = alpha_loss.mean() / masks.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha.to(dtype=torch.float32)).to(self.args.device)
            torch.cuda.empty_cache()
        else:
            alpha_loss = None
            
        if self.args.alpha_text == None:
            _, _, _, log_pis_text, _ = self.actor(rgbds[:,:-1], comms_in[:,:-1], sensors[:,:-1], actions[:,:-1], comms_out[:,:-1], hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)), parenting)
            alpha_text_loss = -(self.log_alpha_text.to(self.args.device) * (log_pis_text + self.target_entropy_text))*masks
            alpha_text_loss = alpha_text_loss.mean() / masks.mean()
            self.alpha_text_opt.zero_grad()
            alpha_text_loss.backward()
            self.alpha_text_opt.step()
            self.alpha_text = torch.exp(self.log_alpha_text.to(dtype=torch.float32)).to(self.args.device)
            torch.cuda.empty_cache()
        else:
            alpha_text_loss = None
            
        time = duration()
        if(self.args.show_duration): print("TRAINED ALPHA:", time - prev_time)
        prev_time = time
                                
                                
                                
        if(accuracy != None):   accuracy = accuracy.item()
        if(rgbd_loss != None):   rgbd_loss = rgbd_loss.mean().item()
        if(comm_loss != None):   comm_loss = comm_loss.mean().item()
        if(sensors_loss != None):   sensors_loss = sensors_loss.mean().item()
        if(complexity != None): complexity = complexity.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(alpha_text_loss != None): alpha_text_loss = alpha_text_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(Q != None): Q = -Q.mean().item()
        for i in range(self.args.critics):
            if(critic_losses[i] != None): 
                critic_losses[i] = critic_losses[i].item()
                critic_losses[i] = log(critic_losses[i]) if critic_losses[i] > 0 else critic_losses[i]
                
        rgbd_prediction_error_curiosity = rgbd_prediction_error_curiosity.mean().item()
        comm_prediction_error_curiosity = comm_prediction_error_curiosity.mean().item()
        sensors_prediction_error_curiosity = sensors_prediction_error_curiosity.mean().item()
        
        rgbd_hidden_state_curiosity = rgbd_hidden_state_curiosity.mean().item()
        comm_hidden_state_curiosity = comm_hidden_state_curiosity.mean().item()
        sensors_hidden_state_curiosity = sensors_hidden_state_curiosity.mean().item()
        
        prediction_error_curiosity = prediction_error_curiosity.mean().item()
        hidden_state_curiosity = hidden_state_curiosity.mean().item()
        
        time = duration()
        #if(self.args.show_duration): print(f"WHOLE EPOCH {self.epochs}:", time - start_time)
                
        return(accuracy, rgbd_loss, comm_loss, sensors_loss, complexity, alpha_loss, alpha_text_loss, actor_loss, critic_losses, 
               extrinsic, Q, intrinsic_curiosity, intrinsic_entropy, intrinsic_imitation, 
               rgbd_prediction_error_curiosity, comm_prediction_error_curiosity, sensors_prediction_error_curiosity, prediction_error_curiosity,
               rgbd_hidden_state_curiosity, comm_hidden_state_curiosity, sensors_hidden_state_curiosity, hidden_state_curiosity)
    
    
                     
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