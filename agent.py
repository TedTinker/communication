#%%

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

from utils import default_args, duration, dkl, calculate_similarity, onehots_to_string, many_onehots_to_strings, action_to_string, cpu_memory_usage, action_map, action_name_list, custom_loss, print
from arena import Arena, get_physics
from task import Task, Task_Runner
from buffer import RecurrentReplayBuffer
from pvrnn import PVRNN
from models import Actor, Critic



class Agent:
    
    def __init__(self, i = -1, GUI = False, args = default_args):
        
        self.agent_num = i
        self.args = args
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        
        self.tasks = {
            "0" : Task(actions = [-1],              objects = 3, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2],      parent = True, args = self.args),
            "1" : Task(actions = [1],               objects = 2, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2],      parent = True, args = self.args),
            "2" : Task(actions = [1],               objects = 2, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2],      parent = True, args = self.args),
            "3" : Task(actions = [0, 1, 2, 3, 4],   objects = 2, colors = [0, 1, 2, 3, 4, 5],   shapes = [0, 1, 2],      parent = True, args = self.args)}
        physicsClient_1 = get_physics(GUI = GUI, time_step = self.args.time_step, steps_per_step = self.args.steps_per_step)
        self.arena_1 = Arena(physicsClient_1, args = self.args)
        physicsClient_2 = get_physics(GUI = False, time_step = self.args.time_step, steps_per_step = self.args.steps_per_step)
        self.arena_2 = Arena(physicsClient_2, args = self.args)
        self.task_runners = {task_name : Task_Runner(task, self.arena_1, self.arena_2, self.args) for i, (task_name, task) in enumerate(self.tasks.items())}
        self.task_name = self.args.task_list[0]
        
        self.target_entropy = self.args.target_entropy
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr, weight_decay = .00001) 
        
        self.target_entropy_text = self.args.target_entropy_text
        self.alpha_text = 1
        self.log_alpha_text = torch.tensor([0.0], requires_grad=True)
        self.alpha_text_opt = optim.Adam(params=[self.log_alpha_text], lr=self.args.alpha_text_lr, weight_decay = .00001) 

        self.forward = PVRNN(self.args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=self.args.forward_lr, weight_decay = .00001)
                           
        self.actor = Actor(self.args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr, weight_decay = .00001) 
        
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []
        for _ in range(self.args.critics):
            self.critics.append(Critic(self.args))
            self.critic_targets.append(Critic(self.args))
            self.critic_targets[-1].load_state_dict(self.critics[-1].state_dict())
            self.critic_opts.append(optim.Adam(self.critics[-1].parameters(), lr=self.args.critic_lr, weight_decay = .00001))
        
        self.memory = RecurrentReplayBuffer(self.args)
        
        self.plot_dict = {
            "args" : self.args,
            "arg_title" : args.arg_title,
            "arg_name" : args.arg_name,
            "episode_dicts" : {}, 
            "agent_lists" : {} if (self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list) else {"forward" : PVRNN(), "actor" : Actor(), "critic" : Critic()},
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
            "prediction_error" : [], 
            "hidden_state" : [[] for _ in range(self.args.layers)]}
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
            prev_task_name = self.task_name
            for i, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs == cumulative_epochs): 
                    
                    prev_time = duration()
                    
                    self.gen_test()  
                    self.save_episodes(swapping = False)
                    self.save_agent()
                    
                    self.task_name = self.args.task_list[i+1] 
                    
                    self.gen_test()  
                    self.save_episodes(swapping = True)
                    self.save_agent()
                    
                    time = duration()
                    if(self.args.show_duration): print("AFTER GEN, SAVE EPISODE, SAVE AGENT (START):", time - prev_time)
                    prev_time = time
                    break
                
            prev_time = duration()
                
            self.training_episode()
            
            time = duration()
            if(self.args.show_duration): print("\n\nTRAINING EPISODE:", time - prev_time)
            if(self.args.show_duration): print(f"{self.epochs} Epochs:", time - start_time)
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
                        prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1,
                        prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, sleep_time = None):
        start_time = duration()
        prev_time = duration()
        with torch.no_grad():
            self.eval()
            parented = self.task.task.parent
            rgbd_1, parent_comm, sensors_1 = self.task.obs()
            rgbd_2, _, sensors_2 = self.task.obs(agent_1 = False)
            recommended_action_1 = self.task.get_recommended_action()
            comm = parent_comm.unsqueeze(0) if parented else prev_comm_out_2
            #print(1)
            (_, _, hp_1), (_, _, hq_1), dkls_1 = self.forward.bottom_to_top_step(hq_1, self.forward.obs_in(rgbd_1, comm, sensors_1), self.forward.action_in(prev_action_1), self.forward.comm_in(prev_comm_out_1)) 
            #print(2)
            action_1, comm_out_1, _, _, ha_1 = self.actor(rgbd_1, parent_comm if parented else prev_comm_out_2, prev_action_1, prev_comm_out_1, hq_1[:,:,0].detach(), ha_1, parented) 
            values_1 = []
            new_hcs_1 = []
            for i in range(self.args.critics):
                value, hc = self.critics[i](rgbd_1, parent_comm if parented else prev_comm_out_2, action_1, comm_out_1, hq_1[:,:,0].detach(), hcs_1[i]) 
                values_1.append(round(value.item(), 3))
                new_hcs_1.append(hc)
            
            if(self.task.task.parent): 
                action_2 = torch.zeros_like(action_1)
                comm_out_2 = None
                hp_2 = None
                hq_2 = None
                values_2 = None
                new_hcs_2 = None
                dkls_2 = None
            else:
                recommended_action_2 = self.task.get_recommended_action(agent_1 = False)
                (_, _, hp_2), (_, _, hq_2), _, dkls_2 = self.forward.bottom_to_top(hq_2, self.forward.obs_in(rgbd_2, prev_comm_out_1, sensors_2), self.forward.action_in(prev_action_2), self.forward.comm_in(prev_comm_out_2))
                action_2, comm_out_2, _, _, ha_2 = self.actor(rgbd_2, prev_comm_out_1, prev_action_2, prev_comm_out_2, hq_2[:,:,0].detach(), ha_2, parented) 
                values_2 = []
                new_hcs_2 = []
                for i in range(self.args.critics):
                    value, hc = self.critics[i](rgbd_2, parent_comm if parented else prev_comm_out_1, action_2, comm_out_2, hq_2[:,:,0].detach(), hcs_2[i]) 
                    values_2.append(round(value.item(), 3))
                    new_hcs_2.append(hc)
                    
            """action_1 = torch.tensor([[[.5, -.5, 1]]])
            action_2 = torch.tensor([[[.5, -.5, 1]]])"""
                                    
            raw_reward, distance_reward, angle_reward, distance_reward_2, angle_reward_2, done, win = self.task.step(action_1[0,0].clone(), action_2[0,0].clone(), sleep_time = sleep_time)
            total_reward = raw_reward + distance_reward + angle_reward 
            total_reward_2 = raw_reward + distance_reward_2 + angle_reward_2
            next_rgbd_1, next_parent_comm, next_sensors_1 = self.task.obs()
            next_rgbd_2, _, next_sensors_2 = self.task.obs(agent_1 = False)
            
            to_push_1 = [
                rgbd_1,
                parent_comm if parented else prev_comm_out_2,
                sensors_1,
                action_1,
                comm_out_1,
                recommended_action_1,
                total_reward,
                next_rgbd_1,
                next_parent_comm if parented else comm_out_2,
                next_sensors_1,
                done]
            
            if(self.task.task.parent): 
                to_push_2 = None
            else:
                to_push_2 = [
                    rgbd_2,
                    prev_comm_out_1,
                    sensors_2,
                    action_2,
                    comm_out_2,
                    recommended_action_2,
                    total_reward,
                    next_rgbd_2,
                    comm_out_1,
                    next_sensors_2,
                    done]
        torch.cuda.empty_cache()
        
        time = duration()
        
        return(action_1, comm_out_1, values_1, hp_1.squeeze(1), hq_1.squeeze(1), ha_1, new_hcs_1, dkls_1,
               action_2, comm_out_2, values_2, None if hp_2 == None else hp_2.squeeze(1), None if hq_2 == None else hq_2.squeeze(1), ha_2, new_hcs_2, dkls_2,
               raw_reward, total_reward, distance_reward, angle_reward, total_reward_2, distance_reward_2, angle_reward_2, done, win, to_push_1, to_push_2)
            
           
    
    def training_episode(self):
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
        
        to_push_list_2 = []
        prev_action_2 = torch.zeros((1, 1, self.args.action_shape))
        prev_comm_out_2 = torch.zeros((1, 1, self.args.max_comm_len, self.args.comm_shape))
        hq_2 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
        ha_2 = torch.zeros((1, 1, self.args.hidden_size)) 
        hcs_2 = [torch.zeros((1, 1, self.args.hidden_size))] * self.args.critics
                    
        self.task = self.task_runners[self.task_name]
        self.task.begin()    
                        
        for step in range(self.args.max_steps):
            self.steps += 1                                                                                             
            if(not done):
                steps += 1
                prev_action_1, prev_comm_out_1, values_1, hp_1, hq_1, ha_1, hcs_1, dkls_1, \
                    prev_action_2, prev_comm_out_2, values_2, hp_2, hq_2, ha_2, hcs_2, dkls_2, \
                        raw_reward, total_reward, distance_reward, angle_reward, total_reward_2, distance_reward_2, angle_reward_2, done, win, to_push_1, to_push_2 = self.step_in_episode(
                            prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1,
                            prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2)
                        
                to_push_list_1.append(to_push_1)
                to_push_list_2.append(to_push_2)
                complete_reward += total_reward
            if(self.steps % self.args.steps_per_epoch == 0):
                prev_time = duration()
                plot_data = self.epoch(self.args.batch_size)
                time = duration()
                if(self.args.show_duration): print("EPOCH:", time - prev_time)
                prev_time = time
                if(plot_data == False): pass
                else:
                    accuracy, rgbd_loss, comm_loss, sensors_loss, complexity, \
                        alpha_loss, alpha_text_loss, actor_loss, critic_losses, \
                            e, q, ic, ie, ii, prediction_error, hidden_state = plot_data
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
                        self.plot_dict["prediction_error"].append(prediction_error)
                        for layer, f in enumerate(hidden_state):
                            self.plot_dict["hidden_state"][layer].append(f)    
                            
            time = duration()
            prev_time = time
                        
        self.task.done()
        self.plot_dict["steps"].append(steps)
        self.plot_dict["rewards"].append(complete_reward)
        goal_action = self.task.task.goal[0]
        win_dict_list = [self.plot_dict["wins_" + action_name.lower()] for action_name in action_name_list]
        for i, win_dict in enumerate(win_dict_list):
            if(i-1 == goal_action): 
                win_dict.append(win)
            else:                   win_dict.append(None)
                             
        for to_push in to_push_list_1:
            rgbd, comm_in, sensors, action, comm_out, recommended_action, total_reward, next_rgbd, next_comm_in, next_sensors, done = to_push
            self.memory.push(
                rgbd,
                comm_in, 
                sensors,
                action, 
                comm_out,
                recommended_action,
                total_reward, 
                next_rgbd,
                next_comm_in, 
                next_sensors,
                done)
            
        for to_push in to_push_list_2:
            if(to_push != None):
                rgbd, comm_in, sensors, action, comm_out, recommended_action, total_reward, next_rgbd, next_comm_in, next_sensors, done = to_push
                self.memory.push(
                    rgbd,
                    comm_in, 
                    sensors,
                    action, 
                    comm_out,
                    recommended_action,
                    total_reward, 
                    next_rgbd,
                    next_comm_in, 
                    next_sensors,
                    done)
                
        self.episodes += 1
        
        
        
    def gen_test(self, test = True, sleep_time = None, verbose = False):
        done = False
        complete_reward = 0
        
        prev_action_1 = torch.zeros((1, 1, self.args.action_shape))
        prev_comm_out_1 = torch.zeros((1, 1, self.args.max_comm_len, self.args.comm_shape))
        hq_1 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
        ha_1 = torch.zeros((1, 1, self.args.hidden_size)) 
        hcs_1 = [torch.zeros((1, 1, self.args.hidden_size))] * self.args.critics
        
        prev_action_2 = torch.zeros((1, 1, self.args.action_shape))
        prev_comm_out_2 = torch.zeros((1, 1, self.args.max_comm_len, self.args.comm_shape))
        hq_2 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
        ha_2 = torch.zeros((1, 1, self.args.hidden_size)) 
        hcs_2 = [torch.zeros((1, 1, self.args.hidden_size))] * self.args.critics
                
        try:
            self.task = self.task_runners[self.task_name]
            self.task.begin(test = test)        
            if(verbose):
                print(self.task.task.goal_human_text, end = " ")
            for step in range(self.args.max_steps):
                #print("Step", step)
                if(not done):
                    prev_action_1, prev_comm_out_1, values_1, hp_1, hq_1, ha_1, hcs_1, dkls_1, \
                        prev_action_2, prev_comm_out_2, values_2, hp_2, hq_2, ha_2, hcs_2, dkls_2,\
                            raw_reward, total_reward, distance_reward, angle_reward, total_reward_2, distance_reward_2, angle_reward_2, done, win, to_push_1, to_push_2 = self.step_in_episode(
                                prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1,
                                prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2, sleep_time)
                    complete_reward += total_reward
                #print("DONE")
            self.task.done()
            if(verbose):
                print("\tWIN!" if win else "\tLOSE!")
            goal_action = self.task.task.goal[0]
            win_dict_list = [self.plot_dict["gen_wins_" + action_name.lower()] for action_name in action_name_list]
            for i, win_dict in enumerate(win_dict_list):
                if(i-1 == goal_action): 
                    win_dict.append(win)
                else: win_dict.append(None)
        except:
            complete_reward = 0
            win = False
            win_dict_list = [self.plot_dict["gen_wins_" + action_name.lower()] for action_name in action_name_list]
            for i, win_dict in enumerate(win_dict_list):
                win_dict.append(None)
        self.plot_dict["gen_rewards"].append(complete_reward)
        return(win)
        
        
        
    def save_episodes(self, swapping = False):
        with torch.no_grad():
            self.task = self.task_runners[self.task_name]
            self.task.begin()       
            comm_from_parent = self.task.task.parent
            if(self.args.agents_per_episode_dict != -1 and self.agent_num > self.args.agents_per_episode_dict): 
                return
            for episode_num in range(self.args.episodes_in_episode_dict):
                episode_dict = {
                    "rgbds_1" : [],
                    "rgbds_2" : [],
                    "comms_in_1" : [],
                    "comms_in_2" : [],
                    "sensors_1" : [],
                    "sensors_2" : [],
                    "recommended_1" : [],
                    "recommended_2" : [],
                    "actions_1" : [],
                    "actions_2" : [],
                    "comms_out_1" : [],
                    "comms_out_2" : [],
                    "birds_eye_1" : [],
                    "birds_eye_2" : [],
                    "raw_rewards" : [],
                    "total_rewards_1" : [],
                    "distance_rewards_1" : [],
                    "angle_rewards_1" : [],
                    "total_rewards_2" : [],
                    "distance_rewards_2" : [],
                    "angle_rewards_2" : [],
                    "critic_predictions_1" : [],
                    "critic_predictions_2" : [],
                    "prior_predicted_rgbds_1" : [],
                    "prior_predicted_comms_in_1" : [],
                    "prior_predicted_sensors_1" : [],
                    "posterior_predicted_rgbds_1" : [],
                    "posterior_predicted_comms_in_1" : [],
                    "posterior_predicted_sensors_1" : [],
                    "prior_predicted_rgbds_2" : [],
                    "prior_predicted_comms_in_2" : [],
                    "prior_predicted_sensors_2" : [],
                    "posterior_predicted_rgbds_2" : [],
                    "posterior_predicted_comms_in_2" : [],
                    "posterior_predicted_sensors_1" : [],
                    "dkls_1" : [],
                    "dkls_2" : []}
                done = False
                
                hps_1 = []
                hqs_1 = []
                prev_action_1 = torch.zeros((1, 1, self.args.action_shape))
                prev_comm_out_1 = torch.zeros((1, 1, self.args.max_comm_len, self.args.comm_shape))
                hq_1 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
                ha_1 = torch.zeros((1, 1, self.args.hidden_size)) 
                hcs_1 = [torch.zeros((1, 1, self.args.hidden_size))] * self.args.critics
                
                hps_2 = []
                hqs_2 = []
                prev_action_2 = torch.zeros((1, 1, self.args.action_shape))
                prev_comm_out_2 = torch.zeros((1, 1, self.args.max_comm_len, self.args.comm_shape))
                hq_2 = torch.zeros((1, self.args.layers, self.args.pvrnn_mtrnn_size)) 
                ha_2 = torch.zeros((1, 1, self.args.hidden_size)) 
                hcs_2 = [torch.zeros((1, 1, self.args.hidden_size))] * self.args.critics
                 
                episode_dict["task"] = self.task.task
                episode_dict["goal"] = "{} ({})".format(self.task.task.goal_text, self.task.task.goal_human_text)
                rgbd_1, parent_comm, sensors_1 = self.task.obs()
                rgbd_2, _, sensors_2 = self.task.obs(agent_1 = False)
                birds_eye_1 = self.task.arena_1.photo_from_above()
                birds_eye_2 = None if comm_from_parent else self.task.arena_2.photo_from_above()
                episode_dict["rgbds_1"].append(rgbd_1[0,:,:,0:3])
                episode_dict["rgbds_2"].append(rgbd_2[0,:,:,0:3])        
                
                comm_in_1 = onehots_to_string(parent_comm[0] if comm_from_parent else prev_comm_out_2[0,0])
                episode_dict["comms_in_1"].append("{} ({})".format(comm_in_1, self.task.task.agent_to_english(comm_in_1)))
                comm_in_2 = onehots_to_string(prev_comm_out_1[0,0])
                episode_dict["comms_in_2"].append("{} ({})".format(comm_in_2, self.task.task.agent_to_english(comm_in_2)))
                
                episode_dict["sensors_1"].append(sensors_1.tolist())
                episode_dict["sensors_2"].append(None if sensors_2 == None else sensors_2.tolist())
                episode_dict["birds_eye_1"].append(birds_eye_1[:,:,0:3])
                episode_dict["birds_eye_2"].append(None if birds_eye_2 == None else birds_eye_2[:,:,0:3])
                
                for step in range(self.args.max_steps):
                    if(not done):
                        recommended_1 = self.task.get_recommended_action()
                        recommended_2 = self.task.get_recommended_action(agent_1 = False)
                        episode_dict["recommended_1"].append(action_to_string(recommended_1))
                        episode_dict["recommended_2"].append(None if recommended_2 == None else action_to_string(recommended_2))
                        prev_action_1, prev_comm_out_1, values_1, hp_1, hq_1, ha_1, hcs_1, dkls_1, \
                            prev_action_2, prev_comm_out_2, values_2, hp_2, hq_2, ha_2, hcs_2, dkls_2, \
                                raw_reward, total_reward, distance_reward, angle_reward, total_reward_2, distance_reward_2, angle_reward_2, done, win, to_push_1, to_push_2 = self.step_in_episode(
                                    prev_action_1, prev_comm_out_1, hq_1, ha_1, hcs_1,
                                    prev_action_2, prev_comm_out_2, hq_2, ha_2, hcs_2)
                        episode_dict["actions_1"].append(prev_action_1)
                        episode_dict["actions_2"].append(prev_action_2)
                        
                        comm_out_1 = onehots_to_string(prev_comm_out_1)
                        episode_dict["comms_out_1"].append("{} ({})".format(comm_out_1, self.task.task.agent_to_english(comm_out_1)))
                        comm_out_2 = onehots_to_string(prev_comm_out_2)
                        episode_dict["comms_out_2"].append("{} ({})".format(comm_out_2, self.task.task.agent_to_english(comm_out_2)))
                        
                        episode_dict["raw_rewards"].append(str(round(raw_reward, 3)))
                        episode_dict["total_rewards_1"].append(str(round(total_reward, 3)))
                        episode_dict["distance_rewards_1"].append(str(round(distance_reward, 3)))
                        episode_dict["angle_rewards_1"].append(str(round(angle_reward, 3)))
                        episode_dict["total_rewards_2"].append(str(round(total_reward_2, 3)))
                        episode_dict["distance_rewards_2"].append(str(round(distance_reward_2, 3)))
                        episode_dict["angle_rewards_2"].append(str(round(angle_reward_2, 3)))
                        episode_dict["critic_predictions_1"].append(values_1)
                        episode_dict["critic_predictions_2"].append(values_2)
                        episode_dict["dkls_1"].append([dkl.sum().item() for dkl in dkls_1])
                        episode_dict["dkls_2"].append(dkls_2 if dkls_2 == None else [dkl.sum().item() for dkl in dkls_2])
                        rgbd_1, parent_comm, sensors_1 = self.task.obs()
                        rgbd_2, _, sensors_2 = self.task.obs(agent_1 = False)
                        birds_eye_1 = self.task.arena_1.photo_from_above()
                        birds_eye_2 = None if comm_from_parent else self.task.arena_2.photo_from_above()
                        episode_dict["rgbds_1"].append(rgbd_1[0,:,:,0:3])
                        episode_dict["rgbds_2"].append(rgbd_2[0,:,:,0:3])
                        
                        comm_in_1 = onehots_to_string(parent_comm[0] if comm_from_parent else prev_comm_out_2[0,0])
                        episode_dict["comms_in_1"].append("{} ({})".format(comm_in_1, self.task.task.agent_to_english(comm_in_1)))
                        comm_in_2 = onehots_to_string(prev_comm_out_1[0,0])
                        episode_dict["comms_in_2"].append("{} ({})".format(comm_in_2, self.task.task.agent_to_english(comm_in_2)))
                        
                        episode_dict["sensors_1"].append(sensors_1.tolist())
                        episode_dict["sensors_2"].append(None if sensors_2 == None else sensors_2.tolist())
                        episode_dict["birds_eye_1"].append(birds_eye_1[:,:,0:3])
                        episode_dict["birds_eye_2"].append(None if birds_eye_2 == None else birds_eye_2[:,:,0:3])
                        hps_1.append(hp_1[:,0].unsqueeze(0))
                        hqs_1.append(hq_1[:,0].unsqueeze(0))
                        hps_2.append(hp_2 if hp_2 == None else hp_2[:,0].unsqueeze(0))
                        hqs_2.append(hq_2 if hq_2 == None else hq_2[:,0].unsqueeze(0))
                self.task.done()
                        
                hp_1 = torch.cat(hps_1, dim = 1)
                hq_1 = torch.cat(hqs_1, dim = 1)
                actions_1 = torch.cat(episode_dict["actions_1"], dim = 1)
                
                comm_out_1 = onehots_to_string(prev_comm_out_1)
                episode_dict["comms_out_1"].append("{} ({})".format(comm_out_1, self.task.task.agent_to_english(comm_out_1)))
                
                episode_dict["actions_1"] = [action_to_string(action) for action in episode_dict["actions_1"]]
                pred_rgbds_p, pred_comm_in_p, pred_sensors_p = self.forward.predict(hp_1, self.forward.action_in(actions_1)) 
                pred_rgbds_q, pred_comm_in_q, pred_sensors_q = self.forward.predict(hq_1, self.forward.action_in(actions_1))
                for step in range(pred_rgbds_p.shape[1]):
                    episode_dict["prior_predicted_rgbds_1"].append(pred_rgbds_p[0,step][:,:,0:3])
                    
                    prior_predicted_comms_in_1 = onehots_to_string(pred_comm_in_p[0,step])
                    episode_dict["prior_predicted_comms_in_1"].append("{} ({})".format(prior_predicted_comms_in_1, self.task.task.agent_to_english(prior_predicted_comms_in_1)))
                    
                    episode_dict["prior_predicted_sensors_1"].append([round(o.item(), 2) for o in pred_sensors_p[0,step]])
                    episode_dict["posterior_predicted_rgbds_1"].append(pred_rgbds_q[0,step][:,:,0:3])
                    
                    posterior_predicted_comms_in_1 = onehots_to_string(pred_comm_in_q[0,step])
                    episode_dict["posterior_predicted_comms_in_1"].append("{} ({})".format(posterior_predicted_comms_in_1, self.task.task.agent_to_english(posterior_predicted_comms_in_1)))
                    
                    episode_dict["posterior_predicted_sensors_1"].append([round(o.item(), 2) for o in pred_sensors_q[0,step]])
                #if(self.agent_num == 1):
                #    for step in range(hp_1.shape[1]):
                #        print("\nIn Saving Episode:")
                #        print("Real: ", episode_dict["comms_1"][step])
                #        print("Prior:", episode_dict["prior_predicted_comms_1"][step])
                #        print("Postr:", episode_dict["posterior_predicted_comms_1"][step])
                #        print("\n")
                
                if(not comm_from_parent):
                    hp_2 = torch.cat(hps_2, dim = 1)
                    hq_2 = torch.cat(hqs_2, dim = 1)
                    actions_2 = torch.cat(episode_dict["actions_2"], dim = 1)
                    
                    comm_out_2 = onehots_to_string(prev_comm_out_1)
                    episode_dict["comms_out_2"].append("{} ({})".format(comm_out_2, self.task.task.agent_to_english(comm_out_2)))
                    
                    episode_dict["actions_2"] = [action_to_string(action) for action in episode_dict["actions_2"]]
                    pred_rgbds_p, pred_comm_in_p, pred_sensors_p = self.forward.predict(hp_2, self.forward.action_in(actions_2))
                    pred_rgbds_q, pred_comm_in_q, pred_sensors_q = self.forward.predict(hq_2, self.forward.action_in(actions_2))
                    for step in range(pred_rgbds_p.shape[1]):
                        episode_dict["prior_predicted_rgbds_2"].append(pred_rgbds_p[0,step][:,:,0:3])

                        prior_predicted_comms_in_2 = onehots_to_string(pred_comm_in_p[0,step])
                        episode_dict["prior_predicted_comms_in_2"].append("{} ({})".format(prior_predicted_comms_in_2, self.task.task.agent_to_english(prior_predicted_comms_in_2)))
                        
                        episode_dict["prior_predicted_sensors_2"].append([round(o.item(), 2) for o in pred_sensors_p[0,step]])
                        episode_dict["posterior_predicted_rgbds_2"].append(pred_rgbds_q[0,step][:,:,0:3])
                        
                        posterior_predicted_comms_in_2 = onehots_to_string(pred_comm_in_q[0,step])
                        episode_dict["posterior_predicted_comms_in_2"].append("{} ({})".format(posterior_predicted_comms_in_2, self.task.task.agent_to_english(posterior_predicted_comms_in_2)))
                        
                        episode_dict["posterior_predicted_sensors_2"].append([round(o.item(), 2) for o in pred_sensors_q[0,step]])
                    #if(self.agent_num == 1):
                    #    for step in range(hp_2.shape[1]):
                    #        print("\nIn Saving Episode:")
                    #        print("Real: ", episode_dict["comms_2"][step])
                    #        print("Prior:", episode_dict["prior_predicted_comms_2"][step])
                    #        print("Postr:", episode_dict["posterior_predicted_comms_2"][step])
                    #        print("\n")
                
                self.plot_dict["episode_dicts"]["{}_{}_{}_{}".format(self.agent_num, self.epochs, episode_num, 1 if swapping else 0)] = episode_dict
        
        
        
    def save_agent(self):
        if(self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list): return
        self.plot_dict["agent_lists"]["{}_{}".format(self.agent_num, self.epochs)] = deepcopy(self.state_dict())
    
    
    
    def epoch(self, batch_size):
        self.train()
        parented = self.task.task.parent
                                
        batch = self.memory.sample(batch_size)
        if(batch == False): return(False)
        
        prev_time = duration()
                        
        self.epochs += 1

        rgbds, comms_in, sensors, actions, comms_out, recommended_actions, rewards, dones, masks = batch
        rgbds = torch.from_numpy(rgbds)
        comms_in = torch.from_numpy(comms_in)
        sensors = torch.from_numpy(sensors)
        actions = torch.from_numpy(actions)
        comms_out = torch.from_numpy(comms_out)
        recommended_actions = torch.from_numpy(recommended_actions)
        rewards = torch.from_numpy(rewards)
        dones = torch.from_numpy(dones)
        masks = torch.from_numpy(masks)
        actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape).to(self.args.device), actions], dim = 1)
        comms_out = torch.cat([torch.zeros(comms_out[:,0].unsqueeze(1).shape).to(self.args.device), comms_out], dim = 1)
        all_masks = torch.cat([torch.ones(masks.shape[0], 1, 1).to(self.args.device), masks], dim = 1)   
        episodes = rewards.shape[0]
        steps = rewards.shape[1]
        
        time = duration()
        start_time = duration()
        prev_time = time
        
        #print("\n\n")
        #print("Agent {}, epoch {}. rgbds: {}. comms in: {}. actions: {}. comms out: {}. recommended actions: {}. rewards: {}. dones: {}. masks: {}.".format(
        #    self.agent_num, self.epochs, rgbds.shape, comms_in.shape, actions.shape, comms_out.shape, recommended_actions.shape, rewards.shape, dones.shape, masks.shape))
        #print("\n\n")
        
                
        
        # Train forward
        (zp_mu, zp_std, hps), (zq_mu, zq_std, hqs), (pred_rgbds, pred_comms, pred_sensors), dkls = self.forward(torch.zeros((episodes, self.args.layers, self.args.pvrnn_mtrnn_size)), rgbds, comms_in, sensors, actions, comms_out)
        hqs = hqs[:,:,0]
        
        rgbd_loss = F.binary_cross_entropy(pred_rgbds, rgbds[:,1:], reduction = "none").mean((-1,-2,-3)).unsqueeze(-1) * masks * self.args.rgbd_scaler
                        
        real_comms = comms_in[:,1:].reshape((episodes * steps, self.args.max_comm_len, self.args.comm_shape))
        real_comms = torch.argmax(real_comms, dim = -1)
        pred_comms = pred_comms.reshape((pred_comms.shape[0] * pred_comms.shape[1], self.args.max_comm_len, self.args.comm_shape))
        pred_comms = pred_comms.transpose(1,2)
    
        #comm_loss = custom_loss(pred_comms, real_comms, max_shift = 0)    
        comm_loss = F.cross_entropy(pred_comms, real_comms, reduction = "none")
        comm_loss = comm_loss.reshape(episodes, steps, self.args.max_comm_len)
        comm_loss = comm_loss.mean(dim=2).unsqueeze(-1) * masks * self.args.comm_scaler
        
        sensors_loss = F.binary_cross_entropy(pred_sensors, sensors[:,1:], reduction = "none")
        sensors_loss = sensors_loss.mean(-1).unsqueeze(-1) * masks * self.args.sensors_scaler
        
        accuracy_for_prediction_error = rgbd_loss + comm_loss + sensors_loss
        accuracy = accuracy_for_prediction_error.mean()
        
        complexity_for_hidden_state = [dkl(zq_mu[:,:,layer], zq_std[:,:,layer], zp_mu[:,:,layer], zp_std[:,:,layer]).mean(-1).unsqueeze(-1) * all_masks for layer in range(self.args.layers)] 
        complexity          = sum([self.args.beta[layer] * complexity_for_hidden_state[layer].mean() for layer in range(self.args.layers)])       
        complexity_for_hidden_state = [layer[:,1:] for layer in complexity_for_hidden_state] 
                                
        time = duration()
        if(self.args.show_duration): print("USED FORWARD:", time - prev_time)
        prev_time = time
                                
        self.forward_opt.zero_grad()
        (accuracy + complexity).backward()
        self.forward_opt.step()
        
        if(self.args.beta == 0): complexity = None
        torch.cuda.empty_cache()
        
        time = duration()
        if(self.args.show_duration): print("TRAINED FORWARD:", time - prev_time)
        prev_time = time
                        
                        
        
        # Get curiosity                  
        #complexity_for_hidden_state = [torch.sigmoid(c) for c in complexity_for_hidden_state]
        complexity_for_hidden_state = [torch.clamp(c, min = 0, max = self.args.dkl_max) for c in complexity_for_hidden_state]  # Or tanh? sigmoid? Or just clamp?
        prediction_error_curiosity = accuracy_for_prediction_error * (self.args.prediction_error_eta if self.args.prediction_error_eta != None else self.prediction_error_eta)
        hidden_state_curiosities = [complexity_for_hidden_state[layer] * (self.args.hidden_state_eta[layer] if self.args.hidden_state_eta[layer] != None else self.hidden_state_eta[layer]) for layer in range(self.args.layers)]
        hidden_state_curiosity = sum(hidden_state_curiosities)
        if(self.args.curiosity == "prediction_error"):  curiosity = prediction_error_curiosity
        elif(self.args.curiosity == "hidden_state"): curiosity = hidden_state_curiosity
        else:                                curiosity = torch.zeros(rewards.shape).to(self.args.device)
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        rewards += curiosity
        
        time = duration()
        if(self.args.show_duration): print("CURIOSITY:", time - prev_time)
        prev_time = time
                        
        
                
        # Train critics
        with torch.no_grad():
            new_actions, new_comms_out, log_pis_next, log_pis_next_text, _ = self.actor(rgbds, comms_in, actions, comms_out, hqs.detach(), torch.zeros((episodes, steps, self.args.hidden_size)), parented)
            Q_target_nexts = []
            for i in range(self.args.critics):
                Q_target_next, _ = self.critic_targets[i](rgbds, comms_in, new_actions, new_comms_out, hqs.detach(), torch.zeros((episodes, steps, self.args.hidden_size)))
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
            Q, _ = self.critics[i](rgbds[:,:-1], comms_in[:,:-1], actions[:,1:], comms_out[:,1:], hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)))
            critic_loss = 0.5*F.mse_loss(Q*masks, Q_targets*masks)
            critic_losses.append(critic_loss)
            Qs.append(Q[0,0].item())
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
        
            self.soft_update(self.critics[i], self.critic_targets[i], self.args.tau)
        
        torch.cuda.empty_cache()
        
        time = duration()
        if(self.args.show_duration): print("TRAINED CRITICS:", time - prev_time)
        prev_time = time
                                
        
        
        # Train alpha
        if self.args.alpha == None:
            _, _, log_pis, _, _ = self.actor(rgbds[:,:-1], comms_in[:,:-1], actions[:,:-1], comms_out[:,:-1], hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)), parented)
            alpha_loss = -(self.log_alpha.to(self.args.device) * (log_pis + self.target_entropy))*masks
            alpha_loss = alpha_loss.mean() / masks.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha).to(self.args.device)
            torch.cuda.empty_cache()
        else:
            alpha_loss = None
            
        if self.args.alpha_text == None:
            _, _, _, log_pis_text, _ = self.actor(rgbds[:,:-1], comms_in[:,:-1], actions[:,:-1], comms_out[:,:-1], hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)), parented)
            alpha_text_loss = -(self.log_alpha_text.to(self.args.device) * (log_pis_text + self.target_entropy_text))*masks
            alpha_text_loss = alpha_text_loss.mean() / masks.mean()
            self.alpha_text_opt.zero_grad()
            alpha_text_loss.backward()
            self.alpha_text_opt.step()
            self.alpha_text = torch.exp(self.log_alpha_text).to(self.args.device)
            torch.cuda.empty_cache()
        else:
            alpha_text_loss = None
            
        time = duration()
        if(self.args.show_duration): print("TRAINED ALPHA:", time - prev_time)
        prev_time = time
                                    
            
        
        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None:      alpha = self.alpha 
            else:                            alpha = self.args.alpha
            if self.args.alpha_text == None: alpha_text = self.alpha_text 
            else:                            alpha_text = self.args.alpha_text
            new_actions, new_comms_out, log_pis, log_pis_text, _ = self.actor(rgbds[:,:-1], comms_in[:,:-1], actions[:,:-1], comms_out[:,:-1], hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)), parented)
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
                Q, _ = self.critics[i](rgbds[:,:-1], comms_in[:,:-1], new_actions, new_comms_out, hqs[:,:-1].detach(), torch.zeros((episodes, steps, self.args.hidden_size)))
                Qs.append(Q)
            Qs_stacked = torch.stack(Qs, dim=0)
            Q, _ = torch.min(Qs_stacked, dim=0)
            Q = Q.mean(-1).unsqueeze(-1)
            
            actor_loss = ((alpha * log_pis - policy_prior_log_prrgbd) + (alpha_text * log_pis_text) - (self.args.delta * recommendation_value) - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()
            
            time = duration()
            if(self.args.show_duration): print("USED ACTOR:", time - prev_time)
            prev_time = time

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
        
        prediction_error_curiosity = prediction_error_curiosity.mean().item()
        hidden_state_curiosities = [hidden_state_curiosity.mean().item() for hidden_state_curiosity in hidden_state_curiosities]
        #hidden_state_curiosities = [hidden_state_curiosity for hidden_state_curiosity in hidden_state_curiosities]
        
        time = duration()
        #if(self.args.show_duration): print(f"WHOLE EPOCH {self.epochs}:", time - start_time)
                
        return(accuracy, rgbd_loss, comm_loss, sensors_loss, complexity, alpha_loss, alpha_text_loss, actor_loss, critic_losses, 
               extrinsic, Q, intrinsic_curiosity, intrinsic_entropy, intrinsic_imitation, prediction_error_curiosity, hidden_state_curiosities)
    
    
                     
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