#%%

import os 
import psutil
from time import sleep
import numpy as np
from math import log
from itertools import accumulate, product
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import zipfile
import gzip
import sys
from collections.abc import Mapping, Container

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

from utils import folder, wheels_joints_to_string, cpu_memory_usage, duration, print_duration, wait_for_button_press, \
    task_map, color_map, shape_map, task_name_list, print, To_Push, empty_goal, rolling_average, Obs, Action, get_goal_from_one_hots, Goal, adjust_action, testing_combos
from utils_submodule import model_start
from arena import Arena, get_physics
from processor import Processor
from buffer import RecurrentReplayBuffer
from pvrnn import PVRNN
from models import Actor, Critic
from plotting_episodes import plot_step
from plotting_for_video import plot_video_step



def print_cpu_usage(string, num):
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_affinity = process.cpu_affinity()
    print(f"{string}: {num} CPU affinity: {cpu_affinity}")
    current_cpu = psutil.cpu_percent(interval=1, percpu=True)
    print(f"{string}: {num} Current CPU usage per core: {current_cpu}")
    
def get_uniform_weight(first_weight, num_weights):
    remaining_sum = 100 - first_weight
    uniform_weight = remaining_sum / (num_weights - 1)
    return uniform_weight

def make_tasks_and_weights(first_weight):
    u = get_uniform_weight(first_weight, 6)
    return([(0, first_weight)] + [(v, u) for v in [1, 2, 3, 4, 5]])

def sizeof_fmt(num, suffix="B"):
    """Convert bytes to human-readable KB, MB, GB, etc."""
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f} P{suffix}"

fwpulr_tasks_and_weights = make_tasks_and_weights(50)



class Agent:
    
    def __init__(self, args, i = -1, GUI = False):
        
        self.agent_num = i
        self.args = args
        self.agent_name = f"{self.args.arg_name}_{self.agent_num}"
        
        self.total_steps = 0
        self.total_episodes = 0
        self.total_epochs = 0
        
        self.reward_inflation = 0
        if(self.args.reward_inflation_type == "None"):
            self.reward_inflation = 1
        self.hidden_state_eta_report_voice_reduction = 1
        
        if self.args.device.type == "cuda":
            print(f"\nIN AGENT: {i} DEVICE: {self.args.device} ({torch.cuda.current_device()} out of {[j for j in range(torch.cuda.device_count())]}, {torch.cuda.get_device_name(torch.cuda.current_device())})\n")
        else:
            print(f"\nIN AGENT: {i} DEVICE: {self.args.device}\n")
            
        self.start_physics(GUI) 
        
        #os.sched_setaffinity(0, {self.args.cpu})
        
        self.processors = {
            "all" :       Processor(
                self.args, self.arena_1, self.arena_2, 
                tasks_and_weights = [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],     
                objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0, 1, 2, 3, 4], parenting = True, full_name = "All Tasks")}

        self.all_processors = {f"{task_map[task].name}_{color_map[color].name}_{shape_map[shape].name}" : 
            Processor(self.args, self.arena_1, self.arena_2, tasks_and_weights = [(task, 1)], objects = 2, colors = [color], shapes = [shape], parenting = True) for task, color, shape in \
                product([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4])}
                #product([1, 2, 5, 6], [0, 5], [0, 4])}
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
        
        self.plot_dict = {
            "args" : self.args,
            "arg_title" : self.args.arg_title,
            "arg_name" : self.args.arg_name,
            "all_processor_names" : self.all_processor_names,
            "testing_combos" : testing_combos,
            
            "division_epochs" : [],
            "steps" : [],
            
            "behavior" : {},
            "composition_data" : {},
            
            "accuracy_loss" : [], 
            "complexity_loss" : [],
            "vision_loss" : [], 
            "touch_loss" : [], 
            "command_voice_loss" : [], 
            "report_voice_loss" : [], 
            
            "actor_loss" : [], 
            "critics_loss" : [[] for _ in range(self.args.critics)], 
            
            "alpha_loss" : [], 
            "alpha_text_loss" : [],
        
            "reward" : [], 
            "gen_reward" : [], 
            "q" : [], 
            "extrinsic" : [], 
            
            "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            
            "vision_prediction_error_curiosity" : [], 
            "touch_prediction_error_curiosity" : [], 
            "command_voice_prediction_error_curiosity" : [], 
            "report_voice_prediction_error_curiosity" : [], 
            "prediction_error_curiosity" : [], 
            
            "vision_hidden_state_curiosity" : [],
            "touch_hidden_state_curiosity" : [],
            "command_voice_hidden_state_curiosity" : [],
            "report_voice_hidden_state_curiosity" : [],
            "hidden_state_curiosity" : [],

            "wins_all" : [],
            "gen_wins_all" : []}
        for t in task_map.values():
            self.plot_dict[f"wins_{t.name}"] = []
            self.plot_dict[f"gen_wins_{t.name}"] = []
            
            
            
    def start_physics(self, GUI = False):
        self.steps = 0
        self.episodes = 0 
        self.epochs = 0 
        physicsClient_1 = get_physics(GUI = GUI, args = self.args)
        self.arena_1 = Arena(physicsClient_1, args = self.args)
        physicsClient_2 = get_physics(GUI = False, args = self.args)
        self.arena_2 = Arena(physicsClient_2, args = self.args)
        self.processor_name = self.args.processor
        
    def give_actor_voice(self):
        self.actor.voice_out.load_state_dict(self.forward.predict_obs.command_voice_out.state_dict())
        
        
        
    def regular_checks(self, force = False, swapping = False, sleep_time = None):
        if(self.args.save_agents): # Most problematic, roughly in order: behavior, compositions, memory, forward, forward_opt.
            if(
                (self.agent_num <= self.args.agents_per_agent_save and self.epochs % self.args.epochs_per_agent_save == 0) or 
                (self.agent_num <= self.args.agents_per_agent_save and force)):
                self.save_agent()
        if(self.args.save_compositions):
            if(self.epochs % self.args.epochs_per_composition_data == 0 or force):
                self.get_composition_data()
        if(self.epochs % self.args.epochs_per_gen_test == 0 or force):
            self.gen_test(sleep_time = sleep_time)  
        
        
        
    def training(self, q = None, sleep_time = None):      
        self.regular_checks(force = True, sleep_time = sleep_time)
        while(True):                
            self.training_episode(sleep_time = sleep_time)
            if(self.check_ping()):
                self.save_dicts()
            percent_done = str(self.epochs / self.args.epochs)
            if(q != None):
                q.put((self.agent_num, percent_done))
            if(self.epochs >= self.args.epochs): 
                linestyle = self.processors[self.processor_name].linestyle
                full_name = self.processors[self.processor_name].full_name
                self.plot_dict["division_epochs"].append((self.total_epochs, linestyle, full_name))
                break
            self.regular_checks(sleep_time = sleep_time)
        self.regular_checks(force = True)
        self.save_dicts(final = True)
        
        
        
    def check_ping(self):
        file_path = os.path.join(folder, self.agent_name)
        if(os.path.isfile(file_path)):
            os.remove(file_path)
            return(True)
        return(False)
        
        
        
    def save_dicts(self, final = False):
        
        self.plot_dict["accumulated_reward"] = list(accumulate(self.plot_dict["reward"]))
        self.plot_dict["accumulated_gen_reward"] = list(accumulate(self.plot_dict["gen_reward"]))
        
        for task_name in task_name_list + ["all"]:
            self.plot_dict["rolled_wins_" + task_name] = rolling_average(self.plot_dict["wins_" + task_name])
            self.plot_dict["rolled_gen_wins_" + task_name] = rolling_average(self.plot_dict["gen_wins_" + task_name])
            
        self.min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in self.min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "all_processor_names", "composition_data", "episode_dicts", "agent_lists", "spot_names", "steps", "behavior"]):
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
                    
        file_end = str(self.agent_num).zfill(3)
        if(not final):
            file_end = f"temp_{file_end}"
                
        with open(f"{folder}/plot_dict_{file_end}.pickle", "wb") as handle:
            pickle.dump(self.plot_dict, handle)
        with open(f"{folder}/min_max_dict_{file_end}.pickle", "wb") as handle:
            pickle.dump(self.min_max_dict, handle)
                
    
    
    def get_agent_obs(self, agent_1 = True):
        parenting = self.processor.parenting
        if(parenting and not agent_1):
            return(None)
        obs = self.processor.obs(agent_1)
        obs.command_voice = obs.command_voice.one_hots.unsqueeze(0).unsqueeze(0) 
        obs.report_voice = obs.report_voice.one_hots.unsqueeze(0).unsqueeze(0) 
        return(obs)
                
                
    
    def step_in_episode(self, 
                        prev_action_1, hq_1, obs_1,
                        prev_action_2, hq_2, obs_2, verbose = False, sleep_time = None, user_action = False):

        with torch.no_grad():
            self.eval()
            parenting = self.processor.parenting
                        
                        
                        
            def agent_step(agent_1 = True):
                
                if(parenting and not agent_1):
                    return(None, None, hq_2, hq_2, None, None, None, None, None)
                                
                prev_action = prev_action_1 if agent_1 else prev_action_2
                partner_prev_voice_out = prev_action_2.voice_out if (agent_1 and not parenting) else torch.zeros((1, 1, self.args.max_voice_len, self.args.voice_shape)) if agent_1 else prev_action_1.voice_out
                hq = hq_1 if agent_1 else hq_2
                obs = obs_1 if agent_1 else obs_2
                
                if(type(obs.command_voice) == Goal):
                    obs.command_voice = obs.command_voice.one_hots
                if(type(obs.report_voice) == Goal):
                    obs.report_voice = obs.report_voice.one_hots
                
                hp, hq, vision_is, touch_is, command_voice_is, report_voice_is = self.forward.bottom_to_top_step(
                    hq_1, self.forward.obs_in(obs), self.forward.action_in(prev_action))

                action, _, _ = self.actor(hq.detach(), parenting) 
                
                if(user_action):
                    user_wheels_joints = adjust_action(action.wheels_joints)
                    action.wheels_joints = user_wheels_joints
                
                values = []
                for i in range(self.args.critics):
                    value = self.critics[i](action, hq.detach()) 
                    values.append(round(value.item(), 3))
                
                return(obs, action, hp, hq, values, vision_is, touch_is, command_voice_is, report_voice_is)
            
            
            
            obs_1, action_1, hp_1, hq_1, values_1, vision_is_1, touch_is_1, command_voice_is_1, report_voice_is_1 = agent_step()
            obs_2, action_2, hp_2, hq_2, values_2, vision_is_2, touch_is_2, command_voice_is_2, report_voice_is_2 = agent_step(agent_1 = False)

            reward, done, win = self.processor.step(action_1.wheels_joints[0,0].clone(), None if action_2 == None else action_2.wheels_joints[0,0].clone(), sleep_time = sleep_time, verbose = verbose)
            
            reward *= self.reward_inflation
            
            def next_agent_step(agent_1 = True):
                
                if(parenting and not agent_1):
                    return(None, None)
                
                next_obs = self.processor.obs(agent_1)
                obs = obs_1 if agent_1 else obs_2 
                action = action_1 if agent_1 else action_2
                partner_voice_out = None if parenting else action_2.voice_out if agent_1 else action_1.voice_out
                
                next_obs.command_voice = next_obs.command_voice.one_hots.unsqueeze(0).unsqueeze(0)
                next_obs.report_voice = next_obs.report_voice.one_hots.unsqueeze(0).unsqueeze(0)

                to_push = To_Push(obs, action, reward, next_obs, done)     
                return(next_obs, to_push)
            
            next_obs_1, to_push_1 = next_agent_step()
            next_obs_2, to_push_2 = next_agent_step(agent_1 = False)
            
        torch.cuda.empty_cache()
        
        return(action_1, values_1, hp_1.squeeze(1), hq_1.squeeze(1), vision_is_1, touch_is_1, command_voice_is_1, report_voice_is_1,
               action_2, values_2, hp_2.squeeze(1), hq_2.squeeze(1), vision_is_2, touch_is_2, command_voice_is_2, report_voice_is_2,
               reward, done, win, to_push_1, to_push_2)
            
           
           
    def start_episode(self):
        done = False
        complete_reward = 0
        steps = 0
        
        def start_agent(agent_1 = True):
            to_push_list = []
            prev_action = Action(torch.zeros((1, 1, self.args.wheels_joints_shape)), torch.zeros((1, 1, self.args.max_voice_len, self.args.voice_shape)))
            hq = torch.zeros((1, 1, self.args.pvrnn_mtrnn_size)) 
            return(to_push_list, prev_action, hq)
                
        return(done, complete_reward, steps, start_agent(), start_agent(agent_1 = False))
           
           
    
    def training_episode(self, sleep_time = None):        
        done, complete_reward, steps, \
            (to_push_list_1, prev_action_1, hq_1), \
            (to_push_list_2, prev_action_2, hq_2) = self.start_episode()
                    
        start_time = duration()
                    
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
                obs_1 = self.get_agent_obs()
                obs_2 = self.get_agent_obs(agent_1 = False)
                prev_action_1, values_1, hp_1, hq_1, vision_is_1, touch_is_1, command_voice_is_1, report_voice_is_1, \
                    prev_action_2, values_2, hp_2, hq_2, vision_is_2, touch_is_2, command_voice_is_2, report_voice_is_2, \
                        reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                            prev_action_1, hq_1, obs_1,
                            prev_action_2, hq_2, obs_2, sleep_time = sleep_time)
                        
                if(self.args.save_behaviors):
                    if(self.args.agents_per_behavior_analysis == -1 or self.agent_num <= self.args.agents_per_behavior_analysis):  
                        if(self.episodes == 0 or self.episodes % self.args.episodes_per_behavior_analysis == 0):
                            self.plot_dict["behavior"][self.episodes].append(get_goal_from_one_hots(to_push_1.next_obs.report_voice)) 
                    
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
        
        percent_done = self.epochs / self.args.epochs
        
        if(self.args.hidden_state_eta_report_voice_reduction_type == "linear"):
            self.hidden_state_eta_report_voice_reduction = 1 - percent_done
        if(self.args.hidden_state_eta_report_voice_reduction_type.startswith("exp")):
            exp = float(self.args.hidden_state_eta_report_voice_reduction_type.split("_")[-1])
            self.hidden_state_eta_report_voice_reduction = 1 - (percent_done ** exp)
        if(self.args.hidden_state_eta_report_voice_reduction_type.startswith("sigmoid")):
            k = float(self.args.hidden_state_eta_report_voice_reduction_type.split("_")[-1])
            self.hidden_state_eta_report_voice_reduction = 1 - (1 / (1 + np.exp(-k * (self.epochs - self.args.epochs/2))))
            
        if(self.args.reward_inflation_type == "linear"):
            self.reward_inflation = percent_done
        if(self.args.reward_inflation_type.startswith("exp")):
            exp = float(self.args.reward_inflation_type.split("_")[-1])
            self.reward_inflation = percent_done ** exp
        if(self.args.reward_inflation_type.startswith("sigmoid")):
            k = float(self.args.reward_inflation_type.split("_")[-1])
            self.reward_inflation = (1 / (1 + np.exp(-k * (self.epochs - self.args.epochs/2))))
                        
        end_time = duration()
        print_duration(start_time, end_time, "\nTraining episode", "\n")
                        
        return(step)
        
        
        
    def gen_test(self, sleep_time = None):
        done, complete_reward, steps, \
            (to_push_list_1, prev_action_1, hq_1), \
            (to_push_list_2, prev_action_2, hq_2) = self.start_episode()
                
        try:
            self.processor = self.processors[self.processor_name]
            self.processor.begin(test = True)        
            for step in range(self.args.max_steps):
                #print("Step", step)
                if(not done):
                    obs_1 = self.get_agent_obs()
                    obs_2 = self.get_agent_obs(agent_1 = False)
                    prev_action_1, values_1, hp_1, hq_1, vision_is_1, touch_is_1, command_voice_is_1, report_voice_is_1, \
                        prev_action_2, values_2, hp_2, hq_2, vision_is_2, touch_is_2, command_voice_is_2, report_voice_is_2, \
                            reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                                prev_action_1, hq_1, obs_1,
                                prev_action_2, hq_2, obs_2, sleep_time = sleep_time)
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
        
        
        
    def save_episodes(self, test = False, verbose = False, display = True, video_display = True, sleep_time = None, waiting = False, user_action = False, dreaming = False):        
        with torch.no_grad():
            self.processor = self.processors[self.processor_name]
            self.processor.begin(test = test)       
            parenting = self.processor.parenting

            common_keys = [
                "obs", "action", 
                "birds_eye", "reward", "critic_predictions", "prior_predictions", "posterior_predictions", 
                "vision_dkl", "touch_dkl", "command_voice_dkl", "report_voice_dkl"]
            episode_dict = {}
            for agent_id in [0, 1]:
                for key in common_keys:
                    episode_dict[f"{key}_{agent_id}"] = []
            episode_dict["reward"] = []
            episode_dict["processor"] = self.processor
            episode_dict["goal"] = self.processor.goal
            
            done = False
            
            done, complete_reward, steps, \
                (to_push_list_1, prev_action_1, hq_1), \
                (to_push_list_2, prev_action_2, hq_2) = self.start_episode()
                    
            hp_1 = deepcopy(hq_1)
            hp_2 = deepcopy(hp_1)
            
            
            
            # In dream-time, this should be the observation.
            previous_dream_obs_q_1 = None
            previous_dream_obs_q_2 = None
            current_dream_obs_q_1 = None
            current_dream_obs_q_2 = None
            
            
                            
            def save_step(obs, agent_1 = True):
                agent_num = 1 if agent_1 else 2
                birds_eye = self.processor.arena_1.photo_from_above() if agent_1 else self.processor.arena_2.photo_from_above()
                dream_obs_q = None
                obs.command_voice = obs.command_voice if parenting else prev_action_2.voice_out if agent_1 else prev_action_1.voice_out
                if(type(obs.command_voice) != Goal):
                    obs.command_voice = get_goal_from_one_hots(obs.command_voice)
                if(type(obs.report_voice) != Goal):
                    obs.report_voice = get_goal_from_one_hots(obs.report_voice)
                
                episode_dict[f"obs_{agent_num}"].append(obs) 
                episode_dict[f"birds_eye_{agent_num}"].append(birds_eye[:,:,0:3])
                
                
                
            def next_prediction(hp, hq, obs, wheels_joints, agent_1 = True):
                agent_num = 1 if agent_1 else 2
                
                pred_obs_p = self.forward.predict(hp.unsqueeze(1), self.forward.wheels_joints_in(wheels_joints)) 
                pred_obs_q = self.forward.predict(hq.unsqueeze(1), self.forward.wheels_joints_in(wheels_joints))
                
                dream_obs_q = deepcopy(pred_obs_q)
                dream_obs_q.vision = dream_obs_q.vision.squeeze(0)
                dream_obs_q.touch = dream_obs_q.touch.squeeze(0)
                
                pred_obs_p.command_voice = get_goal_from_one_hots(pred_obs_p.command_voice)
                pred_obs_q.command_voice = get_goal_from_one_hots(pred_obs_q.command_voice)
                
                pred_obs_p.report_voice = get_goal_from_one_hots(pred_obs_p.report_voice)
                pred_obs_q.report_voice = get_goal_from_one_hots(pred_obs_q.report_voice)
                
                episode_dict[f"prior_predictions_{agent_num}"].append(pred_obs_p)
                episode_dict[f"posterior_predictions_{agent_num}"].append(pred_obs_q)
            
                return(dream_obs_q)
                    
                    
                                    
            def display_step(step, agent_1 = True, done = False, stopping = False, dreaming = False):
                if(not display):
                    return
                #print(f"\n{self.processor.goal.human_text}", end = " ")
                #print("STEP:", step)
                plot_step(step, episode_dict, agent_1 = agent_1, last_step = done, saving = False, dreaming = dreaming, args = self.args)
                if(not self.processor.parenting and not stopping):
                    display_step(step, agent_1 = False, stopping = True)
                if(waiting):
                    WAITING = wait_for_button_press()
                    
            def video_display_step(step, agent_1 = True, done = False, stopping = False, dreaming = False):
                if(not video_display):
                    return
                #print(f"\n{self.processor.goal.human_text}", end = " ")
                #print("STEP:", step)
                plot_video_step(step, episode_dict, agent_1 = agent_1, last_step = done, saving = True, dreaming = dreaming, args = self.args)
                if(not self.processor.parenting and not stopping):
                    video_display_step(step, agent_1 = False, stopping = True)
                if(waiting):
                    WAITING = wait_for_button_press()
                    
                    
            
            for step in range(self.args.max_steps + 1):
                
                # First, save step.
                real_obs_1 = self.get_agent_obs()
                real_obs_2 = self.get_agent_obs(agent_1 = False)
                if(dreaming and step != 0):                     
                    print(f"\n\nDreaming, step {step}\n\n")
                    obs_1 = deepcopy(previous_dream_obs_q_1)
                    obs_2 = deepcopy(previous_dream_obs_q_2)
                else:
                    obs_1 = deepcopy(real_obs_1)
                    obs_2 = deepcopy(real_obs_2)
                
                save_step(real_obs_1)    
                if(not parenting):
                    save_step(real_obs_1, agent_1 = False)  
                    
                display_step(step, dreaming = dreaming)
                video_display_step(step, dreaming = dreaming)
                
                # Then, perform action.
                prev_action_1, values_1, hp_1, hq_1, vision_is_1, touch_is_1, command_voice_is_1, report_voice_is_1, \
                    prev_action_2, values_2, hp_2, hq_2, vision_is_2, touch_is_2, command_voice_is_2, report_voice_is_2, \
                        reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                            prev_action_1, hq_1, obs_1,
                            prev_action_2, hq_2, obs_2, verbose = verbose, sleep_time = sleep_time, user_action = user_action) 
                        
                previous_dream_obs_q_1 = next_prediction(hp_1, hq_1, deepcopy(obs_1), wheels_joints = prev_action_1.wheels_joints, agent_1 = True)
                if(not parenting):
                    previous_dream_obs_q_2 = next_prediction(hp_2, hq_2, deepcopy(obs_2), wheels_joints = prev_action_2.wheels_joints, agent_1 = False)
                        
                episode_dict["reward"].append(str(round(reward, 3)))
                
                def update_episode_dict(index, prev_action, vision_is, touch_is, command_voice_is, report_voice_is, values, reward):
                    episode_dict[f"action_{index}"].append(prev_action)
                    episode_dict[f"vision_dkl_{index}"].append(vision_is.dkl.sum().item())
                    episode_dict[f"touch_dkl_{index}"].append(touch_is.dkl.sum().item())
                    episode_dict[f"command_voice_dkl_{index}"].append(command_voice_is.dkl.sum().item())
                    episode_dict[f"report_voice_dkl_{index}"].append(report_voice_is.dkl.sum().item())
                    episode_dict[f"critic_predictions_{index}"].append(values)
                    episode_dict[f"reward_{index}"].append(str(round(reward, 3)))

                update_episode_dict(1, prev_action_1, vision_is_1, touch_is_1, command_voice_is_1, report_voice_is_1, values_1, reward)
                if not self.processor.parenting:
                    update_episode_dict(2, prev_action_2, vision_is_2, touch_is_2, command_voice_is_2, report_voice_is_2, values_2, reward_2)
                
                if(done):
                    real_obs_1 = self.get_agent_obs()
                    real_obs_2 = self.get_agent_obs(agent_1 = False)
                    save_step(real_obs_1, agent_1 = True)    
                    if(not self.processor.parenting):
                        save_step(real_obs_2, agent_1 = False) 
                    display_step(step + 1, done = True, dreaming = dreaming)
                    video_display_step(step + 1, done = True, dreaming = dreaming)
                    self.processor.done()
                    break
            
            return(win)
                    
                    
                    
    def get_composition_data(self, sleep_time = None):
        if(self.args.agents_per_composition_data != -1 and self.agent_num > self.args.agents_per_composition_data): 
            return
        adjusted_args = deepcopy(self.args)
        adjusted_args.capacity = len(self.all_processors)
        temp_memory = RecurrentReplayBuffer(adjusted_args)
        processor_lens = []
        for processor_name in self.all_processor_names:
            self.processor = self.all_processors[processor_name]
            self.processor.begin(test = None)    
            done, complete_reward, steps, \
                (to_push_list_1, prev_action_1, hq_1), \
                (to_push_list_2, prev_action_2, hq_2) = self.start_episode()
                     
            for step in range(self.args.max_steps):
                #print("Step", step)
                if(not done):
                    obs_1 = self.get_agent_obs()
                    obs_2 = self.get_agent_obs(agent_1 = False)
                    prev_action_1, values_1, hp_1, hq_1, vision_is_1, touch_is_1, command_voice_is_1, report_voice_is_1, \
                        prev_action_2, values_2, hp_2, hq_2, vision_is_2, touch_is_2, command_voice_is_2, report_voice_is_2, \
                            reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                                prev_action_1, hq_1, obs_1,
                                prev_action_2, hq_2, obs_2, sleep_time = sleep_time)
                to_push_list_1.append(to_push_1)
                if(done): break
            #print("DONE")
            self.processor.done()
            processor_lens.append(step)           
            for to_push in to_push_list_1:
                to_push.push(temp_memory)
                
        batch = self.get_batch(temp_memory, len(self.all_processors), random_sample = False)
        vision, touch, command_voice, report_voice, wheels_joints, voice_out, reward, done, mask, all_mask, episodes, steps = batch
        
        hps, hqs, vision_is, touch_is, command_voice_is, report_voice_is, pred_obs_p, pred_obs_q, labels = self.forward(
            torch.zeros((episodes, 1, self.args.pvrnn_mtrnn_size)), 
            Obs(vision, touch, command_voice, report_voice), Action(wheels_joints, voice_out))
        
        labels = labels.detach().cpu().numpy()
        non_zero_mask = labels[:, :, 0] != 0  # This checks if the first element of each sequence is not 0
        all_mask = all_mask.detach().cpu().numpy()  
        
        vision_zq = vision_is.zq.detach().cpu().numpy()
        touch_zq = touch_is.zq.detach().cpu().numpy()
        command_voice_zq = command_voice_is.zq.detach().cpu().numpy()
        report_voice_zq = report_voice_is.zq.detach().cpu().numpy()
        hq = hqs.detach().cpu().numpy()
        
        #print(non_zero_mask)
        
        self.plot_dict["composition_data"][self.epochs] = {
            "labels" : labels, "all_mask" : all_mask, "hq" : hq,
            "vision_zq" : vision_zq, "touch_zq" : touch_zq, "command_voice_zq" : command_voice_zq, "report_voice_zq" : report_voice_zq}
        
        
        
    def get_batch(self, memory, batch_size, random_sample = True):
        batch = memory.sample(batch_size, random_sample = random_sample)
        if(batch == False): return(False)
        
        vision, touch, command_voice, report_voice, wheels_joints, voice_out, reward, done, mask = batch
        vision = torch.from_numpy(vision).to(self.args.device)
        touch = torch.from_numpy(touch).to(self.args.device)
        command_voice = torch.from_numpy(command_voice).to(self.args.device)
        report_voice = torch.from_numpy(report_voice).to(self.args.device)
        wheels_joints = torch.from_numpy(wheels_joints)
        voice_out = torch.from_numpy(voice_out)
        reward = torch.from_numpy(reward).to(self.args.device)
        done = torch.from_numpy(done).to(self.args.device)
        mask = torch.from_numpy(mask)
        wheels_joints = torch.cat([torch.zeros(wheels_joints[:,0].unsqueeze(1).shape), wheels_joints], dim = 1).to(self.args.device)
        voice_out = torch.cat([torch.zeros(voice_out[:,0].unsqueeze(1).shape), voice_out], dim = 1).to(self.args.device)
        all_mask = torch.cat([torch.ones(mask.shape[0], 1, 1), mask], dim = 1).to(self.args.device)
        mask = mask.to(self.args.device)
        episodes = reward.shape[0]
        steps = reward.shape[1]
        
        if(self.args.half):
            vision, touch, command_voice, wheels_joints, voice_out, reward, done, mask, all_mask, mask = \
                vision.to(dtype=torch.float16), touch.to(dtype=torch.float16), command_voice.to(dtype=torch.float16), wheels_joints.to(dtype=torch.float16), \
                voice_out.to(dtype=torch.float16), reward.to(dtype=torch.float16), done.to(dtype=torch.float16), \
                mask.to(dtype=torch.float16), wheels_joints.to(dtype=torch.float16), voice_out.to(dtype=torch.float16), all_mask.to(dtype=torch.float16), mask.to(dtype=torch.float16)
        
        #print("\n\n")
        #print("Agent {}, epoch {}. vision: {}. voice in: {}. wheels_joints: {}. voice out: {}. reward: {}.  done: {}. mask: {}.".format(
        #    self.agent_num, self.epochs, vision.shape, voice_in.shape, wheels_joints.shape, voice_out.shape, reward.shape, done.shape, mask.shape))
        #print("\n\n")
        
        return(vision, touch, command_voice, report_voice, wheels_joints, voice_out, reward, done, mask, all_mask, episodes, steps)
        
    
    
    def epoch(self, batch_size):
        start_time = duration()
        self.epochs += 1
        self.total_epochs += 1
        self.train()
        parenting = self.processor.parenting
                                
        batch = self.get_batch(self.memory, batch_size)
        if(batch == False):
            return(False)
        
        vision, touch, command_voice, report_voice, wheels_joints, voice_out, reward, done, mask, all_mask, episodes, steps = batch
        obs = Obs(vision, touch, command_voice, report_voice)
        actions = Action(wheels_joints, voice_out)
        
        
                
        # Train forward
        hps, hqs, vision_is, touch_is, command_voice_is, report_voice_is, pred_obs_p, pred_obs_q, labels = self.forward(
            torch.zeros((episodes, 1, self.args.pvrnn_mtrnn_size)), 
            obs, actions)
                                
        vision_loss = F.binary_cross_entropy(pred_obs_q.vision, vision[:,1:], reduction = "none").mean((-1,-2,-3)).unsqueeze(-1) * mask * self.args.vision_scaler
                        
        touch_loss = F.mse_loss(pred_obs_q.touch, touch[:,1:], reduction = "none")
        touch_loss = touch_loss.mean(-1).unsqueeze(-1) * mask * self.args.touch_scaler
        
        def compute_individual_voice_loss(real_voice, pred_voice, voice_scaler):
            real_voice = real_voice[:, 1:].reshape((episodes * steps, self.args.max_voice_len, self.args.voice_shape))
            real_voice = torch.argmax(real_voice, dim=-1)
            pred_voice = pred_voice.reshape((pred_voice.shape[0] * pred_voice.shape[1], self.args.max_voice_len, self.args.voice_shape))
            pred_voice = pred_voice.transpose(1, 2)
            voice_loss = F.cross_entropy(pred_voice, real_voice, reduction="none")
            voice_loss = voice_loss.reshape(episodes, steps, self.args.max_voice_len)
            voice_loss = voice_loss.mean(dim=2).unsqueeze(-1) * mask * voice_scaler
            return voice_loss, pred_voice
        
        command_voice_loss, pred_command_voice = compute_individual_voice_loss(
            command_voice, pred_obs_q.command_voice, self.args.command_voice_scaler)

        report_voice_loss, pred_report_voice = compute_individual_voice_loss(
            report_voice, pred_obs_q.report_voice, self.args.report_voice_scaler)
        
        accuracy = (vision_loss + touch_loss + command_voice_loss + report_voice_loss).mean()
        
        vision_complexity = vision_is.dkl.mean(-1).unsqueeze(-1) * all_mask
        touch_complexity = touch_is.dkl.mean(-1).unsqueeze(-1) * all_mask
        command_voice_complexity = command_voice_is.dkl.mean(-1).unsqueeze(-1) * all_mask
        report_voice_complexity = report_voice_is.dkl.mean(-1).unsqueeze(-1) * all_mask
                
        complexity = sum([
            self.args.beta_vision * vision_complexity.mean(),
            self.args.beta_touch * touch_complexity.mean(),
            self.args.beta_command_voice * command_voice_complexity.mean(),
            self.args.beta_report_voice * report_voice_complexity.mean()])       
                                
        self.forward_opt.zero_grad()
        (accuracy + complexity).backward()
        self.forward_opt.step()
        
        torch.cuda.empty_cache()
        
        vision_complexity = vision_complexity[:,1:]
        touch_complexity = touch_complexity[:,1:]
        command_voice_complexity = command_voice_complexity[:,1:]
        report_voice_complexity = report_voice_complexity[:,1:]
                                    
                        
        
        # Get curiosity                 
        vision_prediction_error_curiosity             = self.args.prediction_error_eta_vision           * vision_loss
        touch_prediction_error_curiosity          = self.args.prediction_error_eta_touch        * touch_loss
        command_voice_prediction_error_curiosity     = self.args.prediction_error_eta_command_voice   * command_voice_loss
        report_voice_prediction_error_curiosity = self.args.prediction_error_eta_report_voice   * report_voice_loss
        prediction_error_curiosity                  = vision_prediction_error_curiosity + touch_prediction_error_curiosity + command_voice_prediction_error_curiosity + report_voice_prediction_error_curiosity
        
        vision_hidden_state_curiosity                 = self.args.hidden_state_eta_vision               * torch.clamp(vision_complexity, min = 0, max = self.args.dkl_max)  # Or tanh? sigmoid? Or just clamp?
        touch_hidden_state_curiosity              = self.args.hidden_state_eta_touch            * torch.clamp(touch_complexity, min = 0, max = self.args.dkl_max)
        command_voice_hidden_state_curiosity         = self.args.hidden_state_eta_command_voice       * torch.clamp(command_voice_complexity, min = 0, max = self.args.dkl_max)
        report_voice_hidden_state_curiosity     = self.args.hidden_state_eta_report_voice       * torch.clamp(report_voice_complexity, min = 0, max = self.args.dkl_max) * self.hidden_state_eta_report_voice_reduction
        hidden_state_curiosity                      = vision_hidden_state_curiosity + touch_hidden_state_curiosity + command_voice_hidden_state_curiosity + report_voice_hidden_state_curiosity
        
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
            Q = self.critics[i](Action(wheels_joints[:,1:], voice_out[:,1:]), hqs[:,:-1].detach())
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
            
            loc = torch.zeros(self.args.wheels_joints_shape, dtype=torch.float64).to(self.args.device).float()
            n = self.args.wheels_joints_shape
            scale_tril = torch.tril(torch.ones(n, n)).to(self.args.device).float()
            policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
            policy_prior_log_prrgbd = self.args.normal_alpha * policy_prior.log_prob(new_action.wheels_joints).unsqueeze(-1)
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
                                
                                
                                
        if(accuracy != None):               accuracy = accuracy.item()
        if(vision_loss != None):              vision_loss = vision_loss.mean().item()
        if(touch_loss != None):           touch_loss = touch_loss.mean().item()
        if(command_voice_loss != None):      command_voice_loss = command_voice_loss.mean().item()
        if(report_voice_loss != None):      report_voice_loss = report_voice_loss.mean().item()
        if(complexity != None):             complexity = complexity.item()
        if(alpha_loss != None):             alpha_loss = alpha_loss.item()
        if(alpha_text_loss != None):        alpha_text_loss = alpha_text_loss.item()
        if(actor_loss != None):             actor_loss = actor_loss.item()
        if(Q != None):                      Q = -Q.mean().item()
        for i in range(self.args.critics):
            if(critic_losses[i] != None): 
                critic_losses[i] = critic_losses[i].item()
                critic_losses[i] = log(critic_losses[i]) if critic_losses[i] > 0 else critic_losses[i]
                
        vision_prediction_error_curiosity = vision_prediction_error_curiosity.mean().item()
        touch_prediction_error_curiosity = touch_prediction_error_curiosity.mean().item()
        command_voice_prediction_error_curiosity = command_voice_prediction_error_curiosity.mean().item()
        report_voice_prediction_error_curiosity = report_voice_prediction_error_curiosity.mean().item()
        
        vision_hidden_state_curiosity = vision_hidden_state_curiosity.mean().item()
        touch_hidden_state_curiosity = touch_hidden_state_curiosity.mean().item()
        command_voice_hidden_state_curiosity = command_voice_hidden_state_curiosity.mean().item()
        report_voice_hidden_state_curiosity = report_voice_hidden_state_curiosity.mean().item()
        
        prediction_error_curiosity = prediction_error_curiosity.mean().item()
        hidden_state_curiosity = hidden_state_curiosity.mean().item()
        


        if(self.epochs == 1 or self.epochs >= self.args.epochs or self.epochs % self.args.keep_data == 0):
            self.plot_dict["accuracy_loss"].append(accuracy)
            self.plot_dict["vision_loss"].append(vision_loss)
            self.plot_dict["touch_loss"].append(touch_loss)
            self.plot_dict["command_voice_loss"].append(command_voice_loss)
            self.plot_dict["report_voice_loss"].append(report_voice_loss)
            self.plot_dict["complexity_loss"].append(complexity)                                                                             
            self.plot_dict["alpha_loss"].append(alpha_loss)
            self.plot_dict["alpha_text_loss"].append(alpha_text_loss)
            self.plot_dict["actor_loss"].append(actor_loss)
            for layer, f in enumerate(critic_losses):
                self.plot_dict["critics_loss"][layer].append(f)    
            self.plot_dict["critics_loss"].append(critic_losses)
            self.plot_dict["extrinsic"].append(extrinsic)
            self.plot_dict["q"].append(Q)
            self.plot_dict["intrinsic_curiosity"].append(intrinsic_curiosity)
            self.plot_dict["intrinsic_entropy"].append(intrinsic_entropy)
            self.plot_dict["vision_prediction_error_curiosity"].append(vision_prediction_error_curiosity)
            self.plot_dict["touch_prediction_error_curiosity"].append(touch_prediction_error_curiosity)
            self.plot_dict["command_voice_prediction_error_curiosity"].append(command_voice_prediction_error_curiosity)
            self.plot_dict["report_voice_prediction_error_curiosity"].append(report_voice_prediction_error_curiosity)
            self.plot_dict["prediction_error_curiosity"].append(prediction_error_curiosity)
            self.plot_dict["vision_hidden_state_curiosity"].append(vision_hidden_state_curiosity)    
            self.plot_dict["touch_hidden_state_curiosity"].append(touch_hidden_state_curiosity)  
            self.plot_dict["command_voice_hidden_state_curiosity"].append(command_voice_hidden_state_curiosity)  
            self.plot_dict["report_voice_hidden_state_curiosity"].append(report_voice_hidden_state_curiosity)    
            self.plot_dict["hidden_state_curiosity"].append(hidden_state_curiosity)    
            
        end_time = duration()
        print_duration(start_time, end_time, "\nEpoch", "\n")
        
    
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    
        
    def sizeof_plot_dict(self):
        """Print true deep sizes of self.plot_dict and other large self attributes."""

        total_bytes = 0
        
        print("\n\n\n")
        print("-" * 50)
        print("Key".ljust(30), "Size")
        print("-" * 50)

        # Deep plot_dict
        for key, value in self.plot_dict.items():
            size_estimate = len(pickle.dumps(value))
            if(size_estimate > 100000):
                print(f"{key}:\t{sizeof_fmt(size_estimate)}")

        print("-" * 50)

        # Deep special attributes
        keys_to_check = [
            "arena_1", "arena_2", 
            "log_alpha", "alpha_opt", 
            "forward", "forward_opt", 
            "actor", "actor_opt",
            "memory"
        ]

        for attr_name in keys_to_check:
            attr = getattr(self, attr_name, None)
            if attr is not None:
                try:
                    size_estimate = len(pickle.dumps(attr))
                    if(size_estimate > 100000):
                        print(f"{attr_name}:\t{sizeof_fmt(size_estimate)}")
                except Exception as e:
                    print(f"{attr_name}:\t(Unserializable: {e})")
            else:
                print(f"{attr_name}: (None)")

        print("-" * 50)
        print(f"{'TOTAL'.ljust(30)}\t{sizeof_fmt(len(pickle.dumps(self)))}")
        print("-" * 50)
        print("\n\n\n")
        
        
    
    def save_agent(self):
        if not self.args.local:
            self.sizeof_plot_dict()
            plot_dict_backup = self.plot_dict
            memory_backup = self.memory
            self.plot_dict = None
            self.memory = None
            save_path = f"{folder}/agents/agent_{str(self.agent_num).zfill(4)}_epoch_{str(self.epochs).zfill(6)}.pkl.gz"
            with gzip.open(save_path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.plot_dict = plot_dict_backup
            self.memory = memory_backup
                
    def load_agent(self, load_path):
        with gzip.open(load_path, "rb") as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)
                
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
    agent = Agent(args = args)
    agent.save_episodes()
# %%