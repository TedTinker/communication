#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
from math import log
from itertools import accumulate
from copy import deepcopy

from utils import default_args, dkl, print, goals
from scenario import Scenario
from buffer import RecurrentReplayBuffer
from models import Forward, Actor, Actor_HQ, Critic, Critic_HQ

action_size = 2



class Agent:
    
    def __init__(self, i, args = default_args):
        
        self.start_time = None
        
        self.agent_num = i
        self.args = args
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        self.scenario_desc = self.args.scenario_list[0]
        self.goal_comm = not self.scenario_desc[1]
        self.arms = self.scenario_desc[2]
        self.scenario = Scenario(self.scenario_desc, args = args)
        
        self.target_entropy = args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=args.alpha_lr, weight_decay=0) 
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        
        self.forward = Forward(args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=args.forward_lr, weight_decay=0)   
                           
        self.actor = Actor_HQ(args) if args.actor_hq else Actor(args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=0)     
        
        self.critic1 = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=args.critic_lr, weight_decay=0)
        self.critic1_target = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.train()
        
        self.memory = RecurrentReplayBuffer(args)
        self.plot_dict = {
            "args" : args,
            "arg_title" : args.arg_title,
            "arg_name" : args.arg_name,
            "pred_lists" : {},
            "agent_lists" : {"forward" : Forward, "actor" : Actor_HQ if args.actor_hq else Actor, "critic" : Critic_HQ if args.critic_hq else Critic},
            "rewards" : [], 
            "accuracy" : [], "complexity" : [],
            "alpha" : [], "actor" : [], 
            "critic_1" : [], "critic_2" : [], 
            "extrinsic" : [], "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            "naive" : [], "free" : []}
        
        
        
    def training(self, q):
        self.pred_episodes_hq() if self.args.actor_hq else self.pred_episodes()
        self.save_agent()
        while(True):
            cumulative_epochs = 0
            prev_scenario_desc = self.scenario_desc
            for j, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs < cumulative_epochs): self.scenario_desc = self.args.scenario_list[j] ; break
            if(prev_scenario_desc != self.scenario_desc): 
                self.pred_episodes_hq() if self.args.actor_hq else self.pred_episodes()
                for arena in self.scenario.arenas:
                    arena.stop()
                self.goal_comm = not self.scenario_desc[1]
                self.scenario = Scenario(self.scenario_desc, args = self.args)
                self.memory = RecurrentReplayBuffer(self.args)
                self.pred_episodes_hq() if self.args.actor_hq else self.pred_episodes()
            self.training_episode()
            percent_done = str(self.epochs / sum(self.args.epochs))
            q.put((self.agent_num, percent_done))
            if(self.epochs >= sum(self.args.epochs)): break
            if(self.epochs % self.args.epochs_per_agent_list == 0): self.save_agent()
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        self.pred_episodes_hq() if self.args.actor_hq else self.pred_episodes()
        self.save_agent()
        
        self.min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in self.min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "agent_lists"]):
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
                
                
                
    def save_agent(self):
        if(self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list): return
        self.plot_dict["agent_lists"]["{}_{}".format(self.agent_num, self.epochs)] = deepcopy(self.state_dict())
                
                
                
    def step_in_episode(self, i, prev_a, h_actor, push, verbose):
        to_push = None
        with torch.no_grad():
            o, s, c, gc = self.scenario.obs(i)
            comm = gc if self.goal_comm else c
            a, _, new_h_actor = self.actor(o, s, comm, prev_a[i], h_actor[i], goal_comm = self.goal_comm)
            h_actor[i] = new_h_actor
            action = torch.flatten(a).tolist()
            r, done, action_name = self.scenario.action(i, action, verbose)
            no, ns, nc, ngc = self.scenario.obs(i)
            if(push): 
                to_push = [o, s, c, gc, a, r, no, ns, nc, ngc, done]
        return(a, h_actor, r, done, action_name, to_push)
    
    
    
    def step_in_episode_hq(self, i, prev_a, h_q_m1, push, verbose):
        to_push = None
        with torch.no_grad():
            o, s, c, gc = self.scenario.obs(i)
            comm = gc if self.goal_comm else c
            _, _, h_q = self.forward(o, s, comm, prev_a[i], h_q_m1[i], goal_comm = self.goal_comm)
            a, _, _ = self.actor(h_q, goal_comm = self.goal_comm)
            action = torch.flatten(a).tolist()
            r, done, action_name = self.scenario.action(i, action, verbose)
            no, ns, nc, ngc = self.scenario.obs(i)
            if(push): 
                to_push = [o, s, c, gc, a, r, no, ns, nc, ngc, done]
        return(a, h_q, r, done, action_name, to_push)
    
    
    
    def pred_episodes(self):
        with torch.no_grad():
            if(self.args.agents_per_pred_list != -1 and self.agent_num > self.args.agents_per_pred_list): return
            pred_lists = []
            for episode in range(self.args.episodes_in_pred_list):
                done = False ; prev_a = torch.zeros((1, 1, 4 + self.args.symbols))
                h_actor = torch.zeros((1, 1, self.args.hidden_size))
                h_q     = torch.zeros((1, 1, self.args.hidden_size))
                self.scenario.begin()
                rgbd, spe, comm, goal_comm = self.scenario.obs(0)
                comm = goal_comm if self.goal_comm else comm
                pred_list = [(None, (rgbd.squeeze(0), spe.squeeze(0), comm.squeeze(0)), ((None, None, None), (None, None, None)), ((None, None, None), (None, None, None)))]
                for step in range(self.args.max_steps):
                    if(not done): 
                        rgbd, spe, comm, goal_comm = self.scenario.obs(0)
                        comm = goal_comm if self.goal_comm else comm
                        a, h_actor, _, done, action_name, _ = self.step_in_episode(0, prev_a, h_actor, push = False, verbose = False)
                        (zp_mu, zp_std), (zq_mu, zq_std), h_q_p1 = self.forward(rgbd, spe, comm, prev_a, h_q, goal_comm = self.goal_comm)
                        (rgbd_mu_pred_p, pred_rgbd_p), (spe_mu_pred_p, pred_spe_p), (comm_mu_pred_p, pred_comm_p) = self.forward.get_preds(a, zp_mu, zp_std, h_q, quantity = self.args.samples_per_pred, goal_comm = self.goal_comm)
                        (rgbd_mu_pred_q, pred_rgbd_q), (spe_mu_pred_q, pred_spe_q), (comm_mu_pred_q, pred_comm_q) = self.forward.get_preds(a, zq_mu, zq_std, h_q, quantity = self.args.samples_per_pred, goal_comm = self.goal_comm)
                        pred_rgbd_p = [pred.squeeze(0).squeeze(0) for pred in pred_rgbd_p] ; pred_rgbd_q = [pred.squeeze(0).squeeze(0) for pred in pred_rgbd_q]
                        pred_spe_p = [pred.squeeze(0).squeeze(0) for pred in pred_spe_p]   ; pred_spe_q = [pred.squeeze(0).squeeze(0) for pred in pred_spe_q]
                        pred_comm_p = [pred.squeeze(0).squeeze(0) for pred in pred_comm_p] ; pred_comm_q = [pred.squeeze(0).squeeze(0) for pred in pred_comm_q]
                        rgbd, spe, comm, goal_comm = self.scenario.obs(0)
                        comm = goal_comm if self.goal_comm else comm
                        pred_list.append((
                            action_name, (rgbd.squeeze(0), spe.squeeze(0), comm.squeeze(0)), 
                            ((rgbd_mu_pred_p.squeeze(0).squeeze(0), pred_rgbd_p), (spe_mu_pred_p.squeeze(0).squeeze(0), pred_spe_p), (comm_mu_pred_p.squeeze(0).squeeze(0), pred_comm_p)), 
                            ((rgbd_mu_pred_q.squeeze(0).squeeze(0), pred_rgbd_q), (spe_mu_pred_q.squeeze(0).squeeze(0), pred_spe_q), (comm_mu_pred_q.squeeze(0).squeeze(0), pred_comm_q))))
                        prev_a = a ; h_q = h_q_p1
                pred_lists.append(pred_list)
            self.plot_dict["pred_lists"]["{}_{}_{}".format(self.agent_num, self.epochs, self.scenario.desc)] = pred_lists
            
            
            
    def pred_episodes_hq(self):
        with torch.no_grad():
            if(self.args.agents_per_pred_list != -1 and self.agent_num > self.args.agents_per_pred_list): return
            pred_lists = []
            for episode in range(self.args.episodes_in_pred_list):
                done = False ; prev_a = torch.zeros((1, 1, 4 + self.args.symbols))
                h_q = torch.zeros((1, 1, self.args.hidden_size))
                self.scenario.begin()
                rgbd, spe, comm, goal_comm = self.scenario.obs(0)
                comm = goal_comm if self.goal_comm else comm
                pred_list = [(None, (rgbd.squeeze(0), spe.squeeze(0), comm.squeeze(0)), ((None, None, None), (None, None, None)), ((None, None, None), (None, None, None)))]
                for step in range(self.args.max_steps):
                    if(not done): 
                        rgbd, spe, comm, goal_comm = self.scenario.obs(0)
                        comm = goal_comm if self.goal_comm else comm
                        a, h_q_p1, _, done, action_name, _ = self.step_in_episode_hq(0, prev_a, h_q.unsqueeze(0), push = False, verbose = False)
                        (zp_mu, zp_std), (zq_mu, zq_std), _ = self.forward(rgbd, spe, comm, prev_a, h_q, goal_comm = self.goal_comm)
                        (rgbd_mu_pred_p, pred_rgbd_p), (spe_mu_pred_p, pred_spe_p), (comm_mu_pred_p, pred_comm_p) = self.forward.get_preds(a, zp_mu, zp_std, h_q, quantity = self.args.samples_per_pred, goal_comm = self.goal_comm)
                        (rgbd_mu_pred_q, pred_rgbd_q), (spe_mu_pred_q, pred_spe_q), (comm_mu_pred_q, pred_comm_q) = self.forward.get_preds(a, zq_mu, zq_std, h_q, quantity = self.args.samples_per_pred, goal_comm = self.goal_comm)
                        pred_rgbd_p = [pred.squeeze(0).squeeze(0) for pred in pred_rgbd_p] ; pred_rgbd_q = [pred.squeeze(0).squeeze(0) for pred in pred_rgbd_q]
                        pred_spe_p = [pred.squeeze(0).squeeze(0) for pred in pred_spe_p]   ; pred_spe_q = [pred.squeeze(0).squeeze(0) for pred in pred_spe_q]
                        pred_comm_p = [pred.squeeze(0).squeeze(0) for pred in pred_comm_p] ; pred_comm_q = [pred.squeeze(0).squeeze(0) for pred in pred_comm_q]
                        rgbd, spe, comm, goal_comm = self.scenario.obs(0)
                        comm = goal_comm if self.goal_comm else comm
                        pred_list.append((
                            action_name, (rgbd.squeeze(0), spe.squeeze(0), comm.squeeze(0)), 
                            ((rgbd_mu_pred_p.squeeze(0).squeeze(0), pred_rgbd_p), (spe_mu_pred_p.squeeze(0).squeeze(0), pred_spe_p), (comm_mu_pred_p.squeeze(0).squeeze(0), pred_comm_p)), 
                            ((rgbd_mu_pred_q.squeeze(0).squeeze(0), pred_rgbd_q), (spe_mu_pred_q.squeeze(0).squeeze(0), pred_spe_q), (comm_mu_pred_q.squeeze(0).squeeze(0), pred_comm_q))))
                        prev_a = a ; h_q = h_q_p1
                pred_lists.append(pred_list)
            self.plot_dict["pred_lists"]["{}_{}_{}".format(self.agent_num, self.epochs, self.scenario.desc)] = pred_lists
    
    
    
    def training_episode(self, push = True, verbose = False):
        dones = [False for _ in self.scenario.arenas]
        prev_as = [torch.zeros((1, 1, 4 + self.args.symbols)) for _ in self.scenario.arenas]
        cumulative_rs = [0 for _ in range(self.args.max_steps)]
        hs = [torch.zeros((1, 1, self.args.hidden_size)) for _ in self.scenario.arenas]
        to_be_pushed = [[] for _ in self.scenario.arenas]
        self.scenario.begin()
        if(verbose): print("\n\n\n\n\nSTART!\n")
        
        for step in range(self.args.max_steps):
            self.steps += 1
            for i in range(len(self.scenario.arenas)):
                if(not any(dones)):
                    prev_as[i], hs[i], r, dones[i], _, to_push = self.step_in_episode_hq(i, prev_as, hs, push, verbose) if self.args.actor_hq else self.step_in_episode(i, prev_as, hs, push, verbose)
                    cumulative_rs[step] += r
                    to_be_pushed[i].append(to_push)
            self.scenario.replace_comms()
                
            if(self.steps % self.args.steps_per_epoch == 0):
                #print("episodes: {}. epochs: {}. steps: {}.".format(self.episodes, self.epochs, self.steps))
                plot_data = self.epoch(batch_size = self.args.batch_size)
                if(plot_data == False): pass
                else:
                    l, e, ic, ie, naive, free = plot_data
                    if(self.epochs == 1 or self.epochs >= sum(self.args.epochs) or self.epochs % self.args.keep_data == 0):
                        self.plot_dict["accuracy"].append(l[0][0])
                        self.plot_dict["complexity"].append(l[0][1])
                        self.plot_dict["alpha"].append(l[0][2])
                        self.plot_dict["actor"].append(l[0][3])
                        self.plot_dict["critic_1"].append(l[0][4])
                        self.plot_dict["critic_2"].append(l[0][5])
                        self.plot_dict["extrinsic"].append(e)
                        self.plot_dict["intrinsic_curiosity"].append(ic)
                        self.plot_dict["intrinsic_entropy"].append(ie)
                        self.plot_dict["naive"].append(naive)
                        self.plot_dict["free"].append(free)    
        self.plot_dict["rewards"].append(sum(cumulative_rs)/self.scenario.num_agents)
        self.episodes += 1
        if(push):
            for i in range(len(self.scenario.arenas)):
                _, _, c, gc = self.scenario.obs(i)
                to_be_pushed[i][-1][8] = c
                to_be_pushed[i][-1][9] = gc
            for to_push in to_be_pushed:
                for step, (o, s, c, gc, a, r, no, ns, nc, ngc, done) in enumerate(to_push):
                    self.memory.push(o, s, c, gc, a, cumulative_rs[step], no, ns, nc, ngc, done, done)
                        
    
    
    def epoch(self, batch_size):
                                        
        batch = self.memory.sample(batch_size)
        if(batch == False): return(False)
                        
        self.epochs += 1

        rgbd, spe, comm, goal_comm, actions, rewards, dones, masks = batch
        comm = goal_comm if self.goal_comm else comm 
        actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions], dim = 1)
        episodes = rewards.shape[0] ; steps = rewards.shape[1]
                
        #print("\n\n")
        #print("{}. rgbd: {}. spe: {}. comm: {}. actions: {}. rewards: {}. dones: {}. masks: {}.".format(
        #    self.agent_num, rgbd.shape, spe.shape, comm.shape, actions.shape, rewards.shape, dones.shape, masks.shape))
        #print("\n\n")
                
        

        # Train forward
        h_qs = [torch.zeros((episodes, 1, self.args.hidden_size)).to(rgbd.device)]
        zp_mus = []       ; zp_stds = []
        zq_mus = []       ; zq_stds = []
        zq_pred_rgbd = [] ; zq_pred_spe = [] ; zq_pred_comm = []
        for step in range(steps):
            (zp_mu, zp_std), (zq_mu, zq_std), h_q_p1 = self.forward(rgbd[:, step], spe[:, step], comm[:, step], actions[:, step], h_qs[-1], goal_comm = self.goal_comm)
            (_, zq_preds_rgbd), (_, zq_preds_spe), (_, zq_preds_comm) = self.forward.get_preds(actions[:, step+1], zq_mu, zq_std, h_qs[-1], quantity = self.args.elbo_num, goal_comm = self.goal_comm)
            zp_mus.append(zp_mu) ; zp_stds.append(zp_std)
            zq_mus.append(zq_mu) ; zq_stds.append(zq_std)
            zq_pred_rgbd.append(torch.cat(zq_preds_rgbd, -1)) ; zq_pred_spe.append(torch.cat(zq_preds_spe, -1))
            zq_pred_comm.append(torch.cat(zq_preds_comm, -1))
            h_qs.append(h_q_p1)
        h_qs.append(h_qs.pop(0)) ; h_qs = torch.cat(h_qs, dim = 1) ; next_hqs = h_qs[:, 1:] ; hqs = h_qs[:, :-1]
        zp_mus = torch.cat(zp_mus, dim = 1) ; zp_stds = torch.cat(zp_stds, dim = 1)
        zq_mus = torch.cat(zq_mus, dim = 1) ; zq_stds = torch.cat(zq_stds, dim = 1)
        zq_pred_rgbd = torch.cat(zq_pred_rgbd, dim = 1) ; zq_pred_spe = torch.cat(zq_pred_spe, dim = 1)
        zq_pred_comm = torch.cat(zq_pred_comm, dim = 1) 
        
        next_rgbd_tiled = torch.tile(rgbd[:,1:], (1, 1, 1, 1, self.args.elbo_num))
        next_spe_tiled  = torch.tile(spe[:,1:], (1, 1, self.args.elbo_num))
        next_comm_tiled = torch.tile(comm[:,1:], (1, 1, self.args.elbo_num))
        
        image_loss = F.binary_cross_entropy_with_logits(zq_pred_rgbd, next_rgbd_tiled, reduction = "none").mean((-1,-2,-3)).unsqueeze(-1) * masks / self.args.elbo_num
        speed_loss = self.args.speed_scalar * F.mse_loss(zq_pred_spe, next_spe_tiled,  reduction = "none").mean(-1).unsqueeze(-1) * masks / self.args.elbo_num
        comm_loss  = self.args.comm_scalar * F.mse_loss(zq_pred_comm, next_comm_tiled,  reduction = "none").mean(-1).unsqueeze(-1) * masks / self.args.elbo_num
        accuracy_for_naive = image_loss + speed_loss + comm_loss
        accuracy            = accuracy_for_naive.mean()
        complexity_for_free = dkl(zq_mus, zq_stds, zp_mus, zp_stds).mean(-1).unsqueeze(-1) * masks
        if(self.args.dkl_max != None):
            complexity_for_free = torch.clamp(complexity_for_free, min = 0, max = self.args.dkl_max)
        complexity          = self.args.beta * complexity_for_free.mean() 
                                
        self.forward_opt.zero_grad()
        (accuracy + complexity).backward()
        self.forward_opt.step()
        
        if(self.args.beta == 0): complexity = None
                                
        # Get curiosity                  
        naive_curiosity = self.args.naive_eta * accuracy_for_naive  
        free_curiosity = self.args.free_eta * complexity_for_free
        if(self.args.curiosity == "naive"):  curiosity = naive_curiosity
        elif(self.args.curiosity == "free"): curiosity = free_curiosity
        else:                                curiosity = torch.zeros(rewards.shape)
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        rewards += curiosity
                        
        
                
        # Train critics
        with torch.no_grad():
            new_actions, log_pis_next, _ = self.actor(next_hqs, goal_comm = self.goal_comm) if self.args.actor_hq else self.actor(rgbd[:,1:], spe[:,1:], comm[:,1:], actions[:,1:], goal_comm = self.goal_comm)
            Q_target1_next, _ = self.critic1_target(next_hqs, new_actions) if self.args.critic_hq else self.critic1_target(rgbd[:,1:], spe[:,1:], comm[:,1:], new_actions, goal_comm = self.goal_comm)
            Q_target2_next, _ = self.critic2_target(next_hqs, new_actions) if self.args.critic_hq else self.critic2_target(rgbd[:,1:], spe[:,1:], comm[:,1:], new_actions, goal_comm = self.goal_comm)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            if self.args.alpha == None: Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
            else:                       Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.args.alpha * log_pis_next))
        
        Q_1, _ = self.critic1(hqs.detach(), actions[:,1:]) if self.args.critic_hq else self.critic1(rgbd[:,:-1], spe[:,:-1], comm[:,:-1], actions[:,1:], goal_comm = self.goal_comm)
        critic1_loss = 0.5*F.mse_loss(Q_1*masks, Q_targets*masks)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2, _ = self.critic2(hqs.detach(), actions[:,1:]) if self.args.critic_hq else self.critic2(rgbd[:,:-1], spe[:,:-1], comm[:,:-1], actions[:,1:], goal_comm = self.goal_comm)
        critic2_loss = 0.5*F.mse_loss(Q_2*masks, Q_targets*masks)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
                                
        
        
        # Train alpha
        if self.args.alpha == None:
            _, log_pis, _ = self.actor(hqs.detach(), goal_comm = self.goal_comm) if self.args.actor_hq else self.actor(rgbd[:,:-1], spe[:,:-1], comm[:,:-1], actions[:,:-1], goal_comm = self.goal_comm)
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy))*masks
            alpha_loss = alpha_loss.mean() / masks.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha) 
        else:
            alpha_loss = None
                                                
            
        
        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       alpha = self.args.alpha
            new_actions, log_pis, _ = self.actor(hqs.detach(), goal_comm = self.goal_comm) if self.args.actor_hq else self.actor(rgbd[:,:-1], spe[:,:-1], comm[:,:-1], actions[:,:-1], goal_comm = self.goal_comm)

            if self.args.action_prior == "normal":
                loc = torch.zeros(new_actions.shape[-1], dtype=torch.float64)
                scale_tril = torch.tensor([[0 if j > i else 1 for j in range(new_actions.shape[-1])] for i in range(new_actions.shape[-1])], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_prrgbd = policy_prior.log_prob(new_actions).unsqueeze(-1)
            elif self.args.action_prior == "uniform":
                policy_prior_log_prrgbd = 0.0
            Q_1, _ = self.critic1(hqs.detach(), new_actions) if self.args.critic_hq else self.critic1(rgbd[:,:-1], spe[:,:-1], comm[:,:-1], new_actions, goal_comm = self.goal_comm)
            Q_2, _ = self.critic2(hqs.detach(), new_actions) if self.args.critic_hq else self.critic2(rgbd[:,:-1], spe[:,:-1], comm[:,:-1], new_actions, goal_comm = self.goal_comm)
            Q = torch.min(Q_1, Q_2).mean(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis)*masks).item()
            actor_loss = (alpha * log_pis - policy_prior_log_prrgbd - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            actor_loss = None
                                
                                
                                
        if(accuracy != None):   accuracy = accuracy.item()
        if(complexity != None): complexity = complexity.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): 
            critic1_loss = critic1_loss.item()
            critic1_loss = log(critic1_loss) if critic1_loss > 0 else critic1_loss
        if(critic2_loss != None): 
            critic2_loss = critic2_loss.item()
            critic2_loss = log(critic2_loss) if critic2_loss > 0 else critic2_loss
        losses = np.array([[accuracy, complexity, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        naive_curiosity = naive_curiosity.mean().item()
        free_curiosity = free_curiosity.mean().item()
        if(free_curiosity == 0): free_curiosity = None
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, naive_curiosity, free_curiosity)
    
    
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        return(
            self.forward.state_dict(),
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic1_target.state_dict(),
            self.critic2.state_dict(),
            self.critic2_target.state_dict())

    def load_state_dict(self, state_dict):
        self.forward.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        self.critic1.load_state_dict(state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(state_dict[4])
        self.critic2_target.load_state_dict(state_dict[5])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.forward.eval()
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.forward.train()
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        
        
        
if __name__ == "__main__":
    agent = Agent(0)
# %%
