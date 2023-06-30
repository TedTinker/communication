#%%
import ast
import os
import pickle
import torch
import tkinter as tk
from tkinter import ttk
from math import degrees, radians
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scenario import Scenario



class Agent_and_Episode:
    
    def __init__(self, plot_dict, agent_name):
        
        args = plot_dict["args"]
        agent_lists = plot_dict["agent_lists"]
        state_dict = agent_lists[agent_name]
    
        self.forward        = agent_lists["forward"](args)
        self.actor          = agent_lists["actor"](args)
        self.critic1        = agent_lists["critic"](args)
        self.critic1_target = agent_lists["critic"](args)
        self.critic2        = agent_lists["critic"](args)
        self.critic2_target = agent_lists["critic"](args)
        
        self.forward.load_state_dict(       state_dict[0])
        self.actor.load_state_dict(         state_dict[1])
        self.critic1.load_state_dict(       state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(       state_dict[4])
        self.critic2_target.load_state_dict(state_dict[5])
        
        self.a          = [torch.zeros((1, 1, 4 + args.symbols))]
        self.h_q        = [torch.zeros((1, 1, args.hidden_size))]
        self.h_actor    = [torch.zeros((1, 1, args.hidden_size))]
        self.h_critic1  = [torch.zeros((1, 1, args.hidden_size))]
        self.h_critic2  = [torch.zeros((1, 1, args.hidden_size))]
        self.h_critict1 = [torch.zeros((1, 1, args.hidden_size))]
        self.h_critict2 = [torch.zeros((1, 1, args.hidden_size))]
        self.zq_mu = [] ; self.zq_std = []
        self.Qs = []
        
        self.scenario = Scenario(args.objects, 1, False, True, args)
        self.scenario.begin()
        o, s, comm, goal_comm = self.scenario.obs(0)
        self.o = [o] ; self.s = [s] ; self.comm = [comm] ; self.goal_comm = [goal_comm]
        self.positions = [self.scenario.arenas[0].get_pos_yaw_spe()]
        self.done = False
        self.args = args
        
    def act(self):
        _, (zq_mu, zq_std), h_q = self.forward(self.o[-1], self.s[-1], self.comm[-1], self.goal_comm[-1], self.a[-1], self.h_q[-1])
        self.zq_mu.append(zq_mu) ; self.zq_std.append(zq_std) ; self.h_q.append(h_q)
        if(self.args.actor_hq):
            a, _, _ = self.actor(self.h_q[-1])
        else:
            a, _, h_actor = self.actor(self.o[-1], self.s[-1], self.a[-1], self.h_actor[-1])
            self.h_actor.append(h_actor)
        a = a.flatten()
        yaw, spe, arms, hands, comm = a[0].item(), a[1].item(), a[2].item(), a[3].item(), a[4:].tolist()
        return((yaw, spe, arms, hands, comm))
        
    def step(self, yaw, speed, arms, hands, comm):
        action = torch.tensor([[yaw] + [speed] + [arms] + [hands] + comm]).flatten().tolist()
        _, self.done, _ = self.scenario.action(0, action, True)
        self.a.append(torch.tensor([[yaw] + [speed] + [arms] + [hands] + comm]).unsqueeze(0))
        o, s, comm, goal_comm = self.scenario.obs(0)
        self.o.append(o) ; self.s.append(s) ; self.comm.append(comm); self.goal_comm.append(goal_comm)
        self.positions.append(self.scenario.arenas[0].get_pos_yaw_spe())
        
        if(self.args.critic_hq): 
            Q1 = self.critic1(self.h_q[-1], self.a[-1])
            Q2 = self.critic2(self.h_q[-1], self.a[-1])
            Qt1 = self.critic1_target(self.h_q[-1], self.a[-1])
            Qt2 = self.critic2_target(self.h_q[-1], self.a[-1]) 
        else:
            Q1, h_critic1 = self.critic1(self.o[-1], self.s[-1], self.comm[-1], self.goal_comm[-1], self.a[-1], self.h_critic1[-1])
            Q2, h_critic2 = self.critic2(self.o[-1], self.s[-1], self.comm[-1], self.goal_comm[-1], self.a[-1], self.h_critic2[-1])
            Qt1, h_critict1 = self.critic1_target(self.o[-1], self.s[-1], self.comm[-1], self.goal_comm[-1], self.a[-1], self.h_critict1[-1])
            Qt2, h_critict2 = self.critic2_target(self.o[-1], self.s[-1], self.comm[-1], self.goal_comm[-1], self.a[-1], self.h_critict2[-1])
            self.h_critic1.append(h_critic1)   ; self.h_critic2.append(h_critic2)
            self.h_critict1.append(h_critict1) ; self.h_critict2.append(h_critict2)
        self.Qs.append([Q1.item(), Q2.item(), Qt1.item(), Qt2.item()])
        return(self.Qs, self.done)
    
    def undo(self):
        step = len(self.a) - 1
        if(step == 0): return
        self.a.pop(-1)
        self.h_q.pop(-1)
        self.h_actor.pop(-1)
        self.h_critic1.pop(-1)
        self.h_critic2.pop(-1)
        self.h_critict1.pop(-1)
        self.h_critict2.pop(-1)
        self.Qs.pop(-1)
        self.zq_mu.pop(-1)
        self.zq_std.pop(-1)
        self.o.pop(-1)
        self.s.pop(-1)
        self.positions.pop(-1)
        pos, yaw, spe = self.positions[-1]
        self.scenario.put_agent_here(step, pos, yaw, spe)
        return(self.Qs)
    
    def predict(self, yaw, speed):
        action = torch.tensor([[[yaw, speed]]])
        with(torch.no_grad()):
            (_, zq_preds_rgbd), (_, zq_preds_spe) = self.forward.get_preds(action, self.zq_mu[-1], self.zq_std[-1], self.h_q[-1], quantity = 1)
        pred_rgbd = zq_preds_rgbd[0].squeeze(0).squeeze(0)
        pred_spe  = zq_preds_spe[0].squeeze(0).squeeze(0)
        return(pred_rgbd, pred_spe)



class GUI(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        
        self.plot_dict_dict = {}
        for name in [f for f in os.listdir("saved") if os.path.isdir("saved/" + f) and f != "thesis_pics"]:
            file = "saved/{}/plot_dict.pickle".format(name)
            with open(file, "rb") as handle: plot_dict = pickle.load(handle)
            self.plot_dict_dict[name] = plot_dict 
        saved_args = list(self.plot_dict_dict.keys())
        saved_args.sort()
        default_args = saved_args[0]
        self.argname_var = tk.StringVar()
        self.argname_var.set(default_args)
        self.argname_label = tk.Label(self, text="Arg Name:")
        self.argname_label.grid(row=0, column=0, sticky="e")
        argname_menu = ttk.OptionMenu(self, self.argname_var, default_args, *saved_args, command=self.update_epoch_agent_num)
        argname_menu.grid(row=0, column=1)
        
        self.epoch_var = tk.StringVar(value='0')
        self.epoch_menu = ttk.OptionMenu(self, self.epoch_var, '0', *["0"])
        self.epoch_label = tk.Label(self, text="Epoch:")
        
        self.agent_num_var = tk.StringVar(value='1')
        self.agent_num_menu = ttk.OptionMenu(self, self.agent_num_var, '1', *["1"])
        self.agent_num_label = tk.Label(self, text="Agent Number:")
        
        step_button = tk.Button(self, text="Step", command=self.step)
        undo_button = tk.Button(self, text="Undo", command=self.undo)
        
        self.yaw_label = tk.Label(self, text="Yaw:")
        self.yaw_var = tk.StringVar()
        self.yaw_entry = tk.Entry(self, textvariable=self.yaw_var)
        
        self.speed_label = tk.Label(self, text="Speed:")
        self.speed_var = tk.StringVar()
        self.speed_entry = tk.Entry(self, textvariable=self.speed_var)
        
        self.arms_label = tk.Label(self, text="Arms:")
        self.arms_var = tk.StringVar()
        self.arms_entry = tk.Entry(self, textvariable=self.arms_var)
        
        self.hands_label = tk.Label(self, text="Hands:")
        self.hands_var = tk.StringVar()
        self.hands_entry = tk.Entry(self, textvariable=self.hands_var)
        
        self.comm_label = tk.Label(self, text="Comm:")
        self.comm_var = tk.StringVar()
        self.comm_entry = tk.Entry(self, textvariable=self.comm_var)
        
        predict_button = tk.Button(self, text="Predict", command=self.predict)
        
        self.fig_1, self.ax_1 = plt.subplots(figsize=(5, 4))
        self.ax_1.plot([0], [0])
        self.ax_1.axis("off")
        self.plot_canvas_1 = FigureCanvasTkAgg(self.fig_1, master=self)
        self.plot_canvas_1.draw()
        
        self.fig_2, self.ax_2 = plt.subplots(figsize=(5, 4))
        self.ax_2.plot([0], [0])
        self.ax_2.axis("off")
        self.plot_canvas_2 = FigureCanvasTkAgg(self.fig_2, master=self)
        self.plot_canvas_2.draw()
                
        self.epoch_label.grid(      row=1, column=0, sticky="e")
        self.epoch_menu.grid(       row=1, column=1)
        self.agent_num_label.grid(  row=2, column=0, sticky="e")
        self.agent_num_menu.grid(   row=2, column=1)
        step_button.grid(           row=4, column=0, columnspan=2)
        self.yaw_label.grid(        row=5, column=0, sticky="e")
        self.yaw_entry.grid(        row=5, column=1)
        self.speed_entry.grid(      row=6, column=1)
        self.speed_label.grid(      row=6, column=0, sticky="e")
        self.arms_entry.grid(      row=7, column=1)
        self.arms_label.grid(      row=7, column=0, sticky="e")
        self.hands_entry.grid(      row=8, column=1)
        self.hands_label.grid(      row=8, column=0, sticky="e")
        self.comm_entry.grid(      row=9, column=1)
        self.comm_label.grid(      row=9, column=0, sticky="e")
        undo_button.grid(           row=10, column=0, columnspan=2)
        predict_button.grid(        row=11, column=0, columnspan=2)
        self.plot_canvas_1.get_tk_widget().grid(row=9, column=0, columnspan=2, padx=5, pady=5)
        self.plot_canvas_2.get_tk_widget().grid(row=9, column=6, columnspan=2, padx=5, pady=5)
        
        self.update_epoch_agent_num()
        
        self.agent_and_episode = None
        
        
        
    def update_plot_dict(self):
        self.plot_dict = self.plot_dict_dict[self.argname_var.get()]
        
    def update_epoch_agent_num(self, *args):
        self.update_plot_dict()
        agent_names = list(self.plot_dict["agent_lists"].keys())[3:]
        agent_nums = sorted(set([s.split('_')[0] for s in agent_names]), key = int)
        epochs = sorted(set([s.split('_')[1] for s in agent_names]), key = int)
                        
        self.agent_num_menu['menu'].delete(0, 'end')
        for agent_num in agent_nums:
            self.agent_num_menu['menu'].add_command(label=agent_num, command=lambda num=agent_num: self.agent_num_var.set(num))
        
        self.epoch_menu['menu'].delete(0, 'end')
        for epoch in epochs:
            self.epoch_menu['menu'].add_command(label=epoch, command=lambda ep=epoch: self.epoch_var.set(ep))
            
            
        
    def add_agent_action(self, yaw, spe, arms, hands, comm):
        yaw = yaw * self.args.max_yaw_change
        yaw = [-self.args.max_yaw_change, self.args.max_yaw_change, yaw] ; yaw.sort() ; yaw = yaw[1]
        yaw = round(degrees(yaw))
        spe = self.args.min_speed + ((spe + 1)/2) * \
            (self.args.max_speed - self.args.min_speed)
        spe = [self.args.min_speed, self.args.max_speed, round(spe)] ; spe.sort() ; spe = spe[1]
        self.yaw_var.set(str(yaw))
        self.speed_var.set(str(spe))
        self.arms_var.set(str(arms))
        self.hands_var.set(str(hands))
        self.comm_var.set(str(comm))
        
    def get_real_action(self):
        yaw = float(self.yaw_var.get())
        spe = float(self.speed_var.get())
        arms = float(self.arms_var.get())
        hands = float(self.hands_var.get())
        comm = ast.literal_eval(self.comm_var.get())
        yaw = radians(yaw) / self.args.max_yaw_change
        speed = (((spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)) * 2) - 1
        return(yaw, speed, arms, hands, comm)
            
            
        
    def step(self):
        if(self.agent_and_episode == None):
            agent_name = "{}_{}".format(self.agent_num_var.get(), self.epoch_var.get())
            print("\narg name: {}.\nAgent num: {}. Epoch: {}.\n".format(
                self.plot_dict["arg_name"], self.agent_num_var.get(), self.epoch_var.get()))
            self.agent_and_episode = Agent_and_Episode(self.plot_dict, agent_name)      
            self.args = self.agent_and_episode.args   
            yaw, speed, arms, hands, comm = self.agent_and_episode.act()      
            self.add_agent_action(yaw, speed, arms, hands, comm)    
        else:
            yaw, speed, arms, hands, comm = self.get_real_action()
            Qs, done = self.agent_and_episode.step(yaw, speed, arms, hands, comm)
            if(done): self.agent_and_episode.scenario.arenas[0].stop() ; self.agent_and_episode = None
            else:     yaw, speed, arms, hands, comm = self.agent_and_episode.act()      
            self.add_agent_action(yaw, speed, arms, hands, comm)   
            self.plot_Qs(Qs)
            
    def undo(self):
        Qs = self.agent_and_episode.undo()
        self.plot_Qs(Qs)
            
    def predict(self):
        yaw, speed = self.get_real_action()
        pred_rgbd, pred_spe = self.agent_and_episode.predict(yaw, speed)
        self.ax_1.clear()
        self.ax_1.axis("off")
        self.ax_1.set_title("Speed {}".format(round(pred_spe.item())))
        self.ax_1.imshow(torch.sigmoid(pred_rgbd[:,:,0:3])) 
        self.plot_canvas_1.draw()
        
    def plot_Qs(self, Qs):
        self.ax_2.clear() 
        if(Qs == None): return
        Q1 = [Q[0] for Q in Qs]
        Q2 = [Q[1] for Q in Qs]
        self.ax_2.plot([i for i in range(len(Q1))], Q1)
        self.ax_2.plot([i for i in range(len(Q2))], Q2)
        self.ax_2.set_title("Qs")
        self.plot_canvas_2.draw()
        


if __name__ == "__main__":
    root = tk.Tk()
    GUI(root).pack(fill="both", expand=True)
    root.mainloop()
# %%
