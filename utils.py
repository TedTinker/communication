#%% 

# To do:
#   Adjust so actor only moves arms if it has arms.
#   Test for generalization
#   Maybe separate actors / critics?

import builtins

def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)

import datetime 

start_time = datetime.datetime.now()
    
def duration(start_time = start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)

def estimate_total_duration(proportion_completed, start_time=start_time):
    if(proportion_completed != 0): 
        so_far = datetime.datetime.now() - start_time
        estimated_total = so_far / proportion_completed
        estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    else: estimated_total = "?:??:??"
    return(estimated_total)



import argparse, ast, os, pickle
from math import exp, pi
import numpy as np
from time import sleep

if(os.getcwd().split("/")[-1] != "communication"): os.chdir("communication")

import torch
from torch import nn 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

def literal(arg_string): return(ast.literal_eval(arg_string))

# Meta 
parser.add_argument("--arg_title",          type=str,        default = "default") 
parser.add_argument("--arg_name",           type=str,        default = "default") 
parser.add_argument("--agents",             type=int,        default = 1)
parser.add_argument("--previous_agents",    type=int,        default = 0)
parser.add_argument("--init_seed",          type=float,      default = 777)
parser.add_argument('--device',             type=str,        default = "cpu")
parser.add_argument('--comp',               type=str,        default = "deigo")

# Scenario 
parser.add_argument('--scenario_list',      type=literal,    default = [(1,False,False)])#(3,False,False),(3,False,True),(3,True,True)]) # Are there two agents? Do we use all goals?
parser.add_argument('--max_steps',          type=int,        default = 20)
parser.add_argument('--reward',             type=float,      default = 1)
parser.add_argument('--step_lim_punishment',type=float,      default = -1)
parser.add_argument('--step_cost',          type=float,      default = .99999)
parser.add_argument('--body_size',          type=float,      default = 2)    
parser.add_argument('--image_size',         type=int,        default = 8)
parser.add_argument('--max_yaw_change',     type=float,      default = pi/2)
parser.add_argument('--min_speed',          type=float,      default = 0)
parser.add_argument('--max_speed',          type=float,      default = 300)
parser.add_argument('--steps_per_step',     type=int,        default = 5)
parser.add_argument('--symbols',            type=int,        default = 20)

# Module 
parser.add_argument('--hidden_size',        type=int,        default = 32)   
parser.add_argument('--state_size',         type=int,        default = 32)
parser.add_argument('--actor_hq',           type=literal,    default = True)
parser.add_argument('--critic_hq',          type=literal,    default = False)
parser.add_argument('--forward_lr',         type=float,      default = .01)
parser.add_argument('--alpha_lr',           type=float,      default = .01) 
parser.add_argument('--actor_lr',           type=float,      default = .01)
parser.add_argument('--critic_lr',          type=float,      default = .01)
parser.add_argument('--action_prior',       type=str,        default = "normal")
parser.add_argument("--tau",                type=float,      default = 1)      # For soft-updating target critics

# Complexity 
parser.add_argument('--std_min',            type=int,        default = exp(-20))
parser.add_argument('--std_max',            type=int,        default = exp(2))
parser.add_argument("--beta",               type=float,      default = 0)

# Entropy
parser.add_argument("--alpha",              type=str,        default = 0)        # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float,      default = -4)       # Soft-Actor-Critic entropy aim

# Curiosity
parser.add_argument("--curiosity",          type=str,        default = "none")     # Which kind of curiosity
parser.add_argument("--naive_eta",          type=float,      default = 10)        # Scale curiosity
parser.add_argument("--free_eta",           type=float,      default = 10)        # Scale curiosity
parser.add_argument("--dkl_max",            type=float,      default = 1)        

# Memory buffer
parser.add_argument('--capacity',           type=int,        default = 250)

# Training
parser.add_argument('--epochs',             type=literal,    default = [1000])#,10,10,10])
parser.add_argument('--steps_per_epoch',    type=int,        default = 10)
parser.add_argument('--batch_size',         type=int,        default = 128)
parser.add_argument('--elbo_num',           type=int,        default = 1)
parser.add_argument('--GAMMA',              type=float,      default = .9)
parser.add_argument("--d",                  type=int,        default = 2)        # Delay to train actors
parser.add_argument('--speed_scalar',       type=float,      default = .0001)
parser.add_argument('--comm_scalar',        type=float,      default = .0001)

# Saving data
parser.add_argument('--keep_data',           type=int,        default = 25)

parser.add_argument('--epochs_per_gen',      type=int,        default = 100)

parser.add_argument('--epochs_per_pred_list',type=int,        default = 100)
parser.add_argument('--agents_per_pred_list',type=int,        default = 1)
parser.add_argument('--episodes_in_pred_list',type=int,       default = 1)
parser.add_argument('--samples_per_pred',    type=int,        default = 2)

parser.add_argument('--epochs_per_agent_list',type=int,       default = 100)
parser.add_argument('--agents_per_agent_list',type=int,       default = 1)



try:
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
except:
    import sys ; sys.argv=[''] ; del sys           # Comment this out when using bash
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()

if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

for arg in vars(default_args):
    if(getattr(default_args, arg) == "None"):  default_args.arg = None
    if(getattr(default_args, arg) == "True"):  default_args.arg = True
    if(getattr(default_args, arg) == "False"): default_args.arg = False
    if(getattr(args, arg) == "None"):  args.arg = None
    if(getattr(args, arg) == "True"):  args.arg = True
    if(getattr(args, arg) == "False"): args.arg = False



args_not_in_title = ["arg_title", "id", "agents", "previous_agents", "init_seed", "keep_data", "epochs_per_pred_list", "episodes_in_pred_list", "agents_per_pred_list", "epochs_per_pos_list", "episodes_in_pos_list", "agents_per_pos_list"]
def get_args_title(default_args, args):
    if(args.arg_title[:3] == "___"): return(args.arg_title)
    name = "" ; first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index("arg_name")))
    for arg in arg_list:
        if(arg in args_not_in_title): pass 
        else: 
            default, this_time = getattr(default_args, arg), getattr(args, arg)
            if(this_time == default): pass
            elif(arg == "arg_name"):
                name += "{} (".format(this_time)
            else: 
                if first: first = False
                else: name += ", "
                name += "{}: {}".format(arg, this_time)
    if(name == ""): name = "default" 
    else:           name += ")"
    if(name.endswith(" ()")): name = name[:-3]
    parts = name.split(',')
    name = "" ; line = ""
    for i, part in enumerate(parts):
        if(len(line) > 50 and len(part) > 2): name += line + "\n" ; line = ""
        line += part
        if(i+1 != len(parts)): line += ","
    name += line
    return(name)

args.arg_title = get_args_title(default_args, args)

folder = "saved/" + args.arg_name
if(args.arg_title[:3] != "___" and not args.arg_name in ["default", "finishing_dictionaries", "plotting", "plotting_predictions", "plotting_positions"]):
    try: os.mkdir(folder)
    except: pass
    try: os.mkdir("saved/thesis_pics")
    except: pass
if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

if(args == default_args): print("Using default arguments.")
else:
    for arg in vars(default_args):
        default, this_time = getattr(default_args, arg), getattr(args, arg)
        if(this_time == default): pass
        else: print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
class ConstrainedConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, input):
        return nn.functional.conv_transpose2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.output_padding, self.groups, self.dilation)
        
class Ted_Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernels = [(1,1),(3,3),(5,5)]):
        super(Ted_Conv2d, self).__init__()
        
        self.Conv2ds = nn.ModuleList()
        for kernel, out_channel in zip(kernels, out_channels):
            padding = ((kernel[0]-1)//2, (kernel[1]-1)//2)
            layer = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = in_channels,
                    out_channels = out_channel,
                    kernel_size = kernel,
                    padding = padding,
                    padding_mode = "reflect"),
                nn.PReLU())
            self.Conv2ds.append(layer)
                
    def forward(self, x):
        y = []
        for Conv2d in self.Conv2ds: y.append(Conv2d(x)) 
        return(torch.cat(y, dim = -3))


    
def dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)



def load_dicts(args):
    if(os.getcwd().split("/")[-1] != "saved"): os.chdir("saved")
    plot_dicts = [] ; min_max_dicts = []
        
    complete_order = args.arg_title[3:-3].split("+")
    order = [o for o in complete_order if not o in ["empty_space", "break"]]

    for name in order:
        got_plot_dicts = False ; got_min_max_dicts = False
        while(not got_plot_dicts):
            try:
                with open(name + "/" + "plot_dict.pickle", "rb") as handle: 
                    plot_dicts.append(pickle.load(handle)) ; got_plot_dicts = True
            except: print("Stuck trying to get {}'s plot_dicts...".format(name)) ; sleep(1)
        while(not got_min_max_dicts):
            try:
                with open(name + "/" + "min_max_dict.pickle", "rb") as handle: 
                    min_max_dicts.append(pickle.load(handle)) ; got_min_max_dicts = True 
            except: print("Stuck trying to get {}'s min_max_dicts...".format(name)) ; sleep(1)
    
    min_max_dict = {}
    for key in plot_dicts[0].keys():
        if(not key in ["args", "arg_title", "arg_name", "pred_lists", "pos_lists", "agents_lists", "spot_names"]):
            minimum = None ; maximum = None
            for mm_dict in min_max_dicts:
                if(mm_dict[key] != (None, None)):
                    if(minimum == None):             minimum = mm_dict[key][0]
                    elif(minimum > mm_dict[key][0]): minimum = mm_dict[key][0]
                    if(maximum == None):             maximum = mm_dict[key][1]
                    elif(maximum < mm_dict[key][1]): maximum = mm_dict[key][1]
            min_max_dict[key] = (minimum, maximum)
            
    complete_order = []

    for arg_name in complete_order: 
        if(arg_name in ["break", "empty_space"]): 
            complete_order.append(arg_name)
        else:
            for plot_dict in plot_dicts:
                if(plot_dict["args"].arg_name == arg_name):    
                    complete_order.append(arg_name) ; plot_dicts.append(plot_dict) 
                    
    while(len(complete_order) > 0 and complete_order[0] in ["break", "empty_space"]): complete_order.pop(0)
            
    return(plot_dicts, min_max_dict, complete_order, plot_dicts)

shapes = [f.name for f in os.scandir("pybullet_data/shapes")]
shapes.sort()
shapes = shapes[:1]
colors = [(1,0,0,1),(0,1,0,1),(0,0,1,1),(0,1,1,1),(1,0,1,1),(1,1,0,1)]
colors = colors[:1]
goals = ["touch"]#, "watch", "push", "pull", "topple"]

test_objects = {shape : [color_1, color_2] for shape, color_1, color_2 in zip(shapes, colors, colors[1:] + [colors[0]])}
test_objects = {shape : [] for shape in shapes}

if __name__ == "__main__":
    print(shapes)
    print(colors)
    print(goals)
    print(test_objects)

# %%
