#%% 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import choices
import torch

from utils import default_args, make_objects_and_action, action_map, default_args, custom_loss
from task import Task_Runner, Task
from submodules import Obs_IN, Obs_OUT, Action_IN
from pvrnn import PVRNN

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
import torch.optim as optim
from kornia.color import rgb_to_hsv 
from pytorch_msssim import ssim
from torchinfo import summary as torch_summary

from utils import print, default_args, init_weights, attach_list, detach_list, many_onehots_to_strings, \
    episodes_steps, pad_zeros, Ted_Conv1d, Ted_Conv2d, var, sample, rnn_cnn, duration, ConstrainedConv1d, ConstrainedConv2d
from mtrnn import MTRNN
from arena import Arena, get_physics

args = default_args
args.objects = 4



physicsClient = get_physics(GUI = False, time_step = args.time_step, steps_per_step = args.steps_per_step)
arena_1 = Arena(physicsClient)
arena_2 = None
task_runner = Task_Runner(Task(actions = -1, objects = 4, shapes = 5, colors = 6), arena_1, arena_2)

forward = PVRNN(args = args)
forward.train()
opt = optim.Adam(
    params=forward.parameters(), 
    lr=.0003,#args.forward_lr, 
    weight_decay = .00001) 



def example_images(e, reals, action, nexts, guess):
        
    episodes = reals[0].shape[0]
    steps = reals[0].shape[1]
    fig, axs = plt.subplots(3 * episodes, steps, figsize=(3 * steps, 11 * episodes))

    fig.suptitle("EPOCH {}".format(e))
    fig.patch.set_facecolor('white')
    
    def one_part(this, title, image, comm, other):
        this.set_title("{} :\nCOMM ({}),\nOTHER ({})".format(title, comm, [round(o, 1) for o in other.tolist()])) # comm, [round(o,2) for o in other.tolist()]))
        this.imshow(image)
        this.set_xticks([])
        this.set_yticks([])
        
    for e in range(episodes):
        for s in range(steps):
            one_part(
                axs[e * 3 + 0, s], 
                "BEFORE", 
                reals[0][e, s], 
                reals[1][e][s], 
                reals[2][e, s])
            one_part(
                axs[e * 3 + 1, s], 
                "AFTER {}".format([round(a, 2) for a in action[e,s].tolist()]), 
                nexts[0][e, s], 
                nexts[1][e][s], 
                nexts[2][e, s])
            one_part(
                axs[e * 3 + 2, s], 
                "GENERATED", 
                guess[0][e, s], 
                guess[1][e][s], 
                guess[2][e, s])
        
    plt.show()
    plt.close()
    


def plot_losses(losses):
    plt.plot([loss[0] for loss in losses], color="red", label="RGBD")
    plt.plot([loss[1] for loss in losses], color="blue", label="Comm")
    plt.plot([loss[2] for loss in losses], color="green", label="Other")
    plt.legend()  
    plt.show()
    plt.close()
    
    

def make_batch(batch_size = args.batch_size):
    real_rgbd = []
    real_comm = []
    real_other = []
    actions = []
    for episode in range(batch_size):
        task_runner.begin()
        rgbds = []
        comms = []
        others = []
        acts = []
        for step in range(args.max_steps):
            rgbd, comm, other = task_runner.obs()
            rgbds.append(rgbd)
            comms.append(comm)
            others.append(other)
            recommendation = task_runner.get_recommended_action(verbose = False)#True)
            acts.append(recommendation.unsqueeze(0))
            reward, done, win = task_runner.step(recommendation, verbose = False)#True)
        rgbd, comm, other = task_runner.obs()
        rgbds.append(rgbd)
        comms.append(comm)
        others.append(other)
        rgbds = torch.cat(rgbds, dim = 0).unsqueeze(0)
        comms = torch.cat(comms, dim = 0).unsqueeze(0)
        others = torch.cat(others, dim = 0).unsqueeze(0)
        acts = torch.cat(acts, dim = 0)
        real_rgbd.append(rgbds)
        real_comm.append(comms)
        real_other.append(others)
        actions.append(acts)
        task_runner.done()
    real_rgbd = torch.cat(real_rgbd, dim = 0)
    real_comm = torch.cat(real_comm, dim = 0)
    real_other = torch.cat(real_other, dim = 0)
    actions = torch.stack(actions, dim = 0)
    actions = torch.cat([actions, torch.zeros(actions[:,0].unsqueeze(1).shape).to(args.device)], dim = 1)
    next_rgbd = real_rgbd[:,1:]
    next_comm = real_comm[:,1:]
    next_other = real_other[:,1:]
    return(real_rgbd, real_comm, real_other, actions, next_rgbd, next_comm, next_other)



def add_hsv(rgbd):
    episodes, steps = episodes_steps(rgbd)
    rgbd = rgbd.reshape(episodes * steps, rgbd.shape[2], rgbd.shape[3], rgbd.shape[4]).permute(0, -1, 1, 2)
    rgb = rgbd[:,:-1]
    hsv = rgb_to_hsv(rgb)
    hues = hsv[:,1]
    hue_sin = torch.sin(hues).unsqueeze(1)
    hue_cos = torch.cos(hues).unsqueeze(1)
    hsv = hsv[:,1:]
    hsv = torch.cat((hue_sin, hue_cos, hsv), dim=1)
    rgbdhsv = torch.cat((rgbd, hsv), dim=1)
    rgbdhsv = rgbdhsv.reshape(episodes, steps, rgbdhsv.shape[1], rgbdhsv.shape[2], rgbdhsv.shape[3]).permute(0, 1, 3, 4, 2)
    return(rgbdhsv)



def epoch(e):
    print(e, end = "... ")
    real_rgbd, real_comm, real_other, actions, next_rgbd, next_comm, next_other = make_batch()
        
    _, _, (guess_rgbd, guess_comm, guess_other), _ = forward(None, real_rgbd, real_comm, real_other, actions, torch.zeros_like(real_comm))
        
    next_rgbdhsv = add_hsv(next_rgbd)
    guess_rgbdhsv = add_hsv(guess_rgbd)
        
    rgbd_loss = F.binary_cross_entropy(guess_rgbd, next_rgbd, reduction = "none")
    rgbd_loss = rgbd_loss.mean() * args.rgbd_scaler
    
    next_comm_temp = next_comm.reshape((next_comm.shape[0] * next_comm.shape[1], args.max_comm_len, args.comm_shape))
    next_comm_temp = torch.argmax(next_comm_temp, dim = -1)
    guess_comm_temp = guess_comm.reshape((guess_comm.shape[0] * guess_comm.shape[1], args.max_comm_len, args.comm_shape))
    guess_comm_temp = guess_comm_temp.transpose(1,2)
     
    #comm_loss = custom_loss(guess_comm_temp, next_comm_temp, max_shift = 0)    
    comm_loss = F.cross_entropy(guess_comm_temp, next_comm_temp, reduction = "none")
    comm_loss = comm_loss.mean() * args.comm_scaler
    
    other_loss = F.binary_cross_entropy(guess_other, next_other, reduction = "none")
    other_loss = other_loss.mean() * args.other_scaler
    
    loss = rgbd_loss + comm_loss + other_loss
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    episodes = 3
    steps = args.max_steps

    real_rgbd = real_rgbd[:episodes,:steps,:,:,0:3].detach()
    real_comm = real_comm[:episodes,:steps].detach()
    real_comm = many_onehots_to_strings(real_comm)
    real_other = real_other[:episodes,:steps].detach()
        
    actions = actions[:episodes,:steps].detach()
    
    next_rgbd = next_rgbd[:episodes,:steps,:,:,0:3].detach()
    next_comm = next_comm[:episodes,:steps].detach()
    next_comm = many_onehots_to_strings(next_comm)
    next_other = next_other[:episodes,:steps].detach()
    
    guess_rgbd = guess_rgbd[:episodes,:steps,:,:,0:3].detach()
    guess_comm = guess_comm[:episodes,:steps].detach()
    guess_comm = many_onehots_to_strings(guess_comm)
    guess_other = guess_other[:episodes,:steps].detach()
        
    return(
        (rgbd_loss.detach(), comm_loss.detach(), other_loss.detach()),
        (real_rgbd, real_comm, real_other), 
        actions, 
        (next_rgbd, next_comm, next_other), 
        (guess_rgbd, guess_comm, guess_other))

    

losses = []
for e in range(2000):
    loss, real, actions, next, guess = epoch(e)
    losses.append(loss)
    if(e % 10 == 0):
        example_images(e, real, actions, next, guess)
        plot_losses(losses)


# %%
