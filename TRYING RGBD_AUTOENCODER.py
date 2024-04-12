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
    episodes_steps, pad_zeros, Ted_Conv1d, Ted_Conv2d, var, sample, rnn_cnn, duration, ConstrainedConv1d, ConstrainedConv2d, dkl
from buffer import RecurrentReplayBuffer
from mtrnn import MTRNN
from arena import Arena, get_physics

args = default_args
num_objects = 6
args.objects = num_objects

training_memory = RecurrentReplayBuffer(args)
testing_memory = RecurrentReplayBuffer(args)

physicsClient = get_physics(GUI = False, time_step = args.time_step, steps_per_step = args.steps_per_step)
arena_1 = Arena(physicsClient)
arena_2 = None
task_runner = Task_Runner(Task(actions = [-1], objects = num_objects, shapes = [0, 1, 2, 3, 4], colors = [0, 1, 2, 3, 4, 5]), arena_1, arena_2)

forward = PVRNN(args = args)
forward.train()
opt = optim.Adam(
    params=forward.parameters(), 
    lr=args.forward_lr, 
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
    


def plot_losses(losses, test_losses):
    plt.plot([loss[0] for loss in losses], color=(1, 0, 0), label="RGBD")
    plt.plot([loss[1] for loss in losses], color=(0, 1, 0), label="Comm")
    plt.plot([loss[2] for loss in losses], color=(0, 0, 1), label="Other")
    
    plt.plot([test_loss[0] for test_loss in test_losses], color=(1, .6, .6), label="test RGBD")
    plt.plot([test_loss[1] for test_loss in test_losses], color=(.6, 1, .6), label="test Comm")
    plt.plot([test_loss[2] for test_loss in test_losses], color=(.6, .6, 1), label="test Other")
    
    plt.legend(loc='lower left')  
    plt.show()
    plt.close()
    
    
    
def add_episode(test = False):
    task_runner.begin()
    rgbds = []
    comms = []
    others = []
    actions = []
    rewards = []
    dones = []
    for step in range(args.max_steps):
        rgbd, comm, other = task_runner.obs()
        action = task_runner.get_recommended_action(verbose = False)#True)
        action = torch.tensor([0, 0, 1])
        raw_reward, distance_reward, angle_reward, distance_reward_2, angle_reward_2, done, win = task_runner.step(action, verbose = False)#True)
        rgbds.append(rgbd)
        comms.append(comm)
        others.append(other)
        actions.append(action.unsqueeze(0))
        rewards.append(raw_reward + distance_reward + angle_reward)
        dones.append(done)
    rgbd, comm, other = task_runner.obs()
    rgbds.append(rgbd)
    comms.append(comm)
    others.append(other)
    task_runner.done()
    if(test):
        memory = testing_memory
    else:
        memory = training_memory
    for i in range(len(rewards)):
        memory.push(
            rgbd = rgbds[i], 
            communication_in = comms[i], 
            other = others[i], 
            action = actions[i], 
            recommended_action = actions[i], 
            communication_out = comms[i], 
            reward = rewards[i], 
            next_rgbd = rgbds[i+1], 
            next_communication_in = comms[i+1], 
            next_other = others[i+1], 
            done = dones[i])



def epoch(e):
    print(e, end = "... ")
    add_episode()
    rgbds, comms_in, others, actions, comms_out, recommended_actions, rewards, dones, masks = training_memory.sample(args.batch_size)
    
    rgbds = torch.from_numpy(rgbds)
    comms_in = torch.from_numpy(comms_in)
    others = torch.from_numpy(others)
    actions = torch.from_numpy(actions)
    comms_out = torch.from_numpy(comms_out)
    recommended_actions = torch.from_numpy(recommended_actions)
    rewards = torch.from_numpy(rewards)
    dones = torch.from_numpy(dones)
    masks = torch.from_numpy(masks)
    actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape).to(args.device), actions], dim = 1)
    comms_out = torch.cat([torch.zeros(comms_out[:,0].unsqueeze(1).shape).to(args.device), comms_out], dim = 1)
    all_masks = torch.cat([torch.ones(masks.shape[0], 1, 1).to(args.device), masks], dim = 1)   
    episodes = rewards.shape[0]
    steps = rewards.shape[1]
    
    forward.train()
    
    (zp_mu, zp_std, hps), (zq_mu, zq_std, hqs), (guess_rgbd, guess_comm, guess_other), _ = forward(torch.zeros((episodes, args.layers, args.pvrnn_mtrnn_size)), rgbds, comms_in, others, actions, comms_out)
    
    real_rgbd = rgbds
    real_comm = comms_in
    real_other = others
    
    next_rgbd = real_rgbd[:,1:]
    next_comm = real_comm[:,1:]
    next_other = real_other[:,1:]
    
    real_rgbd = real_rgbd[:,:-1]
    real_comm = real_comm[:,:-1]
    real_other = real_other[:,:-1]
                        
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
    
    complexity_for_hidden_state = [dkl(zq_mu[:,:,layer], zq_std[:,:,layer], zp_mu[:,:,layer], zp_std[:,:,layer]).mean(-1).unsqueeze(-1) * all_masks for layer in range(args.layers)] 
    complexity          = sum([args.beta[layer] * complexity_for_hidden_state[layer].mean() for layer in range(args.layers)])    
    
    loss = rgbd_loss + comm_loss + other_loss + complexity
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    rgbds, comms_in, others, actions, comms_out, recommended_actions, rewards, dones, masks = testing_memory.sample(args.batch_size)
    
    rgbds = torch.from_numpy(rgbds)
    comms_in = torch.from_numpy(comms_in)
    others = torch.from_numpy(others)
    actions = torch.from_numpy(actions)
    comms_out = torch.from_numpy(comms_out)
    recommended_actions = torch.from_numpy(recommended_actions)
    rewards = torch.from_numpy(rewards)
    dones = torch.from_numpy(dones)
    masks = torch.from_numpy(masks)
    actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape).to(args.device), actions], dim = 1)
    comms_out = torch.cat([torch.zeros(comms_out[:,0].unsqueeze(1).shape).to(args.device), comms_out], dim = 1)
    all_masks = torch.cat([torch.ones(masks.shape[0], 1, 1).to(args.device), masks], dim = 1)   
    episodes = rewards.shape[0]
    steps = rewards.shape[1]
    
    
    
    forward.eval()
    
    _, _, (test_guess_rgbd, test_guess_comm, test_guess_other), _ = forward(torch.zeros((episodes, args.layers, args.pvrnn_mtrnn_size)), rgbds, comms_in, others, actions, comms_out)
    
    test_real_rgbd = rgbds
    test_real_comm = comms_in
    test_real_other = others
    
    test_next_rgbd = test_real_rgbd[:,1:]
    test_next_comm = test_real_comm[:,1:]
    test_next_other = test_real_other[:,1:]
    
    test_real_rgbd = test_real_rgbd[:,:-1]
    test_real_comm = test_real_comm[:,:-1]
    test_real_other = test_real_other[:,:-1]
    
    test_rgbd_loss = F.binary_cross_entropy(test_guess_rgbd, test_next_rgbd, reduction = "none")
    test_rgbd_loss = test_rgbd_loss.mean() * args.rgbd_scaler
    
    test_next_comm_temp = test_next_comm.reshape((test_next_comm.shape[0] * test_next_comm.shape[1], args.max_comm_len, args.comm_shape))
    test_next_comm_temp = torch.argmax(test_next_comm_temp, dim = -1)
    test_guess_comm_temp = test_guess_comm.reshape((test_guess_comm.shape[0] * test_guess_comm.shape[1], args.max_comm_len, args.comm_shape))
    test_guess_comm_temp = test_guess_comm_temp.transpose(1,2)
     
    #comm_loss = custom_loss(guess_comm_temp, next_comm_temp, max_shift = 0)    
    test_comm_loss = F.cross_entropy(test_guess_comm_temp, test_next_comm_temp, reduction = "none")
    test_comm_loss = test_comm_loss.mean() * args.comm_scaler
    
    test_other_loss = F.binary_cross_entropy(test_guess_other, test_next_other, reduction = "none")
    test_other_loss = test_other_loss.mean() * args.other_scaler
    
    
        
    real_rgbd = torch.cat([real_rgbd[0].unsqueeze(0), test_real_rgbd[0].unsqueeze(0)])
    real_comm = torch.cat([real_comm[0].unsqueeze(0), test_real_comm[0].unsqueeze(0)])
    real_other = torch.cat([real_other[0].unsqueeze(0), test_real_other[0].unsqueeze(0)])
    
    next_rgbd = torch.cat([next_rgbd[0].unsqueeze(0), test_next_rgbd[0].unsqueeze(0)])
    next_comm = torch.cat([next_comm[0].unsqueeze(0), test_next_comm[0].unsqueeze(0)])
    next_other = torch.cat([next_other[0].unsqueeze(0), test_next_other[0].unsqueeze(0)])
    
    guess_rgbd = torch.cat([guess_rgbd[0].unsqueeze(0), test_guess_rgbd[0].unsqueeze(0)])
    guess_comm = torch.cat([guess_comm[0].unsqueeze(0), test_guess_comm[0].unsqueeze(0)])
    guess_other = torch.cat([guess_other[0].unsqueeze(0), test_guess_other[0].unsqueeze(0)])
        
    episodes = 2
    steps = args.max_steps

    real_rgbd = real_rgbd[:episodes,:steps,:,:,0:3].detach()
    real_comm = real_comm[:episodes,:steps].detach()
    real_comm = many_onehots_to_strings(real_comm)
    real_other = real_other[:episodes,:steps].detach()
        
    actions = actions[:episodes,1:steps+1].detach()
    
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
        (test_rgbd_loss.detach(), test_comm_loss.detach(), test_other_loss.detach()),
        (real_rgbd, real_comm, real_other), 
        actions, 
        (next_rgbd, next_comm, next_other), 
        (guess_rgbd, guess_comm, guess_other))

    
for i in range(args.batch_size):
    add_episode()
for i in range(args.batch_size):
    add_episode(test = True)

losses = []
test_losses = []
for e in range(1500):
    loss, test_loss, real, actions, next, guess = epoch(e)
    losses.append(loss)
    test_losses.append(test_loss)
    if(e % 10 == 0):
        example_images(e, real, actions, next, guess)
        plot_losses(losses, test_losses)


# %%
