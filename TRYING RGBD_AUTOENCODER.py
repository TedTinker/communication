#%% 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import choices
import torch

from utils import default_args, make_objects_and_action, action_map, default_args
from task import Task_Runner, Task
from submodules import RGBD_IN, RGBD_OUT, Action_IN

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
import torch.optim as optim
from pytorch_msssim import ssim
from torchinfo import summary as torch_summary

from utils import print, default_args, init_weights, attach_list, detach_list, \
    episodes_steps, pad_zeros, Ted_Conv1d, Ted_Conv2d, create_comm_mask, var, sample, rnn_cnn, duration, ConstrainedConv1d, ConstrainedConv2d
from mtrnn import MTRNN

args = default_args



class In_And_Out(nn.Module):

    def __init__(self, args = default_args):
        super(In_And_Out, self).__init__()  
        
        self.args = args 
        
        rgbd_size = (1, 4, args.image_size * 4, args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.rgbd_in = RGBD_IN(self.args)
        self.action_in = Action_IN(self.args)
        self.rgbd_out = RGBD_OUT(self.args)
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd, action):
        rgbd = self.rgbd_in(rgbd)
        action = self.action_in(action)
        h_w_action = torch.cat([rgbd, action], dim = -1)
        rgbd = self.rgbd_out(h_w_action)
        return(rgbd)



in_and_out = In_And_Out(args = args)
in_and_out.train()
opt = optim.Adam(params=in_and_out.parameters(), lr=.01, weight_decay = .00001) 
task_runner = Task_Runner(Task(actions = 1, objects = 2, shapes = 5, colors = 6), GUI = False)



def example_images(e, before, action, after, guess):
    fig, axs = plt.subplots(4, 1, figsize=(5, 6), gridspec_kw={'wspace':0.1, 'hspace':0.1})

    fig.suptitle("EPOCH {}".format(e))
    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax4 = axs[3]
    
    ax1.set_title("BEFORE")
    ax1.imshow(before)
    ax1.axis('off') 
    rect = patches.Rectangle((-.5, -.5), before.shape[1], before.shape[0], linewidth=4, edgecolor='black', facecolor='none')
    ax1.add_patch(rect)
    
    ax2.set_title("ACTION")
    ax2.text(.25, .5, action)
    
    ax3.set_title("AFTER")
    ax3.imshow(after)
    ax3.axis('off') 
    rect = patches.Rectangle((-.5, -.5), after.shape[1], after.shape[0], linewidth=4, edgecolor='black', facecolor='none')
    ax3.add_patch(rect)
    
    ax4.set_title("GUESS")
    ax4.imshow(guess)
    ax4.axis('off') 
    rect = patches.Rectangle((-.5, -.5), guess.shape[1], guess.shape[0], linewidth=4, edgecolor='black', facecolor='none')
    ax4.add_patch(rect)
        
    plt.tight_layout()
    plt.show()
    


def plot_losses(losses):
    plt.plot(losses)
    plt.show()
    plt.close()
    
    

def make_batch(batch_size = args.batch_size):
    batch_inputs = []
    actions = []
    batch_outputs = []
    while(True):
        task_runner.begin(goal_action = "WATCH")
        done = False
        rgbd, _, _ = task_runner.obs()
        batch_inputs.append(rgbd)
        recommendation = task_runner.get_recommended_action(verbose = False)#True)
        actions.append(recommendation)
        reward, done, win = task_runner.step(recommendation, verbose = False)#True)
        rgbd, _, _ = task_runner.obs()
        batch_outputs.append(rgbd)
        task_runner.done()
        if(len(batch_inputs) == batch_size):
            break
    batch_inputs = torch.cat(batch_inputs)
    actions = torch.stack(actions)
    batch_outputs = torch.cat(batch_outputs)
    return(batch_inputs, actions, batch_outputs)



def epoch(e):
    batch_inputs, actions, batch_outputs = make_batch()
    guesses = in_and_out(batch_inputs, actions)
    
    rgbd_loss = F.binary_cross_entropy_with_logits(guesses.squeeze(1), batch_outputs, reduction = "none")
    rgbd_loss = rgbd_loss.mean()
    opt.zero_grad()
    rgbd_loss.backward()
    opt.step()
    
    guesses = torch.sigmoid(guesses)
    batch_inputs = batch_inputs[0,:,:,0:3].detach()
    actions = actions[0].detach()
    batch_outputs = batch_outputs[0,:,:,0:3].detach()
    guesses = guesses[0,0,:,:,0:3].detach()
    return(rgbd_loss.detach(), batch_inputs, actions, batch_outputs, guesses)

    

rgbd_losses = []
for e in range(1000):
    rgbd_loss, batch_inputs, actions, batch_outputs, guesses = epoch(e)
    rgbd_losses.append(rgbd_loss)
    if(e % 10 == 0):
        example_images(e, batch_inputs, actions, batch_outputs, guesses)
        plot_losses(rgbd_losses)


# %%
