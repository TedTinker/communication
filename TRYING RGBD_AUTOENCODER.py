#%% 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import choices
import torch

from utils import default_args, make_objects_and_action, action_map, default_args, custom_loss
from task import Task_Runner, Task
from submodules import Obs_IN, Obs_OUT, Action_IN

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



class In_And_Out(nn.Module):

    def __init__(self, args = default_args):
        super(In_And_Out, self).__init__()  
        
        self.args = args 
        
        rgbd_size = (1, 4, args.image_size * 4, args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.obs_in = Obs_IN(self.args)
        self.action_in = Action_IN(self.args)
        
        self.pretend_pvrnn = nn.Sequential(
            nn.Linear(
                in_features = self.args.encode_obs_size,
                out_features = self.args.pvrnn_mtrnn_size),
            nn.PReLU())
        
        self.obs_out = Obs_OUT(self.args)
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd, comm, other, action):
        obs = self.obs_in(rgbd, comm, other)
        action = self.action_in(action)
        h = self.pretend_pvrnn(obs)
        h_w_action = torch.cat([h, action], dim = -1)
        pred_rgbd, pred_comm, pred_other = self.obs_out(h_w_action)
        return(pred_rgbd, pred_comm, pred_other)



physicsClient = get_physics(GUI = False)
arena_1 = Arena(physicsClient)
arena_2 = None
task_runner = Task_Runner(Task(actions = 5, objects = 4, shapes = 5, colors = 6), arena_1, arena_2)

in_and_out = In_And_Out(args = args)
in_and_out.train()
opt = optim.Adam(params=in_and_out.parameters(), lr=.01, weight_decay = .00001) 



def example_images(e, reals, action, nexts, gens):
        
    num = len(reals[0])
    fig, axs = plt.subplots(3, num, figsize=(5 * num, 10))

    fig.suptitle("EPOCH {}".format(e))
    fig.patch.set_facecolor('white')
    
    def one_part(this, title, image, comm, other):
        this.set_title("{} :\nCOMM ({}),\nOTHER ({})".format(title, comm, [round(o, 2) for o in other.tolist()])) # comm, [round(o,2) for o in other.tolist()]))
        this.imshow(image)
        this.set_xticks([])
        this.set_yticks([])
        
    for n in range(num):
        one_part(axs[0,n], "BEFORE", reals[0][n], reals[1][n], reals[2][n])
        one_part(axs[1,n], "AFTER {}".format([round(a, 2) for a in action[n].tolist()[0]]), nexts[0][n], nexts[1][n], nexts[2][n])
        one_part(axs[2,n], "GENERATED", gens[0][n], gens[1][n], gens[2][n])
        
    plt.show()
    plt.close()
    


def plot_losses(losses):
    plt.plot([loss[0] for loss in losses], color = "red")
    plt.plot([loss[1] for loss in losses], color = "blue")
    plt.plot([loss[2] for loss in losses], color = "green")
    plt.show()
    plt.close()
    
    

def make_batch(batch_size = args.batch_size):
    real_rgbd = []
    real_comms = []
    real_others = []
    actions = []
    next_rgbd = []
    next_comms = []
    next_others = []
    while(True):
        task_runner.begin()
        rgbd, comm, other = task_runner.obs()
        real_rgbd.append(rgbd)
        real_comms.append(comm)
        real_others.append(other)
        recommendation = task_runner.get_recommended_action(verbose = False)#True)
        actions.append(recommendation)
        reward, done, win = task_runner.step(recommendation, verbose = False)#True)
        rgbd, comm, other = task_runner.obs()
        next_rgbd.append(rgbd)
        next_comms.append(comm)
        next_others.append(other)
        task_runner.done()
        if(len(real_rgbd) == batch_size):
            break
    real_rgbd = torch.cat(real_rgbd).unsqueeze(1)
    real_comms = torch.cat(real_comms).unsqueeze(1)
    real_others = torch.cat(real_others).unsqueeze(1)
    actions = torch.stack(actions).unsqueeze(1)
    next_rgbd = torch.cat(next_rgbd).unsqueeze(1)
    next_comms = torch.cat(next_comms).unsqueeze(1)
    next_others = torch.cat(next_others).unsqueeze(1)
    return(real_rgbd, real_comms, real_others, actions, next_rgbd, next_comms, next_others)



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
    real_rgbd, real_comms, real_others, actions, next_rgbd, next_comms, next_others = make_batch()
        
    guess_rgbd, guess_comms, guess_others = in_and_out(real_rgbd, real_comms, real_others, actions)
    
    next_rgbdhsv = add_hsv(next_rgbd)
    guess_rgbdhsv = add_hsv(guess_rgbd)
        
    rgbd_loss = F.binary_cross_entropy(guess_rgbd, next_rgbd, reduction = "none")
    rgbd_loss = rgbd_loss.mean()
    
    next_comms_temp = next_comms.reshape((next_comms.shape[0] * next_comms.shape[1], args.max_comm_len, args.comm_shape))
    next_comms_temp = torch.argmax(next_comms_temp, dim = -1)
    guess_comms_temp = guess_comms.reshape((guess_comms.shape[0] * guess_comms.shape[1], args.max_comm_len, args.comm_shape))
    guess_comms_temp = guess_comms_temp.transpose(1,2)
     
    comm_loss = custom_loss(guess_comms_temp, next_comms_temp, max_shift = 0)    
    #comm_loss = F.cross_entropy(guess_comms_temp, next_comms_temp, reduction = "none")
    comm_loss = comm_loss.mean()
    
    other_loss = F.binary_cross_entropy(guess_others, next_others, reduction = "none")
    other_loss = other_loss.mean()
    
    loss = 100 * rgbd_loss + comm_loss + other_loss
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    num = 5

    real_rgbd = real_rgbd[:num,0,:,:,0:3].detach()
    real_comms = real_comms[:num].detach()
    real_comms = many_onehots_to_strings(real_comms)
    real_others = real_others[:num,0].detach()
        
    actions = actions[:num].detach()
    
    next_rgbd = next_rgbd[:num,0,:,:,0:3].detach()
    next_comms = next_comms[:num,0].detach()
    next_comms = many_onehots_to_strings(next_comms)
    next_others = next_others[:num,0].detach()
    
    guess_rgbd = guess_rgbd[:num,0,:,:,0:3].detach()
    guess_comms = guess_comms[:num,0].detach()
    guess_comms = many_onehots_to_strings(guess_comms)
    guess_others = guess_others[:num,0].detach()
        
    return(
        (rgbd_loss.detach(), comm_loss.detach(), other_loss.detach()),
        (real_rgbd, real_comms, real_others), 
        actions, 
        (next_rgbd, next_comms, next_others), 
        (guess_rgbd, guess_comms, guess_others))

    

losses = []
for e in range(1000):
    loss, real, actions, next, guess = epoch(e)
    losses.append(loss)
    if(e % 10 == 0):
        example_images(e, real, actions, next, guess)
        plot_losses(losses)


# %%
