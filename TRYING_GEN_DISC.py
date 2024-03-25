#%% 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import choices
import torch

from utils import default_args, make_objects_and_action, action_map, default_args
from task import Task_Runner, Task
from submodules import RGBD_IN, RGBD_OUT, Action_IN, Discriminator, Obs_IN, Obs_OUT

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
import torch.optim as optim
from pytorch_msssim import ssim
from torchgan.losses import WassersteinGeneratorLoss as WG
from torchgan.losses import WassersteinDiscriminatorLoss as WD
from torchinfo import summary as torch_summary

from utils import print, default_args, init_weights, attach_list, detach_list, onehots_to_string, \
    episodes_steps, pad_zeros, Ted_Conv1d, Ted_Conv2d, create_comm_mask, var, sample, rnn_cnn, duration, ConstrainedConv1d, ConstrainedConv2d
from mtrnn import MTRNN
from arena import Arena, get_physics

args = default_args



class Generator(nn.Module):

    def __init__(self, args = default_args):
        super(Generator, self).__init__()  
        
        self.args = args 
        
        #self.obs_in = Obs_IN(self.args)
        self.rgbd_in = RGBD_IN(self.args)
        self.pretend_pvrnn = nn.Sequential(
            nn.Linear(
                in_features = self.args.encode_rgbd_size,
                out_features = self.args.pvrnn_mtrnn_size),
            nn.PReLU())
        self.action_in = Action_IN(self.args)
        #self.obs_out = Obs_OUT(self.args)
        self.rgbd_out = RGBD_OUT(self.args)
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd, comm, other, action):
        #obs = self.obs_in(rgbd, comm, other)
        obs = self.rgbd_in(rgbd)
        h = self.pretend_pvrnn(obs)
        action = self.action_in(action)
        h_w_action = torch.cat([h, action], dim = -1)
        #pred_rgbd, pred_comm, pred_other = self.obs_out(h_w_action)
        pred_rgbd = self.rgbd_out(h_w_action)
        return(h_w_action, pred_rgbd, None, None) #, pred_comm, pred_other)

gen = Generator(args = args)
gen.train()
opt = optim.Adam(params=gen.parameters(), lr=.01, weight_decay = .00001) 
gen_loser = WG()

disc = Discriminator(args = args)
disc.train()
disc_opt = optim.Adam(params=disc.parameters(), lr=.01, weight_decay = .00001) 
disc_loser = WD()

physicsClient = get_physics(GUI = True)
arena_1 = Arena(physicsClient)
arena_2 = None
task_runner = Task_Runner(Task(actions = 1, objects = 2, shapes = 5, colors = 6), arena_1, arena_2)



def example_images(e, reals, action, nexts, gens):
    fig, axs = plt.subplots(3, 1, figsize=(5, 6))

    fig.suptitle("EPOCH {}".format(e))
    fig.patch.set_facecolor('white')
    
    def one_part(this, title, image, comm, other):
        this.set_title("{} : COMM ({}), OTHER ({})".format(title, None, None)) # comm, [round(o,2) for o in other.tolist()]))
        this.imshow(image)
        this.set_xticks([])
        this.set_yticks([])
        
    one_part(axs[0], "BEFORE", reals[0], reals[1], reals[2])
    one_part(axs[1], "ACTION: {}\nAFTER".format(action), nexts[0], nexts[1], nexts[2])
    one_part(axs[2], "GENERATED", gens[0], gens[1], gens[2])
        
    plt.show()
    plt.close()
    


def plot_losses(disc_losses, gen_losses):
    plt.title("Losses")
    #plt.ylim(-1.1, .1)
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.plot(gen_losses, label='Generator Loss')
    plt.legend()  
    plt.show()
    plt.close()
    
    
    
def accuracy_images(real_accs, fake_accs, gen_accs):
    plt.title("Discriminator Accuracies")
    plt.ylim(-5, 105)
    plt.plot(real_accs, label='Real')
    plt.plot(fake_accs, label='Fake')
    plt.plot(gen_accs, label='Generated')
    plt.legend()
    plt.show()
    plt.close()
    
    

def make_batch(batch_size = args.batch_size):
    real_images = []
    real_comms = []
    real_others = []
    actions = []
    next_images = []
    next_comms = []
    next_others = []
    while(True):
        task_runner.begin(goal_action = "WATCH")
        rgbd, comm, other = task_runner.obs()
        real_images.append(rgbd)
        real_comms.append(comm)
        real_others.append(other)
        recommendation = task_runner.get_recommended_action(verbose = False)#True)
        actions.append(recommendation)
        reward, done, win = task_runner.step(recommendation, verbose = False)#True)
        rgbd, comm, other = task_runner.obs()
        next_images.append(rgbd)
        next_comms.append(comm)
        next_others.append(other)
        task_runner.done()
        if(len(real_images) == batch_size):
            break
    real_images = torch.cat(real_images).unsqueeze(1)
    real_comms = torch.cat(real_comms).unsqueeze(1)
    real_others = torch.cat(real_others).unsqueeze(1)
    actions = torch.stack(actions).unsqueeze(1)
    next_images = torch.cat(next_images).unsqueeze(1)
    next_comms = torch.cat(next_comms).unsqueeze(1)
    next_others = torch.cat(next_others).unsqueeze(1)
    return(real_images, real_comms, real_others, actions, next_images, next_comms, next_others)



def epoch(e):
    real_images, real_comms, real_others, actions, next_images, next_comms, next_others = make_batch()
    h_w_action, gen_images, gen_comms, gen_others = gen(real_images, real_comms, real_others, actions)
    
    real_judgement = disc(h_w_action.detach(), real_images, real_comms, real_others)
    real_judgement_binary = [round(g[0]) for g in real_judgement.squeeze(-1).tolist()]
    real_acc = 100 * sum([1 if g == 1 else 0 for g in real_judgement_binary]) / len(real_judgement_binary)
    print("REAL:", real_judgement.mean().item())
    
    fake_judgement = disc(h_w_action.detach(), gen_images.detach(), gen_comms, gen_others)
    fake_judgement_binary = [round(g[0]) for g in fake_judgement.squeeze(-1).tolist()]
    fake_acc = 100 * sum([1 if g == 0 else 0 for g in fake_judgement_binary]) / len(fake_judgement_binary)
    print("FAKE:", fake_judgement.mean().item())
    
    disc_loss = F.binary_cross_entropy(real_judgement.squeeze(1).squeeze(1), torch.ones([real_judgement.shape[0]]))
    disc_loss += F.binary_cross_entropy(fake_judgement.squeeze(1).squeeze(1), torch.zeros([fake_judgement.shape[0]]))
    
    disc_opt.zero_grad()
    disc_loss.backward()
    disc_opt.step()
    
    h_w_action, gen_images, gen_comms, gen_others = gen(real_images, real_comms, real_others, actions)
    gen_judgement = disc(h_w_action, gen_images, gen_comms, gen_others)
    gen_judgement_binary = [round(g[0]) for g in gen_judgement.squeeze(-1).tolist()]
    gen_acc = 100 * sum([1 if g == 0 else 0 for g in gen_judgement_binary]) / len(gen_judgement_binary)
    
    gen_loss = F.binary_cross_entropy(gen_judgement.squeeze(1).squeeze(1), torch.ones([gen_judgement.shape[0]]))

    opt.zero_grad()
    gen_loss.backward()
    opt.step()
    
    real_images = real_images[0,0,:,:,0:3].detach()
    #real_comms = real_comms[0,0].detach()
    #real_others = real_others[0,0].detach()
    
    actions = actions[0].detach()
    
    next_images = next_images[0,0,:,:,0:3].detach()
    #next_comms = next_comms[0,0].detach()
    #next_others = next_others[0,0].detach()
    
    gen_images = gen_images[0,0,:,:,0:3].detach()
    #gen_comms = gen_comms[0,0].detach()
    #gen_others = gen_others[0,0].detach()
            
    return(
        disc_loss.detach(), 
        gen_loss.detach(), 
        (real_images, None, None), # onehots_to_string(real_comms), real_others), 
        actions, 
        (next_images, None, None), # onehots_to_string(next_comms), next_others), 
        (gen_images, None, None), # onehots_to_string(gen_comms), gen_others),
        real_acc, fake_acc, gen_acc)

    

disc_losses = []
gen_losses = []
real_accs = []
fake_accs = []
gen_accs = []
for e in range(1000):
    disc_loss, gen_loss, reals, actions, nexts, gens, real_acc, fake_acc, gen_acc = epoch(e)
    disc_losses.append(disc_loss)
    gen_losses.append(gen_loss)
    real_accs.append(real_acc)
    fake_accs.append(fake_acc)
    gen_accs.append(gen_acc)
    if(e % 10 == 0):
        example_images(e, reals, actions, nexts, gens)
        plot_losses(disc_losses, gen_losses)
        accuracy_images(real_accs, fake_accs, gen_accs)


## %%
