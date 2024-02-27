#%% 

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils import default_args, detach_list, attach_list, print, init_weights, episodes_steps, var, sample, create_comm_mask, duration
from mtrnn import MTRNN
from submodules import Obs_IN, Action_IN, Comm_IN, Comm_OUT



if __name__ == "__main__":
    
    args = default_args
    episodes = args.batch_size ; steps = args.max_steps
        
    

class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        #self.obs_in = Obs_IN(args)
        #self.action_in = Action_IN(self.args)
        #self.comm_in = Comm_IN(self.args)

        self.lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size, # + 4 * args.hidden_size, 
                out_features = args.hidden_size),
            nn.PReLU())
        
        #self.mtrnn = MTRNN(
        #        input_size = self.args.hidden_size,
        #        hidden_size = self.args.hidden_size, 
        #        time_constant = 1,
        #        args = self.args)
        
        self.comm_out = Comm_OUT(actor = True, args = self.args)
        
        self.mu = nn.Sequential(
            nn.Linear(
                in_features = args.hidden_size, 
                out_features = self.args.action_shape))
        self.std = nn.Sequential(
            nn.Linear(
                in_features = args.hidden_size, 
                out_features = self.args.action_shape),
            nn.Softplus())

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, comm_in, prev_action, prev_comm_out, forward_hidden, action_hidden, parented = True):
        if(len(forward_hidden.shape) == 2): forward_hidden = forward_hidden.unsqueeze(1)
        #if(len(comm_in.shape) == 2):  comm_in = comm_in.unsqueeze(0)
        #if(len(comm_in.shape) == 3):  comm_in = comm_in.unsqueeze(1)
        #if(len(prev_comm_out.shape) == 2):  prev_comm_out = prev_comm_out.unsqueeze(0)
        #if(len(prev_comm_out.shape) == 3):  prev_comm_out = prev_comm_out.unsqueeze(1)
        
        #mask, last_indexes = create_comm_mask(comm_in)
        #comm_in *= mask.unsqueeze(-1).tile((1,1,1,self.args.comm_shape))
        #mask, last_indexes = create_comm_mask(prev_comm_out)
        #prev_comm_out *= mask.unsqueeze(-1).tile((1,1,1,self.args.comm_shape))
        
        #obs = self.obs_in(rgbd, comm_in)
        #prev_action = self.action_in(prev_action)
        #prev_comm_out_encoded = self.comm_in(prev_comm_out)
        # x = torch.cat([obs, prev_action, prev_comm_out_encoded, forward_hidden], dim = -1)
        x = self.lin(forward_hidden)
        #x = self.mtrnn(x, action_hidden)
        #action_hidden = action_hidden[:,-1].unsqueeze(1)
        
        mu, std = var(x, self.mu, self.std, self.args)
        
        sampled = sample(mu, std, self.args.device)
        action = torch.tanh(sampled)
        log_prob = Normal(mu, std).log_prob(sampled) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        
        if(parented):
            comm_out = torch.zeros_like(prev_comm_out)
            comm_log_prob = torch.zeros_like(log_prob)
        else:
            comm_out, comm_log_prob = self.comm_out(forward_hidden)
        
        return action, comm_out, log_prob, comm_log_prob, action_hidden
    
    
    
if __name__ == "__main__":
    
    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(actor,
                                ((episodes, steps, args.image_size, args.image_size * 4, 4), 
                                (episodes, steps, args.max_comm_len, args.comm_shape), 
                                (episodes, steps, args.action_shape),
                                (episodes, steps, args.max_comm_len, args.comm_shape),
                                (episodes, steps, args.pvrnn_mtrnn_size),
                                (episodes, steps, args.hidden_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%

    
    
class Critic(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        #self.obs_in = Obs_IN(self.args)
        self.action_in = Action_IN(self.args)
        #self.comm_in = Comm_IN(self.args)
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + self.args.encode_action_size,
                out_features = self.args.hidden_size),
            nn.PReLU())
        
        #self.mtrnn = MTRNN(
        #        input_size = self.args.hidden_size,
        #        hidden_size = self.args.hidden_size, 
        #        time_constant = 1,
        #        args = self.args)
        
        self.value = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size,
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Linear(                
                in_features = self.args.hidden_size,
                out_features = 1))
        
    def forward(self, rgbd, comm_in, action, comm_out, forward_hidden, critic_hidden):
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        if(len(forward_hidden.shape) == 2): forward_hidden = forward_hidden.unsqueeze(1)
        #if(len(comm_in.shape) == 2):  comm_in = comm_in.unsqueeze(0)
        #if(len(comm_in.shape) == 3):  comm_in = comm_in.unsqueeze(1)
        #if(len(comm_out.shape) == 2):  comm_out = comm_out.unsqueeze(0)
        #if(len(comm_out.shape) == 3):  comm_out = comm_out.unsqueeze(1)
        
        #mask, last_indexes = create_comm_mask(comm_in)
        #comm_in = comm_in * mask.unsqueeze(-1).tile((1,1,1,self.args.comm_shape))
        # Can't mask comm_out, or else actor loses backprop!
        
        #obs = self.obs_in(rgbd, comm_in)
        action = self.action_in(action)
        #comm_out = self.comm_in(comm_out)
        x = torch.cat([forward_hidden, action], dim=-1)
        x = self.lin(x)
        #value = self.mtrnn(x, critic_hidden)
        #critic_hidden = value[:,-1].unsqueeze(1)
        value = self.value(x)
        return(value, None)
    


if __name__ == "__main__":
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(critic, 
                                ((episodes, steps, args.image_size, args.image_size * 4, 4), 
                                (episodes, steps, args.max_comm_len, args.comm_shape), 
                                (episodes, steps, args.action_shape),
                                (episodes, steps, args.max_comm_len, args.comm_shape),
                                (episodes, steps, args.pvrnn_mtrnn_size),
                                (episodes, steps, args.hidden_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

# %%
