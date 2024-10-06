#%% 

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils import default_args, detach_list, attach_list, print, duration, how_many_nans
from utils_submodule import init_weights, episodes_steps, var, sample, model_start, model_end
from mtrnn import MTRNN
from submodules import Obs_IN, Action_IN, Comm_IN, Comm_OUT



if __name__ == "__main__":
    
    args = default_args
    episodes = args.batch_size ; steps = args.max_steps
        
    

class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.action_in = Action_IN(self.args)

        self.lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size, 
                out_features = args.hidden_size),
            nn.PReLU(),
            nn.Linear(
                in_features = args.hidden_size, 
                out_features = args.hidden_size),
            nn.PReLU())
        
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
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, forward_hidden, action_hidden, parenting = True):
        
        start, episodes, steps, [forward_hidden] = model_start(
            [(forward_hidden, "lin")], device = self.args.device, half = self.args.half)
        
        how_many_nans(forward_hidden, "Actor, forward_hidden")
        x = self.lin(forward_hidden)
        
        mu, std = var(x, self.mu, self.std, self.args)
        
        sampled = sample(mu, std, self.args.device)
        if(self.args.half):
            sampled = sampled.to(dtype=torch.float16)
        action = torch.tanh(sampled)
        log_prob = Normal(mu, std).log_prob(sampled) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        
        [forward_hidden, action, log_prob] = model_end(start, episodes, steps, 
            [(forward_hidden, "lin"), (action, "lin"), (log_prob, "lin")], "ACTOR" if self.args.show_duration else None)
        encoded_action = self.action_in(action)
        concatenated = torch.cat([forward_hidden, encoded_action], dim = -1)
        comm_out, comm_log_prob = self.comm_out(concatenated)

        if(parenting):
            comm_out = torch.zeros_like(comm_out)
            comm_log_prob = torch.zeros_like(comm_log_prob)
            if(self.args.half):
                comm_out = comm_out.to(dtype=torch.float16)
                comm_log_prob = comm_log_prob.to(dtype=torch.float16)
        
        return action, comm_out, log_prob, comm_log_prob
    
    
    
if __name__ == "__main__":
    
    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(actor,
                                (
                                (episodes, steps, args.pvrnn_mtrnn_size),
                                (episodes, steps, args.hidden_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%

    
    
class Critic(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.action_in = Action_IN(self.args)
        self.comm_in = Comm_IN(self.args)
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + self.args.encode_action_size + self.args.encode_comm_size,
                out_features = self.args.hidden_size),
            nn.PReLU())
        
        self.value = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size,
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Linear(                
                in_features = self.args.hidden_size,
                out_features = 1))
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, action, comm_out, forward_hidden):        
        
        start, episodes, steps, [action, comm_out, forward_hidden] = model_start(
            [(action, "lin"), (comm_out, "comm"), (forward_hidden, "lin")], device = self.args.device, half = self.args.half)
                
        action = self.action_in(action)
        comm_out = self.comm_in(comm_out)
        x = torch.cat([forward_hidden, action.squeeze(1), comm_out.squeeze(1)], dim=-1)
        x = self.lin(x)
        value = self.value(x)
                
        [value] = model_end(start, episodes, steps, 
            [(value, "lin")], "CRITIC" if self.args.show_duration else None)
        
        return(value)
    


if __name__ == "__main__":
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(critic, 
                                (
                                (episodes, steps, args.action_shape),
                                (episodes, steps, args.max_comm_len, args.comm_shape),
                                (episodes, steps, args.pvrnn_mtrnn_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

# %%