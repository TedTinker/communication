#%% 

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils import default_args, print, duration, Action
from utils_submodule import init_weights, episodes_steps, var, sample, model_start, model_end
from mtrnn import MTRNN
from submodules import Wheels_Joints_IN, Voice_IN, Voice_OUT



if __name__ == "__main__":
    
    args = default_args
    episodes = args.batch_size ; steps = args.max_steps
        
    

class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.wheels_joints_in = Wheels_Joints_IN(self.args)

        self.lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size, 
                out_features = args.hidden_size),
            nn.PReLU(),
            nn.Linear(
                in_features = args.hidden_size, 
                out_features = args.hidden_size),
            nn.PReLU())
        
        self.voice_out = Voice_OUT(actor = True, args = self.args)
                
        self.mu = nn.Sequential(
            nn.Linear(
                in_features = args.hidden_size, 
                out_features = self.args.wheels_joints_shape))
        self.std = nn.Sequential(
            nn.Linear(
                in_features = args.hidden_size, 
                out_features = self.args.wheels_joints_shape),
            nn.Softplus())

        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, forward_hidden, parenting = True):
        
        start_time, episodes, steps, [forward_hidden] = model_start(
            [(forward_hidden, "lin")], device = self.args.device, half = self.args.half)
        
        x = self.lin(forward_hidden)
        
        mu, std = var(x, self.mu, self.std, self.args)
        
        sampled = sample(mu, std, self.args.device)
        if(self.args.half):
            sampled = sampled.to(dtype=torch.float16)
        wheels_joints = torch.tanh(sampled)
        log_prob = Normal(mu, std).log_prob(sampled) - torch.log(1 - wheels_joints.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        
        [forward_hidden, wheels_joints, log_prob] = model_end(start_time, episodes, steps, 
            [(forward_hidden, "lin"), (wheels_joints, "lin"), (log_prob, "lin")], "\tACTOR" if self.args.show_duration else None)
        encoded_wheels_joints = self.wheels_joints_in(wheels_joints)
        concatenated = torch.cat([forward_hidden, encoded_wheels_joints], dim = -1)
        voice_out, voice_log_prob = self.voice_out(concatenated)

        if(parenting):
            voice_out = torch.zeros_like(voice_out)
            voice_log_prob = torch.zeros_like(voice_log_prob)
            if(self.args.half):
                voice_out = voice_out.to(dtype=torch.float16)
                voice_log_prob = voice_log_prob.to(dtype=torch.float16)
                        
        return Action(wheels_joints, voice_out), log_prob, voice_log_prob
    
    
    
if __name__ == "__main__":
    
    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(actor,
                                (
                                (episodes, steps, args.pvrnn_mtrnn_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
#%%

    
    
class Critic(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.wheels_joints_in = Wheels_Joints_IN(self.args)
        self.voice_out_in = Voice_IN(self.args)
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.h_w_wheels_joints_size + self.args.voice_encode_size,
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
        
    def forward(self, action, forward_hidden):        
        
        start_time, episodes, steps, [wheels_joints, voice_out, forward_hidden] = model_start(
            [(action.wheels_joints, "lin"), (action.voice_out, "voice"), (forward_hidden, "lin")], device = self.args.device, half = self.args.half)
                
        wheels_joints = self.wheels_joints_in(wheels_joints)
        voice_out = self.voice_out_in(voice_out)
        x = torch.cat([forward_hidden, wheels_joints.squeeze(1), voice_out.squeeze(1)], dim=-1)
        x = self.lin(x)
        value = self.value(x)
                
        [value] = model_end(start_time, episodes, steps, 
            [(value, "lin")], "\tCRITIC")
        
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
                                (episodes, steps, args.wheels_joints_shape),
                                (episodes, steps, args.max_voice_len, args.voice_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

# %%