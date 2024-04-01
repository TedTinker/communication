#%%
import numpy as np
from math import log

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
from kornia.color import rgb_to_hsv 
import torchgan.layers as gg
from torchinfo import summary as torch_summary

from utils import print, default_args, init_weights, \
    episodes_steps, pad_zeros, Ted_Conv1d, Ted_Conv2d, var, \
    sample, duration, ConstrainedConv1d, ConstrainedConv2d, ResidualBlock, DenseBlock, \
    TransformerModel, ImageTransformer
from mtrnn import MTRNN

d = .01

if __name__ == "__main__":
    
    args = default_args
    episodes = args.batch_size ; steps = args.max_steps



class RGBD_IN(nn.Module):

    def __init__(self, args = default_args):
        super(RGBD_IN, self).__init__()  
        
        self.args = args 
        
        rgbd_size = (1, 4, self.args.image_size, self.args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.rgbd_in = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(
                in_channels = 4, 
                out_channels = self.args.hidden_size, 
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(d),
            
            nn.Conv2d(
                in_channels = self.args.hidden_size, 
                out_channels = self.args.hidden_size, 
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(d))
        
        example = self.rgbd_in(example)
        rgbd_latent_size = example.flatten(1).shape[1]
        
        self.rgbd_in_lin = nn.Sequential(
            nn.Linear(
                in_features = rgbd_latent_size, 
                out_features = self.args.encode_rgbd_size),
            nn.BatchNorm1d(self.args.encode_rgbd_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, rgbd):
        start = duration()
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        episodes, steps = episodes_steps(rgbd)
        rgbd = rgbd.reshape(episodes * steps, rgbd.shape[2], rgbd.shape[3], rgbd.shape[4]).permute(0, -1, 1, 2)
        rgbd = (rgbd * 2) - 1
        rgbd = self.rgbd_in(rgbd).flatten(1)
        rgbd = self.rgbd_in_lin(rgbd)
        rgbd = rgbd.reshape(episodes, steps, rgbd.shape[1])
        #print("RGBD_IN:", duration() - start)
        return(rgbd)
    
    
    
if __name__ == "__main__":
    
    rgbd_in = RGBD_IN(args = args)
    
    print("\n\n")
    print(rgbd_in)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(rgbd_in, 
                                (episodes, steps, args.image_size, args.image_size, 4)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
#%%
    
    

class RGBD_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(RGBD_OUT, self).__init__()  
        
        self.args = args 
        
        self.rgbd_out_lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + self.args.encode_action_size,
                out_features = self.args.hidden_size * (self.args.image_size) * (self.args.image_size)),
            nn.BatchNorm1d(self.args.hidden_size * (self.args.image_size) * (self.args.image_size)),
            nn.PReLU(),
            nn.Dropout(d))
        
        self.rgbd_out = nn.Sequential(
            nn.Conv2d(
                in_channels = self.args.hidden_size,
                out_channels = self.args.hidden_size,
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(d),
            nn.Conv2d(
                in_channels = self.args.hidden_size,
                out_channels = self.args.hidden_size,
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(d),
            
            nn.Conv2d(
                in_channels = self.args.hidden_size,
                out_channels = 4,
                kernel_size = 1),
            nn.Sigmoid())
        
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, h_w_action):
        start = duration()
        if(len(h_w_action.shape) == 2): h_w_action = h_w_action.unsqueeze(1)
        episodes, steps = episodes_steps(h_w_action)
        h_w_action = h_w_action.reshape(episodes * steps, h_w_action.shape[2])
        h_w_action = self.rgbd_out_lin(h_w_action)
        rgbd = h_w_action.reshape(episodes * steps, self.args.hidden_size, self.args.image_size, self.args.image_size)
        rgbd = self.rgbd_out(rgbd)
        rgbd = rgbd.permute(0, 2, 3, 1)
        rgbd = rgbd.reshape(episodes, steps, rgbd.shape[1], rgbd.shape[2], rgbd.shape[3])
        #print("RGBD_OUT:", duration() - start)
        return(rgbd)
    
    
    
if __name__ == "__main__":
    
    rgbd_out = RGBD_OUT(args = args)
    
    print("\n\n")
    print(rgbd_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(rgbd_out, 
                                (episodes, steps, args.pvrnn_mtrnn_size + args.encode_action_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%



class Comm_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_IN, self).__init__()  
        
        self.args = args
        
        self.comm_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.comm_shape,
                embedding_dim = self.args.encode_char_size),
            nn.PReLU(),
            nn.Dropout(d))
        
        self.comm_cnn = nn.Sequential(
            nn.BatchNorm1d(self.args.encode_char_size),
            nn.Conv1d(
                in_channels = self.args.encode_char_size, 
                out_channels = self.args.hidden_size, 
                kernel_size = 1),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(d))
        
        self.comm_rnn = nn.GRU(
            input_size = self.args.hidden_size,
            hidden_size = self.args.hidden_size,
            batch_first = True)
        
        self.batchnorm_1 = nn.BatchNorm1d(self.args.hidden_size)
        
        self.comm_lin = nn.Sequential(
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.encode_comm_size))
        
        self.batchnorm_2 = nn.Sequential(
            nn.BatchNorm1d(self.args.encode_comm_size),
            nn.PReLU())
                
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, comm):
        start = duration()
        if(len(comm.shape) == 2):  comm = comm.unsqueeze(0)
        if(len(comm.shape) == 3):  comm = comm.unsqueeze(1)
        episodes, steps = episodes_steps(comm)
        comm = comm.reshape(episodes * steps, comm.shape[2], comm.shape[3])
        comm = pad_zeros(comm, self.args.max_comm_len)
        comm = torch.argmax(comm, dim = -1).int()
        comm = self.comm_embedding(comm)
        comm = self.comm_cnn(comm.permute((0, 2, 1))).permute((0, 2, 1))
        comm, _ = self.comm_rnn(comm)
        comm = self.batchnorm_1(comm.permute((0, 2, 1))).permute((0, 2, 1))      
        comm = comm.reshape((episodes, steps, self.args.max_comm_len, self.args.hidden_size))
        comm = self.comm_lin(comm)
        comm = comm[:,:,-1]
        comm = self.batchnorm_2(comm.permute((0, 2, 1))).permute((0, 2, 1))
        #print("COMM_IN:", duration() - start)
        return(comm)

    
    
if __name__ == "__main__":
    
    comm_in = Comm_IN(args = args)
    
    print("\n\n")
    print(comm_in)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(comm_in, 
                                (episodes, steps, args.max_comm_len, args.comm_shape)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%



class Comm_OUT(nn.Module):

    def __init__(self, actor = False, args = default_args):
        super(Comm_OUT, self).__init__()  
                
        self.args = args
        self.actor = actor
        
        self.comm_lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + self.args.encode_action_size, 
                out_features = self.args.hidden_size))
        
        self.batchnorm = nn.Sequential(
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(d))
            
        self.comm_rnn = nn.GRU(
            input_size = self.args.hidden_size,
            hidden_size = self.args.hidden_size,
            batch_first = True)
        
        self.comm_cnn = nn.Sequential(
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(d),
            nn.Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = self.args.hidden_size,
                kernel_size = 1),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU())
        
        self.comm_out_mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.comm_shape))
        
        self.comm_out_std = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.comm_shape),
            nn.Softplus())
        
        self.apply(init_weights)
        self.to(self.args.device)
                
    def forward(self, h_w_action):
        start = duration()
        if(len(h_w_action.shape) == 2):   h_w_action = h_w_action.unsqueeze(1)
        #[h_w_action] = attach_list([h_w_action], self.args.device)
        episodes, steps = episodes_steps(h_w_action)
        h_w_action = h_w_action.reshape(episodes * steps, 1, self.args.pvrnn_mtrnn_size + self.args.encode_action_size)
        h_w_action = self.comm_lin(h_w_action)
        h_w_action = self.batchnorm(h_w_action.permute((0, 2, 1))).permute((0, 2, 1))
        comm_h = None
        comm_hs = []
        for i in range(self.args.max_comm_len):
            comm_h, _ = self.comm_rnn(h_w_action, comm_h if comm_h == None else comm_h.permute(1, 0, 2))
            comm_hs.append(comm_h)
        comm_h = torch.cat(comm_hs, dim = -2)
        comm_h = self.comm_cnn(comm_h.permute((0, 2, 1))).permute((0, 2, 1))
        if(self.actor):
            mu, std = var(comm_h, self.comm_out_mu, self.comm_out_std, self.args)
            comm = sample(mu, std, self.args.device)
            comm_out = torch.tanh(comm)
            log_prob = Normal(mu, std).log_prob(comm) - torch.log(1 - comm_out.pow(2) + 1e-6)
            log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
            log_prob = log_prob.mean(-2)
            comm_out = comm_out.reshape(episodes, steps, self.args.max_comm_len, self.args.comm_shape)
            log_prob = log_prob.reshape(episodes, steps, 1)
            #print("COMM_OUT:", duration() - start)
            return(comm_out, log_prob)
        else:
            comm_pred = self.comm_out_mu(comm_h)
            comm_pred = comm_pred.reshape(episodes, steps, self.args.max_comm_len, self.args.comm_shape)
            #print("COMM_OUT:", duration() - start)
            return(comm_pred)
    
    
    
if __name__ == "__main__":
    
    comm_out = Comm_OUT(actor = False, args = args)
    
    print("\n\n")
    print(comm_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(comm_out, 
                                (episodes, steps, args.pvrnn_mtrnn_size + args.encode_action_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    """comm_out = Comm_OUT(actor = True, args = args)
    
    print("\n\n")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(comm_out, 
                                (episodes, steps, args.pvrnn_mtrnn_size + args.encode_action_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))"""
    
#%%


    
class Other_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Other_IN, self).__init__()
        
        self.args = args 
        
        self.other_in = nn.Sequential(
            nn.Linear(
                in_features = self.args.other_shape,
                out_features = self.args.encode_other_size),
            nn.BatchNorm1d(self.args.encode_other_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, other):
        #[other] = attach_list([other], self.args.device)
        if(len(other.shape) == 2):   other = other.unsqueeze(1)
        episodes, steps = episodes_steps(other)
        other = other.reshape(episodes * steps, other.shape[2])
        other = self.other_in(other)
        other = other.reshape(episodes, steps, other.shape[1])
        return(other)

    
    
if __name__ == "__main__":
    
    other_in = Other_IN(args = args)
    
    print("\n\n")
    print(other_in)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(other_in, 
                                (episodes, steps, args.other_shape)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

#%%



class Other_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Other_OUT, self).__init__()  
        
        self.args = args 
        
        self.other_out_lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + self.args.encode_action_size,
                out_features = self.args.other_shape),
            nn.Sigmoid())
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, h_w_action):
        start = duration()
        if(len(h_w_action.shape) == 2): h_w_action = h_w_action.unsqueeze(1)
        episodes, steps = episodes_steps(h_w_action)
        h_w_action = h_w_action.reshape(episodes * steps, self.args.pvrnn_mtrnn_size + self.args.encode_action_size)
        other = self.other_out_lin(h_w_action)
        other = other.reshape(episodes, steps, other.shape[1])
        #print("RGBD_OUT:", duration() - start)
        return(other)
    
    
    
if __name__ == "__main__":
    
    other_out = Other_OUT(args = args)
    
    print("\n\n")
    print(other_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(other_out, 
                                (episodes, steps, args.pvrnn_mtrnn_size + args.encode_action_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
#%%
    
    
    
class Obs_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Obs_IN, self).__init__()  
                
        self.args = args
        self.rgbd_in = RGBD_IN(self.args)
        self.comm_in = Comm_IN(self.args)
        self.other_in = Other_IN(self.args)
        
    def forward(self, rgbd, comm, other):
        rgbd = self.rgbd_in(rgbd)
        comm = self.comm_in(comm)
        other = self.other_in(other)
        return(torch.cat([rgbd, comm, other], dim = -1))
    
    
    
if __name__ == "__main__":
    
    obs_in = Obs_IN(args = args)
    
    print("\n\n")
    print(obs_in)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(obs_in, 
                                ((episodes, steps, args.image_size, args.image_size, 4),
                                (episodes, steps, args.max_comm_len, args.comm_shape),
                                (episodes, steps, args.other_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%
    
    
    
class Obs_OUT(nn.Module):
    
    def __init__(self, args = default_args):
        super(Obs_OUT, self).__init__()  
        
        self.args = args 
        self.rgbd_out = RGBD_OUT(self.args)
        self.comm_out = Comm_OUT(actor = False, args = self.args)
        self.other_out = Other_OUT(self.args)
        
    def forward(self, h_w_action):
        rgbd_pred = self.rgbd_out(h_w_action)
        comm_pred = self.comm_out(h_w_action)
        other_pred = self.other_out(h_w_action)
        return(rgbd_pred, comm_pred, other_pred)
    
    
    
if __name__ == "__main__":
    
    obs_out = Obs_OUT(args = args)
    
    print("\n\n")
    print(obs_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(obs_out, 
                                (episodes, steps, args.pvrnn_mtrnn_size + args.encode_action_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%
    
    
    
class Action_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Action_IN, self).__init__()
        
        self.args = args 
        
        self.action_in = nn.Sequential(
            nn.Linear(
                in_features = self.args.action_shape, 
                out_features = self.args.encode_action_size),
            nn.BatchNorm1d(self.args.encode_action_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, action):
        #[action] = attach_list([action], self.args.device)
        if(len(action.shape) == 2):   action = action.unsqueeze(1)
        episodes, steps = episodes_steps(action)
        action = action.reshape(episodes * steps, action.shape[2])
        action = self.action_in(action)
        action = action.reshape(episodes, steps, action.shape[1])
        return(action)
    
    
    
if __name__ == "__main__":
    
    action_in = Action_IN(args = args)
    
    print("\n\n")
    print(action_in)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(action_in, 
                                (episodes, steps, args.action_shape)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%



class Discriminator(nn.Module):
    
    def __init__(self, args = default_args):
        super(Discriminator, self).__init__()  
        
        self.args = args
        #self.obs_in = Obs_IN(self.args)
        self.rgbd_in = RGBD_IN(self.args)
        
        self.stat_quantity = self.get_stats(
            torch.zeros((1, 1, self.args.image_size, self.args.image_size, 4))).shape[-1]
                
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + self.args.encode_action_size + self.args.encode_rgbd_size + self.stat_quantity, # + self.args.encode_obs_size + self.stat_quantity, 
                out_features = self.args.hidden_size),
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = 1),
            nn.Sigmoid())
        
        self.apply(init_weights)
        self.to(self.args.device)
        
    def get_stats(self, images):
        # Statistics for each image
        c_mean   = images.mean((2, 3))
        #c_q     = torch.quantile(images.flatten(3), q = torch.tensor([.01, .25, .5, .75, .99]), dim = 2)#.permute(1, 2, 0).flatten(1)
        c_var    = torch.var(images, dim = (2, 3)) 
        
        # Statistics for entire batch
        b_mean   = c_mean.mean((0, 1))
        b_mean   = torch.tile(b_mean, (images.shape[0], images.shape[1], 1))
        #b_q     = torch.tile(torch.quantile(images.permute(1, 0, 2, 3).flatten(1), q = torch.tensor([.01, .25,  .5, .75, .99]), dim = 1).flatten(0).unsqueeze(0), (images.shape[0], 1))
        b_var    = torch.var(images, dim = (0, 1, 2, 3))
        b_var    = torch.tile(b_var, (images.shape[0], images.shape[1], 1))
        
        stats = torch.cat([c_mean, c_var, b_mean, b_var], dim = -1).to(self.args.device)
        return(stats)
            
    def forward(self, h_w_action, rgbd, comm, other):
        episodes, steps = episodes_steps(h_w_action)
        stats = self.get_stats(rgbd)
        #obs = self.obs_in(rgbd, comm, other)
        obs = self.rgbd_in(rgbd)
        h_w_action = torch.cat([h_w_action, obs, stats], dim = -1)
        judgement = self.lin(h_w_action)
        return(judgement)
    
    
    
if __name__ == "__main__":
    
    discriminator = Discriminator(args = args)
        
    print("\n\n")
    print(discriminator)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(discriminator, 
                                ((episodes, steps, args.pvrnn_mtrnn_size + args.encode_action_size),
                                (episodes, steps, args.image_size, args.image_size, 4),
                                (episodes, steps, args.max_comm_len, args.comm_shape),
                                (episodes, steps, args.other_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
# %%
