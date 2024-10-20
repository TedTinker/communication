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

from utils import print, default_args, how_many_nans
from utils_submodule import model_start, model_end, generate_2d_sinusoidal_positions, generate_2d_positional_layers, generate_1d_positional_layers, \
    Ted_Conv1d, Ted_Conv2d, ConstrainedConv1d, ConstrainedConv2d,\
    init_weights, hsv_to_circular_hue, pad_zeros, var, sample
from mtrnn import MTRNN

if __name__ == "__main__":
    
    args = default_args
    episodes = args.batch_size ; steps = args.max_steps
    
    

class RGBD_IN(nn.Module):

    def __init__(self, args = default_args):
        super(RGBD_IN, self).__init__()  
        
        self.args = args 
        
        image_dims = 4
        
        rgbd_size = (1, image_dims, self.args.image_size, self.args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.a = nn.Sequential(nn.BatchNorm2d(image_dims))
        
        example = self.a(example)
        rgbd_latent_size = example.flatten(1).shape[1]
                
        self.b = nn.Sequential(
            nn.Linear(
                in_features = rgbd_latent_size, 
                out_features = self.args.rgbd_encode_size),
            nn.BatchNorm1d(self.args.rgbd_encode_size),
            nn.PReLU(),
            nn.Dropout(self.args.dropout))
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, rgbd):
        start, episodes, steps, [rgbd] = model_start([(rgbd, "cnn")], self.args.device, self.args.half)
        
        how_many_nans(rgbd, "RGBD IN, rgbd start")
                
        rgbd = (rgbd * 2) - 1
        a = self.a(rgbd).flatten(1)
        
        how_many_nans(a, "RGBD IN, after a")
        
        encoding = self.b(a)
        
        how_many_nans(encoding, "RGBD IN, after b")
        
        [encoding] = model_end(start, episodes, steps, [(encoding, "lin")], "RGBD_IN" if self.args.show_duration else None)
        return(encoding)
    
    
    
if __name__ == "__main__":
    
    rgbd_in = RGBD_IN(args = args) #.half()
    
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
        self.out_features_channels = self.args.hidden_size
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.h_w_action_size,
                out_features = self.out_features_channels * (self.args.image_size//self.args.divisions) * (self.args.image_size//self.args.divisions)))
        
        self.b = nn.Sequential(
            nn.BatchNorm2d(self.out_features_channels),
            nn.PReLU(),
            nn.Dropout(self.args.dropout),
            
            nn.Conv2d(
                in_channels = self.out_features_channels, 
                out_channels = 4 * (1 if self.args.divisions == 1 else 2 ** self.args.divisions),
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.PixelShuffle(self.args.divisions),
            nn.Tanh())
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, h_w_action):
        start, episodes, steps, [h_w_action] = model_start([(h_w_action, "lin")], self.args.device, self.args.half)
        
        a = self.a(h_w_action)
        a = a.reshape(episodes * steps, self.out_features_channels, self.args.image_size//self.args.divisions, self.args.image_size//self.args.divisions)
        
        rgbd = self.b(a)
        rgbd = (rgbd + 1) / 2
                
        [rgbd] = model_end(start, episodes, steps, [(rgbd, "cnn")], "RGBD_OUT" if self.args.show_duration else None)        
        return(rgbd)
    
    
    
if __name__ == "__main__":
    
    rgbd_out = RGBD_OUT(args = args)
    
    print("\n\n")
    print(rgbd_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(rgbd_out, 
                                (episodes, steps, args.pvrnn_mtrnn_size + args.wheels_shoulders_encode_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%



class Comm_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_IN, self).__init__()  
        
        self.args = args
        
        self.a = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.comm_shape,
                embedding_dim = self.args.char_encode_size),
            nn.PReLU(),
            nn.Dropout(self.args.dropout))
        
        self.b = nn.Sequential(
            #nn.BatchNorm1d(self.args.har_encode_size),
            nn.Conv1d(
                in_channels = self.args.char_encode_size, 
                out_channels = self.args.hidden_size, 
                kernel_size = self.args.max_comm_len),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(self.args.dropout))
        
        self.c = nn.GRU(
            input_size = self.args.hidden_size,
            hidden_size = self.args.hidden_size,
            batch_first = True)
                
        self.d = nn.Sequential(
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.comm_encode_size))
                
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, comm):
        start, episodes, steps, [comm] = model_start([(comm, "comm")], self.args.device, self.args.half)
                
        comm = pad_zeros(comm, self.args.max_comm_len)
        comm = torch.argmax(comm, dim = -1).int()
                
        a = self.a(comm)
        b = self.b(a.permute((0, 2, 1))).permute((0, 2, 1))
        c, _ = self.c(b)    
        c = c.reshape(episodes, steps, self.args.hidden_size)
        encoding = self.d(c)
        
        [encoding] = model_end(start, episodes, steps, [(encoding, "lin")], "COMM_IN" if self.args.show_duration else None)
        return(encoding)

    
    
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
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.h_w_action_size, 
                out_features = self.args.hidden_size * self.args.max_comm_len),
            nn.BatchNorm1d(self.args.hidden_size * self.args.max_comm_len),
            nn.PReLU(),
            nn.Dropout(self.args.dropout))
            
        self.b = nn.GRU(
            input_size = self.args.hidden_size + self.args.max_comm_len,
            hidden_size = self.args.hidden_size,
            batch_first = True)
        
        self.mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.comm_shape))
        
        self.std = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.comm_shape),
            nn.Softplus())
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
                
    def forward(self, h_w_action):
        start, episodes, steps, [h_w_action] = model_start([(h_w_action, "lin")], self.args.device, self.args.half)
                
        h_w_action = h_w_action.reshape(episodes * steps, self.args.pvrnn_mtrnn_size + self.args.wheels_shoulders_encode_size)
        a = self.a(h_w_action)
        a = a.reshape(episodes * steps, self.args.max_comm_len, self.args.hidden_size)
        positional_layers = generate_1d_positional_layers(episodes * steps, self.args.max_comm_len, self.args.device)
        if(self.args.half):
            positional_layers = positional_layers.to(dtype=torch.float16)  
        a = torch.cat([a, positional_layers.squeeze(1)], dim = -1)
        b, _ = self.b(a)
        
        if(self.actor):
            mu, std = var(b, self.mu, self.std, self.args)
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
            comm_pred = self.mu(b)
            # Maybe softplus this?
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
                                (episodes, steps, args.pvrnn_mtrnn_size + args.wheels_shoulders_encode_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

#%%


    
class Sensors_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Sensors_IN, self).__init__()
        
        self.args = args 
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.sensors_shape,
                out_features = self.args.sensors_encode_size),
            #nn.BatchNorm1d(self.args.sensors_encode_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, sensors):
        start, episodes, steps, [sensors] = model_start([(sensors, "lin")], self.args.device, self.args.half)
        encoding = self.a(sensors)
        [encoding] = model_end(start, episodes, steps, [(encoding, "lin")], "SENSORS_IN" if self.args.show_duration else None)
        return(encoding)

    
    
if __name__ == "__main__":
    
    sensors_in = Sensors_IN(args = args)
    
    print("\n\n")
    print(sensors_in)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(sensors_in, 
                                (episodes, steps, args.sensors_shape)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

#%%



class Sensors_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Sensors_OUT, self).__init__()  
        
        self.args = args 
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.h_w_action_size,
                out_features = self.args.sensors_shape),
            nn.Tanh())
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, h_w_action):
        start, episodes, steps, [h_w_action] = model_start([(h_w_action, "lin")], self.args.device, self.args.half)
        sensors = self.a(h_w_action)
        sensors = (sensors + 1) / 2
        [sensors] = model_end(start, episodes, steps, [(sensors, "lin")], "SENSORS_OUT" if self.args.show_duration else None)
        return(sensors)
    
    
    
if __name__ == "__main__":
    
    sensors_out = Sensors_OUT(args = args)
    
    print("\n\n")
    print(sensors_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(sensors_out, 
                                (episodes, steps, args.pvrnn_mtrnn_size + args.wheels_shoulders_encode_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
#%%
    
    
    
class Obs_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Obs_IN, self).__init__()  
                
        self.args = args
        self.rgbd_in = RGBD_IN(self.args)
        self.comm_in = Comm_IN(self.args)
        self.sensors_in = Sensors_IN(self.args)
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, rgbd, comm, sensors):
        rgbd = self.rgbd_in(rgbd)
        comm = self.comm_in(comm)
        sensors = self.sensors_in(sensors)
        return(torch.cat([rgbd, comm, sensors], dim = -1))
    
    
    
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
                                (episodes, steps, args.sensors_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%
    
    
    
class Obs_OUT(nn.Module):
    
    def __init__(self, args = default_args):
        super(Obs_OUT, self).__init__()  
        
        self.args = args 
        self.rgbd_out = RGBD_OUT(self.args)
        self.comm_out = Comm_OUT(actor = False, args = self.args)
        self.sensors_out = Sensors_OUT(self.args)
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, h_w_action):
        rgbd_pred = self.rgbd_out(h_w_action)
        comm_pred = self.comm_out(h_w_action)
        sensors_pred = self.sensors_out(h_w_action)
        return(rgbd_pred, comm_pred, sensors_pred)
    
    
    
if __name__ == "__main__":
    
    obs_out = Obs_OUT(args = args)
    
    print("\n\n")
    print(obs_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(obs_out, 
                                (episodes, steps, args.pvrnn_mtrnn_size + args.wheels_shoulders_encode_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%
    
    
    
class Wheels_Shoulders_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Wheels_Shoulders_IN, self).__init__()
        
        self.args = args 
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.wheels_shoulders_shape, 
                out_features = self.args.wheels_shoulders_encode_size),
            #nn.BatchNorm1d(self.args.wheels_shoulders_encode_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, wheels_shoulders):
        start, episodes, steps, [wheels_shoulders] = model_start([(wheels_shoulders, "lin")], self.args.device, self.args.half)
        encoded = self.a(wheels_shoulders)
        [encoded] = model_end(start, episodes, steps, [(encoded, "lin")], "WHEELS_SHOULDERS_IN" if self.args.show_duration else None)

        return(encoded)
    
    
    
if __name__ == "__main__":
    
    wheels_shoulders_in = Wheels_Shoulders_IN(args = args)
    
    print("\n\n")
    print(wheels_shoulders_in)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(wheels_shoulders_in, 
                                (episodes, steps, args.wheels_shoulders_shape)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%