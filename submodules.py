#%%
import numpy as np
from math import log

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
import torchgan.layers as gg
from torchinfo import summary as torch_summary

from utils import print, default_args
from utils_submodule import model_start, model_end, init_weights, pad_zeros, var, sample
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
        start_time, episodes, steps, [rgbd] = model_start([(rgbd, "cnn")], self.args.device, self.args.half)
        rgbd = (rgbd * 2) - 1
        a = self.a(rgbd).flatten(1)
        encoding = self.b(a)
        [encoding] = model_end(start_time, episodes, steps, [(encoding, "lin")], "\tRGBD_IN")
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
                in_features = self.args.h_w_wheels_shoulders_size,
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
        
    def forward(self, h_w_wheels_shoulders):
        start_time, episodes, steps, [h_w_wheels_shoulders] = model_start([(h_w_wheels_shoulders, "lin")], self.args.device, self.args.half)
        
        a = self.a(h_w_wheels_shoulders)
        a = a.reshape(episodes * steps, self.out_features_channels, self.args.image_size//self.args.divisions, self.args.image_size//self.args.divisions)
        
        rgbd = self.b(a)
        rgbd = (rgbd + 1) / 2
                
        [rgbd] = model_end(start_time, episodes, steps, [(rgbd, "cnn")], "\tRGBD_OUT")        
        return(rgbd)
    
    
    
if __name__ == "__main__":
    
    rgbd_out = RGBD_OUT(args = args)
    
    print("\n\n")
    print(rgbd_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(rgbd_out, 
                                (episodes, steps, args.h_w_wheels_shoulders_size)))
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
        start_time, episodes, steps, [sensors] = model_start([(sensors, "lin")], self.args.device, self.args.half)
        encoding = self.a(sensors)
        [encoding] = model_end(start_time, episodes, steps, [(encoding, "lin")], "\tSENSORS_IN")
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
                in_features = self.args.h_w_wheels_shoulders_size,
                out_features = self.args.sensors_shape),
            nn.Tanh())
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, h_w_wheels_shoulders):
        start_time, episodes, steps, [h_w_wheels_shoulders] = model_start([(h_w_wheels_shoulders, "lin")], self.args.device, self.args.half)
        sensors = self.a(h_w_wheels_shoulders)
        sensors = (sensors + 1) / 2
        [sensors] = model_end(start_time, episodes, steps, [(sensors, "lin")], "\tSENSORS_OUT")
        return(sensors)
    
    
    
if __name__ == "__main__":
    
    sensors_out = Sensors_OUT(args = args)
    
    print("\n\n")
    print(sensors_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(sensors_out, 
                                (episodes, steps, args.h_w_wheels_shoulders_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))


    
#%%



class Voice_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Voice_IN, self).__init__()  
        
        self.args = args
        
        self.a = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.voice_shape,
                embedding_dim = self.args.char_encode_size),
            nn.PReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(
                in_features = self.args.char_encode_size,
                out_features = self.args.hidden_size),
            #nn.BatchNorm1d(self.args.sensors_encode_size),
            nn.PReLU())
                
        self.b = nn.GRU(
            input_size = self.args.hidden_size,
            hidden_size = self.args.hidden_size,
            batch_first = True)
                
        self.c = nn.Sequential(
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.voice_encode_size))
                
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, voice):
        start_time, episodes, steps, [voice] = model_start([(voice, "voice")], self.args.device, self.args.half)
                
        voice = pad_zeros(voice, self.args.max_voice_len)
        voice = torch.argmax(voice, dim = -1).int()
                
        a = self.a(voice)
        _, b = self.b(a)    
        b = b.reshape(episodes, steps, self.args.hidden_size)
        encoding = self.c(b)
        
        [encoding] = model_end(start_time, episodes, steps, [(encoding, "lin")], "\tVoice_IN")
        return(encoding)

    
    
if __name__ == "__main__":
    
    voice_in = Voice_IN(args = args)
    
    print("\n\n")
    print(voice_in)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(voice_in, 
                                (episodes, steps, args.max_voice_len, args.voice_shape)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
#%%



class Voice_OUT(nn.Module):

    def __init__(self, actor = False, args = default_args):
        super(Voice_OUT, self).__init__()  
                
        self.args = args
        self.actor = actor
        
        if(self.args.try_thing):
            self.a = nn.Sequential(
                nn.Linear(
                    in_features = self.args.h_w_wheels_shoulders_size, 
                    out_features = self.args.hidden_size * self.args.max_voice_len),
                #nn.BatchNorm1d(self.args.hidden_size * self.args.max_voice_len),
                nn.PReLU(),
                nn.Dropout(self.args.dropout))
        else:
            self.a = nn.Sequential(
                nn.Linear(
                    in_features = self.args.h_w_wheels_shoulders_size, 
                    out_features = self.args.hidden_size * self.args.max_voice_len),
                nn.BatchNorm1d(self.args.hidden_size * self.args.max_voice_len),
                nn.PReLU(),
                nn.Dropout(self.args.dropout))
            
        self.b = nn.GRU(
            input_size = self.args.hidden_size,
            hidden_size = self.args.hidden_size,
            batch_first = True)
        
        self.mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.voice_shape))
        
        self.std = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.voice_shape),
            nn.Softmax())
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
                
    def forward(self, h_w_wheels_shoulders):
        
        start_time, episodes, steps, [h_w_wheels_shoulders] = model_start([(h_w_wheels_shoulders, "lin")], self.args.device, self.args.half)
                
        h_w_wheels_shoulders = h_w_wheels_shoulders.reshape(episodes * steps, self.args.h_w_wheels_shoulders_size)
        a = self.a(h_w_wheels_shoulders)
        a = a.reshape(episodes * steps, self.args.max_voice_len, self.args.hidden_size)
        b, _ = self.b(a)
        
        if(self.actor):
            mu, std = var(b, self.mu, self.std, self.args)
            voice = sample(mu, std, self.args.device)
            voice_out = torch.tanh(voice)
            log_prob = Normal(mu, std).log_prob(voice) - torch.log(1 - voice_out.pow(2) + 1e-6)
            log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
            log_prob = log_prob.mean(-2)
            voice_out = voice_out.reshape(episodes, steps, self.args.max_voice_len, self.args.voice_shape)
            log_prob = log_prob.reshape(episodes, steps, 1)
            return(voice_out, log_prob)
        else:
            voice_pred = self.mu(b)
            voice_pred = voice_pred.reshape(episodes, steps, self.args.max_voice_len, self.args.voice_shape)
            return(voice_pred)
    
    
    
if __name__ == "__main__":
    
    voice_out = Voice_OUT(actor = False, args = args)
        
    print("\n\n")
    print(voice_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(voice_out, 
                                (episodes, steps, args.h_w_wheels_shoulders_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
#%%
    
    
    
class Obs_OUT(nn.Module):
    
    def __init__(self, args = default_args):
        super(Obs_OUT, self).__init__()  
        
        self.args = args 
        self.rgbd_out = RGBD_OUT(self.args)
        self.sensors_out = Sensors_OUT(self.args)
        self.father_voice_out = Voice_OUT(actor = False, args = self.args)
        self.mother_voice_out = Voice_OUT(actor = False, args = self.args)
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, h_w_wheels_shoulders):
        rgbd_pred = self.rgbd_out(h_w_wheels_shoulders)
        sensors_pred = self.sensors_out(h_w_wheels_shoulders)
        father_voice_pred = self.father_voice_out(h_w_wheels_shoulders)
        mother_voice_pred = self.mother_voice_out(h_w_wheels_shoulders)
        return(rgbd_pred, sensors_pred, father_voice_pred, mother_voice_pred)
    
    
    
if __name__ == "__main__":
    
    obs_out = Obs_OUT(args = args)
    
    print("\n\n")
    print(obs_out)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(obs_out, 
                                (episodes, steps, args.h_w_wheels_shoulders_size)))
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
            #nn.BatchNorm1d(self.args.encode_wheels_shoulders_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def forward(self, wheels_shoulders):
        start_time, episodes, steps, [wheels_shoulders] = model_start([(wheels_shoulders, "lin")], self.args.device, self.args.half)
        encoded = self.a(wheels_shoulders)
        [encoded] = model_end(start_time, episodes, steps, [(encoded, "lin")], "\tWheels_Shoulders_IN")

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
