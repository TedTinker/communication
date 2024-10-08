#%% 

import torch
from torch.distributions import Normal
from torchvision.transforms import Resize
from math import log, sqrt
from torch import nn 

from utils import duration, print



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

def episodes_steps(this):
    return(this.shape[0], this.shape[1])

# Is device used correctly here?
def pad_zeros(value, length):
    rows_to_add = length - value.size(-2)
    if(rows_to_add == 0):
        return(value)
    padding_shape = list(value.shape)
    padding_shape[-2] = rows_to_add
    if(value.get_device() == -1):
        device = "cpu"
    else:
        device = value.get_device()
    padding = torch.zeros(padding_shape).to(device)
    padding[..., 0] = 1
    value = torch.cat([value, padding], dim=-2)
    return value

def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std, device):
    e = Normal(0, 1).sample(std.shape).to(device)
    return(mu + e * std)



def rnn_cnn(do_this, to_this):
    episodes, steps = episodes_steps(to_this)
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)



def model_start(model_input_list, device = "cpu", half = False):
    start = duration()
    new_model_inputs = []
    for model_input, layer_type in model_input_list:
        model_input = model_input.to(device)
        if(half):
            model_input = model_input.to(dtype=torch.float16)
        if(layer_type == "lin"):
            if(len(model_input.shape) == 2):   model_input = model_input.unsqueeze(1)
            episodes, steps = episodes_steps(model_input)
            model_input = model_input.reshape(episodes * steps, model_input.shape[2])
        if(layer_type == "cnn"):
            if(len(model_input.shape) == 4): model_input = model_input.unsqueeze(1)
            episodes, steps = episodes_steps(model_input)
            model_input = model_input.reshape(episodes * steps, model_input.shape[2], model_input.shape[3], model_input.shape[4]).permute(0, -1, 1, 2)
        if(layer_type == "comm"):
            if(len(model_input.shape) == 2):  model_input = model_input.unsqueeze(0)
            if(len(model_input.shape) == 3):  model_input = model_input.unsqueeze(1)
            episodes, steps = episodes_steps(model_input)
            model_input = model_input.reshape(episodes * steps, model_input.shape[2], model_input.shape[3])
        new_model_inputs.append(model_input)
    return start, episodes, steps, new_model_inputs

def model_end(start, episodes, steps, model_output_list, duration_text = None):
    new_model_outputs = []
    for model_output, layer_type in model_output_list:
        if(layer_type == "lin"):
            model_output = model_output.reshape(episodes, steps, model_output.shape[-1])
        if(layer_type == "cnn"):
            model_output = model_output.permute(0, 2, 3, 1)
            model_output = model_output.reshape(episodes, steps, model_output.shape[1], model_output.shape[2], model_output.shape[3])
        if(layer_type == "comm"):
            model_output = model_output.reshape(episodes, steps, model_output.shape[1], model_output.shape[2])
        new_model_outputs.append(model_output)
    if(duration_text != None):
        print(duration_text + ":", duration() - start)
    return new_model_outputs



class ConstrainedConv1d(nn.Conv1d):
    def forward(self, input):
        return nn.functional.conv1d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
class Ted_Conv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_sizes = [1,2,3], stride = 1):
        super(Ted_Conv1d, self).__init__()
        
        self.Conv1ds = nn.ModuleList()
        for kernel, out_channel in zip(kernel_sizes, out_channels):
            padding = (kernel-1)//2
            layer = nn.Sequential(
                ConstrainedConv1d(
                    in_channels = in_channels,
                    out_channels = out_channel,
                    kernel_size = kernel,
                    padding = padding,
                    padding_mode = "reflect",
                    stride = stride))
            self.Conv1ds.append(layer)
                
    def forward(self, x):
        y = []
        for Conv1d in self.Conv1ds: y.append(Conv1d(x)) 
        return(torch.cat(y, dim = -2))
    
    
    
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
class Ted_Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_sizes = [(1,1),(3,3),(5,5)], stride = 1):
        super(Ted_Conv2d, self).__init__()
        
        self.Conv2ds = nn.ModuleList()
        for kernel, out_channel in zip(kernel_sizes, out_channels):
            if(type(kernel) == int): 
                kernel = (kernel, kernel)
            padding = ((kernel[0]-1)//2, (kernel[1]-1)//2)
            layer = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = in_channels,
                    out_channels = out_channel,
                    kernel_size = kernel,
                    padding = padding,
                    padding_mode = "reflect",
                    stride = stride))
            self.Conv2ds.append(layer)
                
    def forward(self, x):
        y = []
        for Conv2d in self.Conv2ds: y.append(Conv2d(x)) 
        return(torch.cat(y, dim = -3))