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

def generate_1d_positional_layers(batch_size, length, device='cpu'):
    x = torch.linspace(-1, 1, steps=length).view(1, 1, 1, length).repeat(batch_size, 1, length, 1)
    x = x.to(device)
    return x
    
def generate_2d_positional_layers(batch_size, image_size, device='cpu'):
    x = torch.linspace(-1, 1, steps=image_size).view(1, 1, 1, image_size).repeat(batch_size, 1, image_size, 1)
    y = torch.linspace(-1, 1, steps=image_size).view(1, 1, image_size, 1).repeat(batch_size, 1, 1, image_size)
    x, y = x.to(device), y.to(device)
    return torch.cat([x, y], dim=1)



def generate_2d_sinusoidal_positions(batch_size, image_size, d_model=2, device='cpu'):
    assert d_model % 2 == 0, "d_model should be even."
    x = torch.arange(image_size, dtype=torch.float32, device=device).unsqueeze(0).expand(image_size, image_size)
    y = torch.arange(image_size, dtype=torch.float32, device=device).unsqueeze(1).expand(image_size, image_size)
    x = x.unsqueeze(2).tile((1, 1, d_model // 2))
    y = y.unsqueeze(2).tile((1, 1, d_model // 2))

    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-log(10000.0) / (d_model // 2)))
    div_term = torch.tile(div_term.unsqueeze(0).unsqueeze(0), (image_size, image_size, 1))
    
    pe = torch.zeros(image_size, image_size, d_model, device=device)
    pe[:, :, 0::2] = torch.sin(x * div_term) + torch.sin(y * div_term)
    pe[:, :, 1::2] = torch.cos(x * div_term) + torch.cos(y * div_term)
    pe = torch.tile(pe.unsqueeze(0), (batch_size, 1, 1, 1))
    pe = pe.permute(0, 3, 1, 2) / 2
    return pe



def hsv_to_circular_hue(hsv_image):
    hue = hsv_image[:, 0, :, :]
    hue_sin = (torch.sin(hue) + 1) / 2
    hue_cos = (torch.cos(hue) + 1) / 2
    hsv_circular = torch.stack([hue_sin, hue_cos, hsv_image[:, 1, :, :], hsv_image[:, 2, :, :]], dim=1)
    return hsv_circular



def model_start(model_input_list, device = "cpu"):
    start = duration()
    new_model_inputs = []
    for model_input, layer_type in model_input_list:
        model_input = model_input.to(device)
        #if(str(device) != "cpu"):
        #    model_input = model_input.to(dtype=torch.float16)
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
    
    

class ResidualBlock2d(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, paddings, padding_modes, activations, dropouts):
        super(ResidualBlock2d, self).__init__()
        
        # Ensure all parameter lists have compatible lengths
        assert len(channels) - 1 == len(kernel_sizes) == len(strides) == len(paddings) == \
               len(padding_modes) == len(activations) == len(dropouts) - 1, "All parameter lists must have compatible lengths."
        
        self.layers = nn.ModuleList()
        self.adjust_channels = channels[0] != channels[-1] or strides[0] != 1
        self.final_activation = activations[-1] is not None
        
        # Add the convolutional layers, batch normalization, activation functions, and dropout based on the parameters
        for i in range(len(channels) - 1):
            self.layers.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    padding_mode=padding_modes[i]))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))
            
            if activations[i] is not None:  # Check if the activation is specified
                self.layers.append(activations[i]())
            
            if dropouts[i] > 0:  # Add dropout layer if dropout rate is greater than 0
                self.layers.append(nn.Dropout2d(dropouts[i]))
                
        self.final_layer = nn.ModuleList()
        if self.final_activation:
            self.final_layer.append(activations[-1]())
        if dropouts[-1] > 0:
            self.final_layer.append(nn.Dropout2d(dropouts[-1]))
        self.final_layer.append(nn.BatchNorm2d(channels[-1]))
        
        # Add an adjustment layer to match the shortcut connection, if necessary
        if self.adjust_channels:
            self.adjustment_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=channels[0],
                    out_channels=channels[-1],
                    kernel_size=1,
                    stride=strides[0],  # Adjust based on the first stride value if needed
                    padding=0),
                nn.BatchNorm2d(channels[-1]))
        
    def forward(self, x):
        identity = x
        for layer in self.layers:
            x = layer(x)
        if self.adjust_channels:
            identity = self.adjustment_layer(identity)
        x += identity
        for layer in self.final_layer:
            x = layer(x)
        return x
    


class ResidualBlock1d(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, paddings, padding_modes, activations, dropouts):
        super(ResidualBlock1d, self).__init__()
        
        # Ensure all parameter lists have compatible lengths
        assert len(channels) - 1 == len(kernel_sizes) == len(strides) == len(paddings) == \
               len(padding_modes) == len(activations) == len(dropouts) - 1, "All parameter lists must have compatible lengths."
        
        self.layers = nn.ModuleList()
        self.adjust_channels = channels[0] != channels[-1] or strides[0] != 1
        self.final_activation = activations[-1] is not None
        
        # Add the convolutional layers, batch normalization, activation functions, and dropout based on the parameters
        for i in range(len(channels) - 1):
            self.layers.append(
                nn.Conv1d(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    padding_mode=padding_modes[i]))
            self.layers.append(nn.BatchNorm1d(channels[i+1]))
            
            if activations[i] is not None:  # Check if the activation is specified
                self.layers.append(activations[i]())
            
            if dropouts[i] > 0:  # Add dropout layer if dropout rate is greater than 0
                self.layers.append(nn.Dropout1d(dropouts[i]))
                
        self.final_layer = nn.ModuleList()
        if self.final_activation:
            self.final_layer.append(activations[-1]())
        if dropouts[-1] > 0:
            self.final_layer.append(nn.Dropout1d(dropouts[-1]))
        self.final_layer.append(nn.BatchNorm1d(channels[-1]))
        
        # Add an adjustment layer to match the shortcut connection, if necessary
        if self.adjust_channels:
            self.adjustment_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels[0],
                    out_channels=channels[-1],
                    kernel_size=1,
                    stride=strides[0],  # Adjust based on the first stride value if needed
                    padding=0),
                nn.BatchNorm1d(channels[-1]))
        
    def forward(self, x):
        identity = x
        for layer in self.layers:
            x = layer(x)
        if self.adjust_channels:
            identity = self.adjustment_layer(identity)
        x += identity
        for layer in self.final_layer:
            x = layer(x)
        return x
    
    
    
class DenseBlock(nn.Module):
    def __init__(self, input_channels, growth_rates, kernel_sizes, paddings, padding_modes, activations, dropouts):
        super(DenseBlock, self).__init__()
        
        assert len(growth_rates) == len(kernel_sizes) == len(activations) == len(dropouts)
        
        self.layers = nn.ModuleList()
        for i in range(len(growth_rates)):
            layer_channels = input_channels + sum(growth_rates[:i])
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(layer_channels),
                activations[i](),
                nn.Conv2d(
                    in_channels = layer_channels, 
                    out_channels = growth_rates[i], 
                    kernel_size = kernel_sizes[i], 
                    padding = paddings[i],
                    padding_mode = padding_modes[i]),
                nn.Dropout2d(dropouts[i])))
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            these_features = torch.cat(features, 1)
            new_features = layer(these_features)
            features.append(new_features)
        return torch.cat(features, 1)
    
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'position encoding' matrix in advance
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register 'pe' as a constant buffer that does not require gradients
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Add position encoding to input tensor
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
    
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, src, src_mask = None):
        src = self.encoder(src) * sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
    
    
    
class ImageTransformer(nn.Module):
    def __init__(self, in_channels, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(ImageTransformer, self).__init__()
        self.transformer = TransformerModel(ntoken, ninp, nhead, nhid, nlayers, dropout)
        
        # Resize and flatten image to fit transformer input
        self.resize = Resize((ninp, ninp))  # Assuming square images for simplicity
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels * ninp * ninp, ninp)  # Adapt to transformer input dimension
        
    def forward(self, img, src_mask = None):
        # img shape: [batch_size, in_channels, H, W]
        img = self.resize(img)
        img = self.flatten(img)
        img = self.linear(img)
        img = img.unsqueeze(0)  # Add sequence dimension
        output = self.transformer(img, src_mask)
        return output