#%%
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
import torchgan.layers as gg
from torchinfo import summary as torch_summary

from utils import print, default_args, init_weights, attach_list, detach_list, \
    episodes_steps, pad_zeros, Ted_Conv1d, Ted_Conv2d, create_comm_mask, var, sample, rnn_cnn, duration, ConstrainedConv1d, ConstrainedConv2d
from mtrnn import MTRNN

d = .01

if __name__ == "__main__":
    
    args = default_args
    episodes = args.batch_size ; steps = args.max_steps



class RGBD_IN(nn.Module):

    def __init__(self, args = default_args):
        super(RGBD_IN, self).__init__()  
        
        self.args = args 
        
        rgbd_size = (1, 4, args.image_size, args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.rgbd_in = nn.Sequential(
            nn.Conv2d(
                in_channels = 4,
                out_channels = self.args.hidden_size,
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(self.args.hidden_size),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size = (2, 2),
                stride = (2, 2)),
            nn.Dropout(d),
            
            nn.Conv2d(
                in_channels = self.args.hidden_size,
                out_channels = self.args.hidden_size,
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(self.args.hidden_size),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size = (2, 2),
                stride = (2, 2)),
            nn.Dropout(d),
            
            nn.Conv2d(
                in_channels = self.args.hidden_size,
                out_channels = self.args.hidden_size,
                kernel_size = 1),
            nn.BatchNorm2d(self.args.hidden_size),
            nn.PReLU())
        
        example = self.rgbd_in(example)
        rgbd_latent_size = example.flatten(1).shape[1]
        
        self.rgbd_in_lin = nn.Sequential(
            nn.Linear(
                in_features = rgbd_latent_size, 
                out_features = args.encode_rgbd_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd):
        start = duration()
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        episodes, steps = episodes_steps(rgbd)
        rgbd = rgbd.permute(0, 1, 4, 2, 3)
        rgbd = (rgbd * 2) - 1
        rgbd = rnn_cnn(self.rgbd_in, rgbd).flatten(2)
        rgbd = self.rgbd_in_lin(rgbd)
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
    
    
    
class Comm_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Comm_IN, self).__init__()  
        
        self.args = args
        
        self.comm_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.comm_shape,
                embedding_dim = self.args.encode_size),
            nn.PReLU(),
            nn.Dropout(d))
        
        self.comm_cnn = nn.Sequential(
            Ted_Conv1d(
                in_channels = self.args.encode_size, 
                out_channels = [self.args.hidden_size // 4] * 4, 
                kernels = [1, 3, 5, 7]),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU(),
            nn.Dropout(d))
        
        self.comm_rnn = nn.GRU(
            input_size = self.args.hidden_size,
            hidden_size = self.args.hidden_size,
            batch_first = True)
        
        self.comm_lin = nn.Sequential(
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.encode_comm_size),
            nn.PReLU())
                
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(self, comm):
        start = duration()
        if(len(comm.shape) == 2):  comm = comm.unsqueeze(0)
        if(len(comm.shape) == 3):  comm = comm.unsqueeze(1)
        episodes, steps = episodes_steps(comm)
        comm = pad_zeros(comm, self.args.max_comm_len)
        mask, last_indices = create_comm_mask(comm)
        comm = torch.argmax(comm, dim = -1)
        comm = self.comm_embedding(comm.int())
        comm = comm.reshape((episodes*steps, self.args.max_comm_len, self.args.encode_size))
        comm = self.comm_cnn(comm.permute((0, 2, 1))).permute((0, 2, 1))
        comm, _ = self.comm_rnn(comm)
        comm = comm.reshape((episodes, steps, self.args.max_comm_len, self.args.hidden_size))
        idx0 = torch.arange(comm.size(0)).unsqueeze(1).expand_as(last_indices)
        idx1 = torch.arange(comm.size(1)).expand_as(last_indices) 
        comm = comm[idx0, idx1, last_indices, :] 
        comm = self.comm_lin(comm)
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


    
class Other_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Other_IN, self).__init__()
        
        self.args = args 
        
        self.other_in = nn.Sequential(
            nn.Linear(
                in_features = self.args.other_shape,
                out_features = args.encode_other_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, other):
        #[other] = attach_list([other], self.args.device)
        if(len(other.shape) == 2):   other = other.unsqueeze(1)
        other = self.other_in(other)
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
    
    
    
class Action_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Action_IN, self).__init__()
        
        self.args = args 
        
        self.action_in = nn.Sequential(
            nn.Linear(
                in_features = self.args.action_shape, 
                out_features = args.encode_action_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, action):
        #[action] = attach_list([action], self.args.device)
        if(len(action.shape) == 2):   action = action.unsqueeze(1)
        action = self.action_in(action)
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
    
#%%
    


class RGBD_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(RGBD_OUT, self).__init__()  
        
        self.args = args 
        
        self.rgbd_out_lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + self.args.encode_action_size,
                out_features = 4 * (self.args.image_size//4) * (self.args.image_size//4)),
            nn.PReLU(),
            nn.Dropout(d))
        
        self.rgbd_out = nn.Sequential(
            nn.Conv2d(
                in_channels = 4,
                out_channels = self.args.hidden_size * 4,
                kernel_size = 1),
            nn.BatchNorm2d(self.args.hidden_size * 4),
            nn.PReLU(),
            nn.PixelShuffle(upscale_factor = 2),
            nn.Dropout(d),
            
            nn.Conv2d(
                in_channels = self.args.hidden_size,
                out_channels = self.args.hidden_size * 4,
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(self.args.hidden_size * 4),
            nn.PReLU(),
            nn.PixelShuffle(upscale_factor = 2),
            nn.Dropout(d),
            
            nn.Conv2d(
                in_channels = self.args.hidden_size,
                out_channels = self.args.hidden_size,
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(self.args.hidden_size),
            nn.PReLU(),
            nn.Conv2d(
                in_channels = self.args.hidden_size,
                out_channels = 4,
                kernel_size = 1),)
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, h_w_action):
        start = duration()
        if(len(h_w_action.shape) == 2): h_w_action = h_w_action.unsqueeze(1)
        episodes, steps = episodes_steps(h_w_action)
        h_w_action = self.rgbd_out_lin(h_w_action)
        rgbd = h_w_action.reshape(episodes, steps, 4, self.args.image_size//4, self.args.image_size//4)
        rgbd = rnn_cnn(self.rgbd_out, rgbd)
        rgbd = rgbd.permute(0, 1, 3, 4, 2)
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



class Comm_OUT(nn.Module):

    def __init__(self, actor = False, args = default_args):
        super(Comm_OUT, self).__init__()  
                
        self.args = args
        self.actor = actor
        
        if(self.actor):
            self.comm_lin = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + args.encode_action_size, 
                    out_features = self.args.hidden_size),
                nn.PReLU(),
                nn.Dropout(d))
        else:
            self.comm_lin = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + self.args.encode_action_size, 
                    out_features = self.args.hidden_size),
                nn.PReLU(),
                nn.Dropout(d))
            
        self.comm_rnn = nn.GRU(
            input_size = self.args.hidden_size,
            hidden_size = self.args.hidden_size,
            batch_first = True)
        
        self.comm_cnn = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(d),
            Ted_Conv1d(
                in_channels = self.args.hidden_size, 
                out_channels = [self.args.hidden_size // 4] * 4,
                kernels = [1, 3, 5, 7]),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.PReLU())
        
        self.comm_out_mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size, 
                out_features = self.args.comm_shape))
        
        if(self.actor):
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
        h_w_action = self.comm_lin(h_w_action)
        h_w_action = h_w_action.reshape(episodes * steps, 1, self.args.hidden_size)
        comm_h = None
        comm_hs = []
        for i in range(self.args.max_comm_len):
            comm_h, _ = self.comm_rnn(h_w_action, comm_h if comm_h == None else comm_h.permute(1, 0, 2))
            comm_hs.append(comm_h)
        comm_h = torch.cat(comm_hs, dim = -2)
        comm_h = self.comm_cnn(comm_h.permute((0, 2, 1))).permute((0, 2, 1))
        comm_h = comm_h.reshape(episodes, steps, self.args.max_comm_len, self.args.hidden_size)
        
        if(self.actor):
            mu, std = var(comm_h, self.comm_out_mu, self.comm_out_std, self.args)
            comm = sample(mu, std, self.args.device)
            comm_out = torch.tanh(comm)
            log_prob = Normal(mu, std).log_prob(comm) - torch.log(1 - comm_out.pow(2) + 1e-6)
            log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
            log_prob = log_prob.mean(-2)
            return(comm_out, log_prob)
        else:
            comm_pred = self.comm_out_mu(comm_h)
            mask, last_indexes = create_comm_mask(comm_pred)
            comm_pred = comm_pred * mask.unsqueeze(-1).tile((1, 1, 1, self.args.comm_shape))
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
    
    comm_out = Comm_OUT(actor = True, args = args)
    
    print("\n\n")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(comm_out, 
                                (episodes, steps, args.pvrnn_mtrnn_size + args.encode_action_size)))
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
        other = self.other_out_lin(h_w_action)
        
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
    
    
# %%



class Discriminator(nn.Module):
    
    def __init__(self, args = default_args):
        super(Discriminator, self).__init__()  
        
        self.args = args
        self.obs_in = Obs_IN(self.args)
        
        self.stat_quantity = self.get_stats(
            torch.zeros((1, 1, self.args.image_size, self.args.image_size, 4))).shape[1]
        
        self.lin = nn.Sequential(
            nn.Linear(
                in_features = self.args.pvrnn_mtrnn_size + self.args.encode_action_size + self.args.encode_obs_size, 
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
        c_mean   = images.mean((2,3))
        #c_q      = torch.quantile(images.flatten(3), q = torch.tensor([.01, .25, .5, .75, .99]), dim = 2)#.permute(1, 2, 0).flatten(1)
        c_var    = torch.var(images, dim = (2, 3)) 
        
        # Statistics for entire batch
        b_mean   = torch.tile(c_mean.mean((0,1)), (images.shape[0], images.shape[1], 1))
        #b_q      = torch.tile(torch.quantile(images.permute(1, 0, 2, 3).flatten(1), q = torch.tensor([.01, .25,  .5, .75, .99]), dim = 1).flatten(0).unsqueeze(0), (images.shape[0], 1))
        b_var    = torch.tile(torch.var(images, dim = (0,1,2,3)).unsqueeze(0).unsqueeze(0), (images.shape[0], 1))
        
        stats = torch.cat([c_mean,
                        b_mean, b_var], dim = 1).to(self.args.device)
        return(stats)
    
    
    """def get_stats(images):
        # Statistics for each color of each images
        c_mean   = images.mean(2).mean(2)
        c_q      = torch.quantile(images.flatten(2), q = torch.tensor([.25, .5, .75]), dim = 2).permute(1, 2, 0).flatten(1)
        c_mode   = images.mode(2)[0].mode(2)[0]
        c_var    = torch.var(images, dim = (2, 3)) 
        
        # Statistics for entire batch
        b_mean   = torch.tile(c_mean.mean(0).unsqueeze(0), (images.shape[0], 1))
        b_q      = torch.tile(torch.quantile(images.permute(1, 0, 2, 3).flatten(1), q = torch.tensor([.01, .05, .15, .25, .35, .5, .65, .75, .85, .95, .99]), dim = 1).flatten(0).unsqueeze(0), (images.shape[0], 1))
        b_mode   = torch.tile(c_mode.mode(0)[0].unsqueeze(0), (images.shape[0], 1))
        b_var    = torch.tile(torch.var(images, dim = (0,1,2,3)).unsqueeze(0).unsqueeze(0), (images.shape[0], 1))
        
        #for stat in [c_mean, c_q, c_mode, c_var, b_mean, b_q, b_mode, b_var]:
        #    print(stat.shape)
        
        stats = torch.cat([c_mean, c_q, c_mode, c_var, 
                        b_mean, b_q, b_mode, b_var], dim = 1)
        return(stats)
    """
            
    def forward(self, h_w_action, rgbd, comm, other):
        episodes, steps = episodes_steps(h_w_action)
        obs = self.obs_in(rgbd, comm, other)
        h_w_action = torch.cat([h_w_action, obs], dim = -1)
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
                                ((episodes, steps, args.pvrnn_mtrnn_size +args.encode_action_size),
                                (episodes, steps, args.image_size, args.image_size, 4),
                                (episodes, steps, args.max_comm_len, args.comm_shape),
                                (episodes, steps, args.other_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
# %%
