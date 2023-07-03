#%% 

import torch
from torch import nn 
from torch.distributions import Normal
import torch.nn.functional as F
from torchinfo import summary as torch_summary
from torchgan.layers import SelfAttention2d

from utils import default_args, init_weights, ConstrainedConv2d, ConstrainedConvTranspose2d, Ted_Conv2d, shapes, colors, goals, print
spe_size = 1 ; action_size = 4



def episodes_steps(this):
    return(this.shape[0], this.shape[1])

def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std):
    e = Normal(0, 1).sample(std.shape).to("cuda" if std.is_cuda else "cpu")
    return(mu + e * std)

def rnn_cnn(do_this, to_this):
    episodes, steps = episodes_steps(to_this)
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)



class RGBD_IN(nn.Module):

    def __init__(self, args = default_args):
        super(RGBD_IN, self).__init__()  
        
        rgbd_size = (1, 4, args.image_size, args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.rgbd_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 4,
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU(),
            nn.AvgPool2d(
                kernel_size = (3,3),
                stride = (2,2),
                padding = (1,1)),
            ConstrainedConv2d(
                in_channels = 16,
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU(),
            nn.AvgPool2d(
                kernel_size = (3,3),
                stride = (2,2),
                padding = (1,1)),
            )
        example = self.rgbd_in(example)
        rgbd_latent_size = example.flatten(1).shape[1]
        
        self.rgbd_in_lin = nn.Sequential(
            nn.Linear(rgbd_latent_size, args.hidden_size),
            nn.PReLU())
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        rgbd = (rgbd.permute(0, 1, 4, 2, 3) * 2) - 1
        rgbd = rnn_cnn(self.rgbd_in, rgbd).flatten(2)
        rgbd = self.rgbd_in_lin(rgbd)
        return(rgbd)
    
    
    
class Spe_IN(nn.Module):
        
    def __init__(self, args = default_args):
        super(Spe_IN, self).__init__()  
        
        self.args = args 
        
        self.spe_in = nn.Sequential(
            nn.Linear(1, args.hidden_size),
            nn.PReLU())
        
    def forward(self, spe):
        if(len(spe.shape) == 2): spe = spe.unsqueeze(1)
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        return(self.spe_in(spe))
    


class Comm_IN(nn.Module):
        
    def __init__(self, args = default_args):
        super(Comm_IN, self).__init__()  
        
        self.comm_in = nn.Sequential(
            nn.Linear(args.symbols, args.hidden_size),
            nn.PReLU())
        
        self.goal_comm_in = nn.Sequential(
            nn.Linear(len(shapes) + len(colors) + len(goals), args.hidden_size),
            nn.PReLU())
        
    def forward(self, comm, goal_comm):
        if(len(comm.shape) == 2): comm = comm.unsqueeze(1)
        if(goal_comm): return(self.goal_comm_in(comm))
        else:          return(self.comm_in(comm))
        
        
        
class Action_IN(nn.Module):
        
    def __init__(self, args = default_args):
        super(Action_IN, self).__init__()  
        
        self.args = args
        
        self.velocity_in = nn.Sequential(
            nn.Linear(2, args.hidden_size),
            nn.PReLU())
        
        self.arms_in = nn.Sequential(
            nn.Linear(2, args.hidden_size),
            nn.PReLU())
        
        self.comm_in = nn.Sequential(
            nn.Linear(self.args.symbols, args.hidden_size),
            nn.PReLU())
        
    def forward(self, action, arms, comm):
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        episodes, steps = episodes_steps(action)
        v = self.velocity_in(action[:,:,:2])
        if(arms): a = self.arms_in(action[:,:,2:4])
        else:     a = torch.zeros((episodes, steps, self.args.hidden_size))
        if(comm): c = self.comm_in(action[:,:,4:])
        else:     c = torch.zeros((episodes, steps, self.args.hidden_size))
        return(v + a + c)
        
        
        

class Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        
        self.rgbd_in = RGBD_IN(args)
        
        self.spe_in = Spe_IN(args)
        
        self.comm_in = Comm_IN(args)
        
        self.prev_action_in = Action_IN(args)
        
        self.action_in = Action_IN(args)
        
        self.h_in = nn.Sequential(
            nn.PReLU())
        
        self.zp_mu = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Tanh())
        self.zp_std = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Softplus())
        
        self.zq_mu = nn.Sequential(
            nn.Linear(3 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Tanh())
        self.zq_std = nn.Sequential(
            nn.Linear(3 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Softplus())
        
        self.gru = nn.GRU(
            input_size =  args.state_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.gen_shape = (4, args.image_size//4, args.image_size//4)
        self.rgbd_out_lin = nn.Sequential(
            nn.Linear(2 * args.hidden_size, self.gen_shape[0] * self.gen_shape[1] * self.gen_shape[2]),
            nn.PReLU())
        
        self.rgbd_out = nn.Sequential(
            ConstrainedConv2d(
                in_channels = self.gen_shape[0],
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU(),
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
            ConstrainedConv2d(
                in_channels = 16,
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU(),
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
            ConstrainedConv2d(
                in_channels = 16,
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU(),
            ConstrainedConv2d(
                in_channels = 16,
                out_channels = 4,
                kernel_size = (1,1)))
        
        self.spe_out = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, spe_size))
        
        self.comm_out = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.symbols),
            nn.Sigmoid())
        
        self.goal_comm_out = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, len(shapes) + len(colors) + len(goals)),
            nn.Sigmoid())
        
        self.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd, spe, comm, prev_a, h_q_m1, arms = False, communicating = False, goal_comm = False):
        rgbd = self.rgbd_in(rgbd)
        spe = self.spe_in(spe)
        comm = self.comm_in(comm, goal_comm)
        prev_a = self.prev_action_in(prev_a, arms, communicating)
        relu_h_q_m1 = self.h_in(h_q_m1)
        zp_mu, zp_std = var(torch.cat((relu_h_q_m1, prev_a),                    dim=-1), self.zp_mu, self.zp_std, self.args)
        zq_mu, zq_std = var(torch.cat((relu_h_q_m1, prev_a, rgbd + spe + comm), dim=-1), self.zq_mu, self.zq_std, self.args)        
        zq = sample(zq_mu, zq_std)
        h_q, _ = self.gru(zq, h_q_m1.permute(1, 0, 2))
        return((zp_mu, zp_std), (zq_mu, zq_std), h_q)

    def get_preds(self, action, z_mu, z_std, h_q_m1, quantity = 1, arms = False, communicating = False, goal_comm = False):
        episodes, steps = episodes_steps(z_mu)
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        h_q_m1 = h_q_m1.permute(1, 0, 2)
        h, _ = self.gru(z_mu, h_q_m1)        
        action = self.action_in(action, arms, communicating)
        
        rgbd = self.rgbd_out_lin(torch.cat((h, action), dim=-1)).view((episodes, steps, self.gen_shape[0], self.gen_shape[1], self.gen_shape[2]))
        rgbd_mu_pred = rnn_cnn(self.rgbd_out, rgbd).permute(0, 1, 3, 4, 2)
        spe_mu_pred  = self.spe_out(torch.cat((h, action), dim=-1))
        if(goal_comm): comm_mu_pred = self.goal_comm_out(torch.cat((h, action), dim=-1))
        else:          comm_mu_pred = self.comm_out(     torch.cat((h, action), dim=-1))
                
        pred_rgbd = [] ; pred_spe = [] ; pred_comm = [] 
        for _ in range(quantity):
            z = sample(z_mu, z_std)
            h, _ = self.gru(z, h_q_m1)
            rgbd = self.rgbd_out_lin(torch.cat((h, action), dim=-1)).view((episodes, steps, self.gen_shape[0], self.gen_shape[1], self.gen_shape[2]))
            pred_rgbd.append((rnn_cnn(self.rgbd_out, rgbd).permute(0, 1, 3, 4, 2)))
            pred_spe.append(self.spe_out(torch.cat((h, action), dim=-1)))
            if(goal_comm): pred_comm.append(self.goal_comm_out(torch.cat((h, action), dim=-1)))
            else:          pred_comm.append(self.comm_out(torch.cat((h, action), dim=-1)))
        return((rgbd_mu_pred, pred_rgbd), (spe_mu_pred, pred_spe), (comm_mu_pred, pred_comm))



class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.rgbd_in = RGBD_IN(args)
        
        self.spe_in = Spe_IN(args)
        
        self.comm_in = Comm_IN(args)
        
        self.action_in = Action_IN(args)
        
        self.h_in = nn.Sequential(
            nn.PReLU())
        
        self.gru = nn.GRU(
            input_size =  2 * args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.velocity_mu = nn.Sequential(
            nn.Linear(args.hidden_size, 2))
        self.velocity_std = nn.Sequential(
            nn.Linear(args.hidden_size, 2),
            nn.Softplus())
        
        self.arms_mu = nn.Sequential(
            nn.Linear(args.hidden_size, 2))
        self.arms_std = nn.Sequential(
            nn.Linear(args.hidden_size, 2),
            nn.Softplus())
        
        self.symbol_mu = nn.Sequential(
            nn.Linear(args.hidden_size, args.symbols))
        self.symbol_std = nn.Sequential(
            nn.Linear(args.hidden_size, args.symbols),
            nn.Softmax(dim=-1))

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, spe, comm, prev_action, h = None, arms = False, communicating = False, goal_comm = False):
        episodes, steps = episodes_steps(rgbd)
        rgbd = self.rgbd_in(rgbd)
        spe = self.spe_in(spe)
        comm = self.comm_in(comm, goal_comm)
        prev_action = self.action_in(prev_action, arms, communicating)
        h, _ = self.gru(torch.cat((rgbd + spe + comm, prev_action), dim=-1), h)
        relu_h = self.h_in(h)
        
        velocity_mu, velocity_std = var(relu_h, self.velocity_mu, self.velocity_std, self.args)
        velocity_x = sample(velocity_mu, velocity_std)
        velocity_action = torch.tanh(velocity_x)
        velocity_log_prob = Normal(velocity_mu, velocity_std).log_prob(velocity_x) - torch.log(1 - velocity_action.pow(2) + 1e-6)
        
        if(arms):
            arms_mu, arms_std = var(relu_h, self.arms_mu, self.arms_std, self.args)
            arms_x = sample(arms_mu, arms_std)
            arms_action = torch.tanh(arms_x)
            arms_log_prob = Normal(arms_mu, arms_std).log_prob(arms_x) - torch.log(1 - arms_action.pow(2) + 1e-6)
        else:
            arms_action = torch.zeros((episodes, steps, 2))
            arms_log_prob = torch.zeros((episodes, steps, 2))

        if(communicating):
            symbol_mu, symbol_std = var(relu_h, self.symbol_mu, self.symbol_std, self.args)
            symbol_x = sample(symbol_mu, symbol_std)
            symbol_action = torch.tanh(symbol_x)
            symbol_log_prob = Normal(symbol_mu, symbol_std).log_prob(symbol_x) - torch.log(1 - symbol_action.pow(2) + 1e-6)
            symbol_action = F.softmax(symbol_action)
            symbol_action = (symbol_action * 2) - 1
        else:
            symbol_action = torch.zeros((episodes, steps, self.args.symbols))
            symbol_log_prob = torch.zeros((episodes, steps, self.args.symbols))
        
        action = torch.cat((velocity_action, arms_action, symbol_action), dim=-1)
        log_prob = torch.cat((velocity_log_prob, arms_log_prob, symbol_log_prob), dim=-1)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        
        return(action, log_prob, h)
    
    
    
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.rgbd_in = RGBD_IN(args)
        
        self.spe_in = Spe_IN(args)
        
        self.comm_in = Comm_IN(args)
        
        self.action_in = Action_IN(args)
        
        self.h_in = nn.Sequential(
            nn.PReLU())
        
        self.gru = nn.GRU(
            input_size =  2 * args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, 1))

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, spe, comm, action, h = None, arms = False, communicating = False, goal_comm = False):
        rgbd = self.rgbd_in(rgbd)
        spe = self.spe_in(spe)
        comm = self.comm_in(comm, goal_comm)
        action = self.action_in(action, arms, communicating)
        h, _ = self.gru(torch.cat((rgbd + spe + comm, action), dim=-1), h)
        Q = self.lin(self.h_in(h))
        return(Q, h)
    
    
    
class Actor_HQ(nn.Module):

    def __init__(self, args = default_args):
        super(Actor_HQ, self).__init__()
        
        self.args = args
        
        self.lin = nn.Sequential(
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU())
        
        self.velocity_mu = nn.Sequential(
            nn.Linear(args.hidden_size, 2))
        self.velocity_std = nn.Sequential(
            nn.Linear(args.hidden_size, 2),
            nn.Softplus())
        
        self.arms_mu = nn.Sequential(
            nn.Linear(args.hidden_size, 2))
        self.arms_std = nn.Sequential(
            nn.Linear(args.hidden_size, 2),
            nn.Softplus())
        
        self.symbol_mu = nn.Sequential(
            nn.Linear(args.hidden_size, args.symbols))
        self.symbol_std = nn.Sequential(
            nn.Linear(args.hidden_size, args.symbols),
            nn.Softmax(dim=-1))

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, h, arms = False, communicating = False, goal_comm = False):
        episodes, steps = episodes_steps(h)
        x = self.lin(h)
        
        velocity_mu, velocity_std = var(x, self.velocity_mu, self.velocity_std, self.args)
        velocity_x = sample(velocity_mu, velocity_std)
        velocity_action = torch.tanh(velocity_x)
        velocity_log_prob = Normal(velocity_mu, velocity_std).log_prob(velocity_x) - torch.log(1 - velocity_action.pow(2) + 1e-6)
        
        if(arms):
            arms_mu, arms_std = var(x, self.arms_mu, self.arms_std, self.args)
            arms_x = sample(arms_mu, arms_std)
            arms_action = torch.tanh(arms_x)
            arms_log_prob = Normal(arms_mu, arms_std).log_prob(arms_x) - torch.log(1 - arms_action.pow(2) + 1e-6)
        else:
            arms_action = torch.zeros((episodes, steps, 2))
            arms_log_prob = torch.zeros((episodes, steps, 2))

        if(communicating):
            symbol_mu, symbol_std = var(x, self.symbol_mu, self.symbol_std, self.args)
            symbol_x = sample(symbol_mu, symbol_std)
            symbol_action = torch.tanh(symbol_x)
            symbol_log_prob = Normal(symbol_mu, symbol_std).log_prob(symbol_x) - torch.log(1 - symbol_action.pow(2) + 1e-6)
            symbol_action = F.softmax(symbol_action)
            symbol_action = (symbol_action * 2) - 1
        else:
            symbol_action = torch.zeros((episodes, steps, self.args.symbols))
            symbol_log_prob = torch.zeros((episodes, steps, self.args.symbols))
            
        action = torch.cat((velocity_action, arms_action, symbol_action), dim=-1)
        log_prob = torch.cat((velocity_log_prob, arms_log_prob, symbol_log_prob), dim=-1)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        
        return(action, log_prob, None)
    
    
    
class Critic_HQ(nn.Module):

    def __init__(self, args = default_args):
        super(Critic_HQ, self).__init__()
        
        self.args = args
        
        self.action_in = Action_IN(args)
        
        self.lin = nn.Sequential(
            nn.PReLU(),
            nn.Linear(2 * args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, 1))

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, h, action, arms = False, comm = False):
        a = self.action_in(action, arms, comm)
        Q = self.lin(torch.cat((h, a), dim=-1))
        return(Q, None)
    


if __name__ == "__main__":
    
    args = default_args
    args.dkl_rate = 1
    
    
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, (
        (3, 1, args.image_size, args.image_size, 4), 
        (3, 1, spe_size), 
        (3, 1, args.symbols), 
        (3, 1, action_size + args.symbols), 
        (3, 1, args.hidden_size))))
    


    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, (
        (3, 1, args.image_size, args.image_size, 4), 
        (3, 1, spe_size), 
        (3, 1, args.symbols), 
        (3, 1, action_size + args.symbols))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, (
        (3, 1, args.image_size, args.image_size, 4), 
        (3, 1, spe_size), 
        (3, 1, args.symbols), 
        (3, 1, action_size + args.symbols))))
    
    
    
    actor = Actor_HQ(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, args.hidden_size))))
    
    
    
    critic = Critic_HQ(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, args.hidden_size), (3, 1, action_size + args.symbols))))

# %%
