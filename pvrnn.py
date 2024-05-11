#%%
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, var, sample, attach_list, detach_list, episodes_steps, pad_zeros, dkl, duration
from mtrnn import MTRNN
from submodules import Obs_IN, Obs_OUT, Action_IN, Comm_IN



if __name__ == "__main__":
    
    args = default_args
    episodes = args.batch_size ; steps = args.max_steps



class PVRNN_LAYER(nn.Module):
    
    def __init__(self, time_scale = 1, bottom = False, top = False, args = default_args):
        super(PVRNN_LAYER, self).__init__()
        
        self.args = args 
        self.bottom = bottom
        self.top = top
            
        # Prior: Previous hidden state, plus action if bottom.  
        self.zp_mu = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (self.args.encode_action_size + self.args.encode_comm_size if self.bottom else 0), 
                    out_features = self.args.state_size),
                nn.Tanh())
        self.zp_std = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (self.args.encode_action_size + self.args.encode_comm_size if self.bottom else 0), 
                    out_features = self.args.state_size),
                nn.Softplus())
                            
        # Posterior: Previous hidden state, plus observation and action if bottom, plus lower-layer hidden state otherwise.
        self.zq_mu = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (self.args.encode_obs_size + self.args.encode_action_size + self.args.encode_comm_size if self.bottom else self.args.pvrnn_mtrnn_size), 
                    out_features = self.args.state_size),
                nn.Tanh())
        self.zq_std = nn.Sequential(
                nn.Linear(
                    in_features = self.args.pvrnn_mtrnn_size + (self.args.encode_obs_size + self.args.encode_action_size + self.args.encode_comm_size if self.bottom else self.args.pvrnn_mtrnn_size), 
                    out_features = self.args.state_size),
                nn.Softplus())
                            
        # New hidden state: Previous hidden state, zq value, plus higher-layer hidden state if not top.
        self.mtrnn = MTRNN(
                input_size = self.args.state_size + (self.args.pvrnn_mtrnn_size if not self.top else 0),
                hidden_size = self.args.pvrnn_mtrnn_size, 
                time_constant = time_scale,
                args = self.args)
            
        self.apply(init_weights)
        self.to(self.args.device)
        
    def forward(
        self, 
        prev_hidden_states, 
        obs = None, prev_actions = None, prev_comms_out = None, 
        hidden_states_below = None, 
        prev_hidden_states_above = None):
                
        if(self.bottom):
            zp_inputs = torch.cat([prev_hidden_states, prev_actions, prev_comms_out], dim = -1)
            zq_inputs = torch.cat([prev_hidden_states, obs, prev_actions, prev_comms_out], dim = -1)
        else:
            zp_inputs = prev_hidden_states 
            zq_inputs = torch.cat([prev_hidden_states, hidden_states_below], dim = -1)
            
        episodes, steps = episodes_steps(zp_inputs)
        zp_inputs = zp_inputs.reshape(episodes * steps, zp_inputs.shape[2])
        zq_inputs = zq_inputs.reshape(episodes * steps, zq_inputs.shape[2])
        if(prev_hidden_states_above != None):
            prev_hidden_states_above = prev_hidden_states_above.reshape(episodes * steps, prev_hidden_states_above.shape[2])
            
        zp_mu, zp_std = var(zp_inputs, self.zp_mu, self.zp_std, self.args)
        zp = sample(zp_mu, zp_std, self.args.device)
        zq_mu, zq_std = var(zq_inputs, self.zq_mu, self.zq_std, self.args)
        zq = sample(zq_mu, zq_std, self.args.device)
        kullback_leibler = dkl(zp_mu, zp_std, zq_mu, zq_std)
            
        if(self.top):
            mtrnn_inputs_p = zp
        else:
            mtrnn_inputs_p = torch.cat([zp, prev_hidden_states_above], dim = -1)
            
        if(self.top):
            mtrnn_inputs_q = zq 
        else:
            mtrnn_inputs_q = torch.cat([zq, prev_hidden_states_above], dim = -1)
                    
        mtrnn_inputs_p = mtrnn_inputs_p.reshape((episodes, steps, mtrnn_inputs_p.shape[1]))
        mtrnn_inputs_q = mtrnn_inputs_q.reshape((episodes, steps, mtrnn_inputs_q.shape[1]))
            
        new_hidden_states_p = self.mtrnn(mtrnn_inputs_p, prev_hidden_states)
        new_hidden_states_q = self.mtrnn(mtrnn_inputs_q, prev_hidden_states)
        
        zp_mu = zp_mu.reshape((episodes, steps, zp_mu.shape[1]))
        zp_std = zp_std.reshape((episodes, steps, zp_std.shape[1]))
        zq_mu = zq_mu.reshape((episodes, steps, zq_mu.shape[1]))
        zq_std = zq_std.reshape((episodes, steps, zq_std.shape[1]))
        kullback_leibler = kullback_leibler.reshape((episodes, steps, kullback_leibler.shape[1]))
                        
        return(
            (zp_mu, zp_std, new_hidden_states_p),
            (zq_mu, zq_std, new_hidden_states_q),
            kullback_leibler)        
        
    
if __name__ == "__main__":
    
    bottom_top_layer = PVRNN_LAYER(bottom = True, top = True, args = args)
    
    print("\n\nBOTTOM-TOP")
    print(bottom_top_layer)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(bottom_top_layer, 
                                ((episodes, 1, args.pvrnn_mtrnn_size), 
                                (episodes, 1, args.encode_obs_size),
                                (episodes, 1, args.encode_action_size),
                                (episodes, 1, args.encode_comm_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    bottom_layer = PVRNN_LAYER(bottom = True, args = args)
    
    print("\n\nBOTTOM")
    print(bottom_layer)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(bottom_layer, 
                                ((episodes, 1, args.pvrnn_mtrnn_size), 
                                (episodes, 1, args.encode_obs_size),
                                (episodes, 1, args.encode_action_size),
                                (episodes, 1, args.encode_comm_size),
                                (1,), # No hidden_states_below 
                                (episodes, 1, args.pvrnn_mtrnn_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    top_layer = PVRNN_LAYER(top = True, args = args)
    
    print("\n\nTOP")
    print(top_layer)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(top_layer, 
                                ((episodes, 1, args.pvrnn_mtrnn_size), 
                                (1,), # No obs
                                (1,), # No actions
                                (1,), # No comms out 
                                (episodes, 1, args.pvrnn_mtrnn_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    middle_layer = PVRNN_LAYER(args = args)
    
    print("\n\nMIDDLE")
    print(middle_layer)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(middle_layer, 
                                ((episodes, 1, args.pvrnn_mtrnn_size), 
                                (1,), # No obs
                                (1,), # No actions
                                (1,), # comms out
                                (episodes, 1, args.pvrnn_mtrnn_size),
                                (episodes, 1, args.pvrnn_mtrnn_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%
    
class PVRNN(nn.Module):
    
    def __init__(self, args = default_args):
        super(PVRNN, self).__init__()
        
        self.args = args 
        
        self.obs_in = Obs_IN(self.args)
        self.action_in = Action_IN(self.args)
        self.comm_in = Comm_IN(self.args)
        
        pvrnn_layers = []
        for layer in range(self.args.layers): 
            pvrnn_layers.append(
                PVRNN_LAYER(
                    self.args.time_scales[layer], 
                    bottom = layer == 0, 
                    top = layer + 1 == self.args.layers, 
                    args = self.args))
            
        self.pvrnn_layers = nn.ModuleList(pvrnn_layers)
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(args.device)
        
    def predict(self, h, action):
        h_w_actions = torch.cat([h, action], dim = -1)
        pred_rgbd, pred_comms, pred_sensors = self.predict_obs(h_w_actions)
        return(pred_rgbd, pred_comms, pred_sensors)
        
    def bottom_to_top_step(self, prev_hidden_states, obs = None, prev_actions = None, prev_comms_out = None):
        if(obs != None and len(obs.shape) == 2): 
            obs = obs.unsqueeze(1)
        if(prev_actions != None and len(prev_actions.shape) == 2): 
            prev_actions = prev_actions.unsqueeze(1)
        if(prev_comms_out != None and len(prev_comms_out.shape) == 2): 
            prev_comms_out = prev_comms_out.unsqueeze(1)
                                
        zp_mu_list = []
        zp_std_list = []
        zq_mu_list = []
        zq_std_list = []
        new_hidden_states_list_p = []
        new_hidden_states_list_q = []
        dkls = []
                        
        for layer in range(self.args.layers):
            (zp_mu, zp_std, new_hidden_states_p), (zq_mu, zq_std, new_hidden_states_q), dkl = \
                self.pvrnn_layers[layer](
                    prev_hidden_states[:,layer].unsqueeze(1), 
                    obs, prev_actions, prev_comms_out,
                    new_hidden_states_list_q[-1] if layer > 0 else None, 
                    prev_hidden_states[:,layer+1].unsqueeze(1) if layer + 1 < self.args.layers else None)
    
            for l, o in zip(
                [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list_p, new_hidden_states_list_q, dkls],
                [zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p, new_hidden_states_q, dkl]):            
                l.append(o)
                
        lists = [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list_p, new_hidden_states_list_q, dkls]
        for i in range(len(lists)):
            lists[i] = torch.cat(lists[i], dim=1)
        zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p, new_hidden_states_q, dkl = lists
                                
        return(
            (zp_mu.unsqueeze(1), zp_std.unsqueeze(1), new_hidden_states_p.unsqueeze(1)),
            (zq_mu.unsqueeze(1), zq_std.unsqueeze(1), new_hidden_states_q.unsqueeze(1)),
            dkls)
    
    def forward(self, prev_hidden_states, rgbd, comms_in, sensors, prev_actions, prev_comms_out):
        zp_mu_list = []
        zp_std_list = []
        zq_mu_list = []
        zq_std_list = []
        new_hidden_states_list_p = []
        new_hidden_states_list_q = []
        
        prev_time = duration()
                
        episodes, steps = episodes_steps(rgbd)
        if(prev_hidden_states == None):
            prev_hidden_states = torch.zeros((episodes, self.args.layers, self.args.pvrnn_mtrnn_size))
        obs = self.obs_in(rgbd, comms_in, sensors)
        prev_actions = self.action_in(prev_actions)
        prev_comms_out = self.comm_in(prev_comms_out)
                        
        for step in range(steps):
            (zp_mu, zp_std, new_hidden_states_p), (zq_mu, zq_std, new_hidden_states_q), dkls = \
                self.bottom_to_top_step(prev_hidden_states, obs[:,step], prev_actions[:,step], prev_comms_out[:,step])
            
            for l, o in zip(
                [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list_p, new_hidden_states_list_q],
                [zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p, new_hidden_states_q]):     
                l.append(o)
                
            prev_hidden_states = new_hidden_states_q.squeeze(1)
            
        lists = [zp_mu_list, zp_std_list, zq_mu_list, zq_std_list, new_hidden_states_list_p, new_hidden_states_list_q]
        for i in range(len(lists)):
            lists[i] = torch.cat(lists[i], dim=1)
        zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p, new_hidden_states_q = lists
        
        pred_rgbd, pred_comms, pred_sensors = self.predict(new_hidden_states_q[:, :-1, 0], prev_actions[:, 1:])
        
        return(
            (zp_mu, zp_std, new_hidden_states_p),
            (zq_mu, zq_std, new_hidden_states_q),
            (pred_rgbd, pred_comms, pred_sensors),
            dkls)
        
        
        
if __name__ == "__main__":
    
    args.layers = 1
    args.time_scales = [1]
    
    pvrnn = PVRNN(args = args)
    
    print("\n\nPVRNN: ONE LAYER")
    print(pvrnn)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(pvrnn, 
                                ((episodes, args.layers, args.pvrnn_mtrnn_size), 
                                (episodes, steps+1, args.image_size, args.image_size, 4), 
                                (episodes, steps+1, args.max_comm_len, args.comm_shape),
                                (episodes, steps+1, args.sensors_shape),
                                (episodes, steps+1, args.action_shape),
                                (episodes, steps+1, args.max_comm_len, args.comm_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

    """
    args.layers = 5
    args.time_scales = [1, 1, 1, 1, 1]
    
    pvrnn = PVRNN(args = args)
    
    print("\n\nPVRNN: MANY LAYERS")
    print(pvrnn)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(pvrnn, 
                                ((episodes, args.layers, args.pvrnn_mtrnn_size), 
                                (episodes, steps+1, args.image_size, args.image_size * 4, 4), 
                                (episodes, steps+1, args.max_comm_len, args.comm_shape),
                                (episodes, steps+1, args.sensors_shape),
                                (episodes, steps+1, args.action_shape),
                                (episodes, steps+1, args.max_comm_len, args.comm_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    """

            

# %%