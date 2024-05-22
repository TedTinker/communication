#%%
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils import default_args, attach_list, detach_list, dkl, duration
from submodule_utils import init_weights, episodes_steps, pad_zeros, var, sample, model_start, model_end
from mtrnn import MTRNN
from submodules import RGBD_IN, Comm_IN, Sensors_IN, Obs_IN, Action_IN, RGBD_OUT, Comm_OUT, Sensors_OUT



if __name__ == "__main__":
    
    args = default_args
    episodes = args.batch_size ; steps = args.max_steps
    
    


# Not yet implemented, but should make separate curiosities easier.
class ZP_ZQ(nn.Module):
    
    def __init__(self, zp_in_features, zq_in_features, out_features, args):
        super(ZP_ZQ, self).__init__()
        
        self.args = args 
            
        self.zp_mu = nn.Sequential(
                nn.Linear(
                    in_features = zp_in_features, 
                    out_features = out_features),
                nn.Tanh())
        self.zp_std = nn.Sequential(
                nn.Linear(
                    in_features = zp_in_features, 
                    out_features = out_features),
                nn.Softplus())
                            
        self.zq_mu = nn.Sequential(
                nn.Linear(
                    in_features = zq_in_features, 
                    out_features = out_features),
                nn.Tanh())
        self.zq_std = nn.Sequential(
                nn.Linear(
                    in_features = zq_in_features, 
                    out_features = out_features),
                nn.Softplus())
            
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
        
    def forward(self, zp_inputs, zq_inputs):                                    
        if(self.args.half):
            zp_inputs = zp_inputs.to(dtype=torch.float16)
            zq_inputs = zq_inputs.to(dtype=torch.float16)
        
        zp_mu, zp_std = var(zp_inputs, self.zp_mu, self.zp_std, self.args)
        zp = sample(zp_mu, zp_std, self.args.device)
        zq_mu, zq_std = var(zq_inputs, self.zq_mu, self.zq_std, self.args)
        zq = sample(zq_mu, zq_std, self.args.device)
        kullback_leibler = dkl(zp_mu, zp_std, zq_mu, zq_std)
                            
        return(
            (zp, zp_mu, zp_std),
            (zq, zq_mu, zq_std),
            kullback_leibler)       



class PVRNN_LAYER(nn.Module):
    
    def __init__(self, 
                 time_scale = 1, bottom = False, top = False, 
                 obs_size = 0, state_size = 0, in_hidden_size = 0, out_hidden_size = 0,
                 below_hidden_size = 0, above_hidden_size = 0, args = default_args):
        super(PVRNN_LAYER, self).__init__()
        
        self.args = args 
        self.bottom = bottom
        self.top = top
            
        
        self.z = ZP_ZQ(
            # Prior: Previous hidden state, plus action if bottom.  
            zp_in_features = in_hidden_size + \
                (self.args.encode_action_size + self.args.encode_comm_size if self.bottom else 0),
            # Posterior: Previous hidden state, plus observation and action if bottom, plus lower-layer hidden state otherwise.
            zq_in_features = in_hidden_size + \
                (self.args.encode_action_size + self.args.encode_comm_size if self.bottom else below_hidden_size) + obs_size, 
            out_features = state_size, args = self.args)
                            
        # New hidden state: Previous hidden state, zq value, plus higher-layer hidden state if not top.
        self.mtrnn = MTRNN(
                input_size = state_size + above_hidden_size,
                hidden_size = in_hidden_size, 
                time_constant = time_scale,
                args = self.args)
        
        self.hidden_resize = nn.Sequential(
            nn.Linear(
                in_features = in_hidden_size,
                out_features = out_hidden_size),
            nn.PReLU())
            
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
        
    def forward(
        self, 
        prev_hidden_states, 
        obs = None, prev_actions = None, prev_comms_out = None, 
        hidden_states_below = None, 
        prev_hidden_states_above = None):
                        
        prev_hidden_states = prev_hidden_states.to(self.args.device)
                        
        if(self.bottom):
            zp_inputs = torch.cat([prev_hidden_states, prev_actions, prev_comms_out], dim = -1)
            zq_inputs = torch.cat([prev_hidden_states, prev_actions, prev_comms_out, obs], dim = -1)
        else:
            zp_inputs = prev_hidden_states 
            zq_inputs = torch.cat([prev_hidden_states, hidden_states_below, obs], dim = -1)
                                                
        episodes, steps = episodes_steps(zp_inputs)
        zp_inputs = zp_inputs.reshape(episodes * steps, zp_inputs.shape[2])
        zq_inputs = zq_inputs.reshape(episodes * steps, zq_inputs.shape[2])
        if(len(prev_hidden_states_above.shape) == 1):
            prev_hidden_states_above = None
        if(prev_hidden_states_above != None):
            prev_hidden_states_above = prev_hidden_states_above.reshape(episodes * steps, prev_hidden_states_above.shape[2])
                                                
        if(self.args.half):
            zp_inputs = zp_inputs.to(dtype=torch.float16)
            zq_inputs = zq_inputs.to(dtype=torch.float16)
                
        (zp, zp_mu, zp_std), (zq, zq_mu, zq_std), kullback_leibler = self.z(zp_inputs, zq_inputs)
                                
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
        new_hidden_states_p = self.hidden_resize(new_hidden_states_p)
        new_hidden_states_q = self.mtrnn(mtrnn_inputs_q, prev_hidden_states)
        new_hidden_states_q = self.hidden_resize(new_hidden_states_q)
                        
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
    
    bottom_top_layer = PVRNN_LAYER(
        time_scale = 1, bottom = True, top = True, 
        obs_size = args.encode_rgbd_size, 
        state_size = args.rgbd_state_size, 
        in_hidden_size = args.rgbd_hidden_size + args.comm_hidden_size + args.sensors_hidden_size, 
        out_hidden_size = args.rgbd_hidden_size,
        below_hidden_size = 0,
        above_hidden_size = 0,
        args = args)
    
    print("\n\nBOTTOM-TOP")
    print(bottom_top_layer)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(bottom_top_layer, 
                                ((episodes, 1, args.rgbd_hidden_size + args.comm_hidden_size + args.sensors_hidden_size), 
                                (episodes, 1, args.encode_rgbd_size),
                                (episodes, 1, args.encode_action_size),
                                (episodes, 1, args.encode_comm_size),
                                (1,), #hidden_states_below = None, 
                                (1,)))) #prev_hidden_states_above = None)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        
    
    
    bottom_layer = PVRNN_LAYER(
        time_scale = 1, bottom = True, top = False, 
        obs_size = args.encode_rgbd_size, 
        state_size = args.rgbd_state_size, 
        in_hidden_size = args.rgbd_hidden_size + args.comm_hidden_size + args.sensors_hidden_size, 
        out_hidden_size = args.rgbd_hidden_size,
        below_hidden_size = 0,
        above_hidden_size = args.pvrnn_mtrnn_size,
        args = args)
    
    print("\n\nBOTTOM")
    print(bottom_layer)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(bottom_layer, 
                                ((episodes, 1, args.rgbd_hidden_size + args.comm_hidden_size + args.sensors_hidden_size), 
                                (episodes, 1, args.encode_rgbd_size),
                                (episodes, 1, args.encode_action_size),
                                (episodes, 1, args.encode_comm_size),
                                (1,), #hidden_states_below = None, 
                                (episodes, 1, args.pvrnn_mtrnn_size)))) # prev_hidden_states_above = None)))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        
    
    
    top_layer = PVRNN_LAYER(
        time_scale = 1, bottom = False, top = True, 
        obs_size = args.encode_rgbd_size, 
        state_size = args.rgbd_state_size, 
        in_hidden_size = args.rgbd_hidden_size + args.comm_hidden_size + args.sensors_hidden_size, 
        out_hidden_size = args.rgbd_hidden_size,
        below_hidden_size = args.pvrnn_mtrnn_size,
        above_hidden_size = 0,
        args = args)
    
    print("\n\nTop")
    print(top_layer)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(top_layer, 
                                ((episodes, 1, args.rgbd_hidden_size + args.comm_hidden_size + args.sensors_hidden_size), 
                                (episodes, 1, args.encode_rgbd_size),
                                (episodes, 1, args.encode_action_size),
                                (episodes, 1, args.encode_comm_size),
                                (episodes, 1, args.pvrnn_mtrnn_size), #hidden_states_below = None, 
                                (1,)))) # prev_hidden_states_above = None)))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
    middle_layer = PVRNN_LAYER(
        time_scale = 1, bottom = False, top = False, 
        obs_size = args.encode_rgbd_size, 
        state_size = args.rgbd_state_size, 
        in_hidden_size = args.rgbd_hidden_size + args.comm_hidden_size + args.sensors_hidden_size, 
        out_hidden_size = args.rgbd_hidden_size,
        below_hidden_size = args.pvrnn_mtrnn_size,
        above_hidden_size = args.pvrnn_mtrnn_size,
        args = args)
    
    print("\n\nMiddle")
    print(middle_layer)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(middle_layer, 
                                ((episodes, 1, args.rgbd_hidden_size + args.comm_hidden_size + args.sensors_hidden_size), 
                                (episodes, 1, args.encode_rgbd_size),
                                (episodes, 1, args.encode_action_size),
                                (episodes, 1, args.encode_comm_size),
                                (episodes, 1, args.pvrnn_mtrnn_size), #hidden_states_below = None, 
                                (episodes, 1, args.pvrnn_mtrnn_size)))) # prev_hidden_states_above = None)))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        
    
#%%
    
class PVRNN(nn.Module):
    
    def __init__(self, args = default_args):
        super(PVRNN, self).__init__()
        
        self.args = args 
        
        self.obs_in = Obs_IN(self.args)
        self.action_in = Action_IN(self.args)
        self.comm_out_in = Comm_IN(self.args)
        
        self.rgbd_layer = PVRNN_LAYER(
            time_scale = 1, bottom = True, top = True if self.args.layers == 0 else False, 
            obs_size = self.args.encode_rgbd_size + self.args.encode_comm_size + self.args.encode_sensors_size, 
            state_size = self.args.rgbd_state_size, 
            in_hidden_size = self.args.rgbd_hidden_size + self.args.comm_hidden_size + self.args.sensors_hidden_size, 
            out_hidden_size = self.args.rgbd_hidden_size,
            below_hidden_size = 0,
            above_hidden_size = 0 if self.args.layers == 0 else self.args.pvrnn_mtrnn_size,
            args = self.args)
        
        self.comm_layer = PVRNN_LAYER(
            time_scale = 1, bottom = True, top = True if self.args.layers == 0 else False, 
            obs_size = self.args.encode_rgbd_size + self.args.encode_comm_size + self.args.encode_sensors_size, 
            state_size = self.args.comm_state_size, 
            in_hidden_size = self.args.rgbd_hidden_size + self.args.comm_hidden_size + self.args.sensors_hidden_size, 
            out_hidden_size = self.args.comm_hidden_size,
            below_hidden_size = 0,
            above_hidden_size = 0 if self.args.layers == 0 else self.args.pvrnn_mtrnn_size,
            args = self.args)
        
        self.sensors_layer = PVRNN_LAYER(
            time_scale = 1, bottom = True, top = True if self.args.layers == 0 else False, 
            obs_size = self.args.encode_rgbd_size + self.args.encode_comm_size + self.args.encode_sensors_size, 
            state_size = self.args.sensors_state_size, 
            in_hidden_size = self.args.rgbd_hidden_size + self.args.comm_hidden_size + self.args.sensors_hidden_size, 
            out_hidden_size = self.args.sensors_hidden_size,
            below_hidden_size = 0,
            above_hidden_size = 0 if self.args.layers == 0 else self.args.pvrnn_mtrnn_size,
            args = self.args)
        
        pvrnn_layers = []
        for layer in range(self.args.layers - 1): 
            pvrnn_layers.append(
                PVRNN_LAYER(
                    time_scale = 1, bottom = False, top = layer == self.args.layers, 
                    obs_size = 0, #args.encode_sensor_size, 
                    state_size = self.args.state_size, 
                    in_hidden_size = self.args.pvrnn_mtrnn_size, 
                    out_hidden_size = self.args.pvrnn_mtrnn_size,
                    below_hidden_size = 0, #0,
                    above_hidden_size = self.args.pvrnn_mtrnn_size,
                    args = self.args))
            
        self.pvrnn_layers = nn.ModuleList(pvrnn_layers)
        self.predict_rgbd = RGBD_OUT(self.args)
        self.predict_comm = Comm_OUT(actor = False, args = self.args)
        self.predict_sensors = Sensors_OUT(self.args)
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
        
    def predict(self, rgbd_hidden, comm_hidden, sensors_hidden, action):
        h_w_actions = torch.cat([rgbd_hidden, action], dim = -1)
        pred_rgbd = self.predict_rgbd(h_w_actions)
        h_w_actions = torch.cat([comm_hidden, action], dim = -1)
        pred_comms = self.predict_comm(h_w_actions)
        h_w_actions = torch.cat([sensors_hidden, action], dim = -1)
        pred_sensors = self.predict_sensors(h_w_actions)
        return(pred_rgbd, pred_comms, pred_sensors)
        
    def bottom_to_top_step(self, 
            prev_hidden_states, prev_obs_hidden_states, obs = None, 
            prev_actions = None, prev_comms_out = None):
        if(obs != None and len(obs.shape) == 2): 
            obs = obs.unsqueeze(1)
        if(prev_actions != None and len(prev_actions.shape) == 2): 
            prev_actions = prev_actions.unsqueeze(1)
        if(prev_comms_out != None and len(prev_comms_out.shape) == 2): 
            prev_comms_out = prev_comms_out.unsqueeze(1)
        
        #prev_hidden_states, 
        #obs = None, prev_actions = None, prev_comms_out = None, 
        #hidden_states_below = (1,), 
        #prev_hidden_states_above = (1,)
                        
        (zp_mu_rgbd, zp_std_rgbd, new_hidden_states_p_rgbd), (zq_mu_rgbd, zq_std_rgbd, new_hidden_states_q_rgbd), dkl_rgbd = \
            self.rgbd_layer(
                prev_hidden_states = prev_obs_hidden_states, 
                obs = obs, prev_actions = prev_actions, prev_comms_out = prev_comms_out,
                hidden_states_below = None,
                prev_hidden_states_above = prev_hidden_states[:,0].unsqueeze(1) if self.args.layers > 0 else torch.tensor((1,)))
                                                                       
        (zp_mu_comm, zp_std_comm, new_hidden_states_p_comm), (zq_mu_comm, zq_std_comm, new_hidden_states_q_comm), dkl_comm = \
            self.comm_layer(
                prev_hidden_states = prev_obs_hidden_states, 
                obs = obs, prev_actions = prev_actions, prev_comms_out = prev_comms_out,
                hidden_states_below = None,
                prev_hidden_states_above = prev_hidden_states[:,0].unsqueeze(1) if self.args.layers > 0 else torch.tensor((1,)))
                                                      
        (zp_mu_sensors, zp_std_sensors, new_hidden_states_p_sensors), (zq_mu_sensors, zq_std_sensors, new_hidden_states_q_sensors), dkl_sensors = \
            self.sensors_layer(
                prev_hidden_states = prev_obs_hidden_states, 
                obs = obs, prev_actions = prev_actions, prev_comms_out = prev_comms_out,
                hidden_states_below = None,
                prev_hidden_states_above = prev_hidden_states[:,0].unsqueeze(1) if self.args.layers > 0 else torch.tensor((1,)))
                                                            
        if(self.args.layers > 0):     
            zp_mu_list = []
            zp_std_list = []
            zq_mu_list = []
            zq_std_list = []
            new_hidden_states_list_p = []
            new_hidden_states_list_q = []
            dkls = []        
            # Gotta replace obs.                                                        
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
                lists[i] = torch.cat(lists[i], dim=1).unsqueeze(1)
            zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p, new_hidden_states_q, dkls = lists
        else:
            zp_mu, zp_std, zq_mu, zq_std, new_hidden_states_p, new_hidden_states_q, dkls = [], [], [], [], [], [], []
                                                            
        return(
            (zp_mu_rgbd, zp_std_rgbd, new_hidden_states_p_rgbd), 
            (zq_mu_rgbd, zq_std_rgbd, new_hidden_states_q_rgbd), 
            dkl_rgbd,
            (zp_mu_comm, zp_std_comm, new_hidden_states_p_comm), 
            (zq_mu_comm, zq_std_comm, new_hidden_states_q_comm), 
            dkl_comm,
            (zp_mu_sensors, zp_std_sensors, new_hidden_states_p_sensors), 
            (zq_mu_sensors, zq_std_sensors, new_hidden_states_q_sensors), 
            dkl_sensors,
            (zp_mu, zp_std, new_hidden_states_p),
            (zq_mu, zq_std, new_hidden_states_q),
            dkls)
    
    def forward(self, 
                prev_hidden_states, prev_obs_hidden_states,
                rgbd, comms_in, sensors, prev_actions, prev_comms_out):
        
        zp_mu_rgbd_list = []
        zp_std_rgbd_list = []
        zq_mu_rgbd_list = []
        zq_std_rgbd_list = []
        new_hidden_states_list_p_rgbd = []
        new_hidden_states_list_q_rgbd = []
        dkls_rgbd_list = []
        
        zp_mu_comm_list = []
        zp_std_comm_list = []
        zq_mu_comm_list = []
        zq_std_comm_list = []
        new_hidden_states_list_p_comm = []
        new_hidden_states_list_q_comm = []
        dkls_comm_list = []
        
        zp_mu_sensors_list = []
        zp_std_sensors_list = []
        zq_mu_sensors_list = []
        zq_std_sensors_list = []
        new_hidden_states_list_p_sensors = []
        new_hidden_states_list_q_sensors = []
        dkls_sensors_list = []
        
        zp_mu_list = []
        zp_std_list = []
        zq_mu_list = []
        zq_std_list = []
        new_hidden_states_list_p = []
        new_hidden_states_list_q = []
        dkls_list = []
        
        prev_time = duration()
                
        episodes, steps = episodes_steps(rgbd)
        if(prev_hidden_states == None):
            prev_hidden_states = torch.zeros((episodes, self.args.layers, self.args.pvrnn_mtrnn_size))
            
        obs = self.obs_in(rgbd, comms_in, sensors)
        
        prev_actions = self.action_in(prev_actions)
        prev_comms_out = self.comm_out_in(prev_comms_out)
                                
        for step in range(steps):
            (zp_mu_rgbd, zp_std_rgbd, new_hidden_states_p_rgbd), \
            (zq_mu_rgbd, zq_std_rgbd, new_hidden_states_q_rgbd), \
            dkl_rgbd, \
            (zp_mu_comm, zp_std_comm, new_hidden_states_p_comm), \
            (zq_mu_comm, zq_std_comm, new_hidden_states_q_comm), \
            dkl_comm, \
            (zp_mu_sensors, zp_std_sensors, new_hidden_states_p_sensors), \
            (zq_mu_sensors, zq_std_sensors, new_hidden_states_q_sensors), \
            dkl_sensors, \
            (zp_mu, zp_std, new_hidden_states_p), \
            (zq_mu, zq_std, new_hidden_states_q), \
            dkls = \
                self.bottom_to_top_step(prev_hidden_states, prev_obs_hidden_states, obs[:,step], 
                                        prev_actions[:,step], prev_comms_out[:,step])
                            
            for l, o in zip(
                [zp_mu_rgbd_list,       zp_std_rgbd_list,       zq_mu_rgbd_list,    zq_std_rgbd_list,       new_hidden_states_list_p_rgbd,      new_hidden_states_list_q_rgbd,      dkls_rgbd_list,
                 zp_mu_comm_list,       zp_std_comm_list,       zq_mu_comm_list,    zq_std_comm_list,       new_hidden_states_list_p_comm,      new_hidden_states_list_q_comm,      dkls_comm_list,
                 zp_mu_sensors_list,    zp_std_sensors_list,    zq_mu_sensors_list, zq_std_sensors_list,    new_hidden_states_list_p_sensors,   new_hidden_states_list_q_sensors,   dkls_sensors_list,
                 zp_mu_list,            zp_std_list,            zq_mu_list,         zq_std_list,            new_hidden_states_list_p,           new_hidden_states_list_q,           dkls_list],
                [zp_mu_rgbd,            zp_std_rgbd,            zq_mu_rgbd,         zq_std_rgbd,            new_hidden_states_p_rgbd,           new_hidden_states_q_rgbd,           dkl_rgbd,
                 zp_mu_comm,            zp_std_comm,            zq_mu_comm,         zq_std_comm,            new_hidden_states_p_comm,           new_hidden_states_q_comm,           dkl_comm,
                 zp_mu_sensors,         zp_std_sensors,         zq_mu_sensors,      zq_std_sensors,         new_hidden_states_p_sensors,        new_hidden_states_q_sensors,        dkl_sensors,
                 zp_mu,                 zp_std,                 zq_mu,              zq_std,                 new_hidden_states_p,                new_hidden_states_q,                dkls]):     
                l.append(o)
                                
            if(self.args.layers > 0):
                prev_hidden_states = new_hidden_states_q.squeeze(1)
            prev_obs_hidden_states = torch.cat([new_hidden_states_q_rgbd, new_hidden_states_q_comm, new_hidden_states_q_sensors], dim = -1)
            
        lists = \
            [   zp_mu_rgbd_list,       zp_std_rgbd_list,       zq_mu_rgbd_list,        zq_std_rgbd_list,       new_hidden_states_list_p_rgbd,      new_hidden_states_list_q_rgbd,       dkls_rgbd_list,
                zp_mu_comm_list,       zp_std_comm_list,       zq_mu_comm_list,        zq_std_comm_list,       new_hidden_states_list_p_comm,      new_hidden_states_list_q_comm,       dkls_comm_list,
                zp_mu_sensors_list,    zp_std_sensors_list,    zq_mu_sensors_list,     zq_std_sensors_list,    new_hidden_states_list_p_sensors,   new_hidden_states_list_q_sensors,    dkls_sensors_list,]
        for i in range(len(lists)):
            lists[i] = torch.cat(lists[i], dim=1)
        zp_mu_rgbd,                 zp_std_rgbd,            zq_mu_rgbd,             zq_std_rgbd,            new_hidden_states_p_rgbd,           new_hidden_states_q_rgbd,       dkl_rgbd,\
        zp_mu_comm,                 zp_std_comm,            zq_mu_comm,             zq_std_comm,            new_hidden_states_p_comm,           new_hidden_states_q_comm,       dkl_comm,\
        zp_mu_sensors,              zp_std_sensors,         zq_mu_sensors,          zq_std_sensors,         new_hidden_states_p_sensors,        new_hidden_states_q_sensors,    dkl_sensors = lists
            
        if(self.args.layers > 0):
            lists = \
                [zp_mu_list,            zp_std_list,            zq_mu_list,             zq_std_list,            new_hidden_states_list_p,           new_hidden_states_list_q,   dkls_list]
            for i in range(len(lists)):
                lists[i] = torch.cat(lists[i], dim=1)
            zp_mu,                      zp_std,                 zq_mu,                  zq_std,                 new_hidden_states_p,                new_hidden_states_q,        dkls = lists
        
        pred_rgbd, pred_comms, pred_sensors = self.predict(
            new_hidden_states_q_rgbd[:, :-1], 
            new_hidden_states_q_comm[:, :-1],
            new_hidden_states_q_sensors[:, :-1],
            prev_actions[:, 1:])
        
        return(
            (pred_rgbd, pred_comms, pred_sensors),
            (zp_mu_rgbd, zp_std_rgbd, new_hidden_states_p_rgbd),
            (zq_mu, zq_std, new_hidden_states_q_rgbd),
            dkl_rgbd,
            (zp_mu, zp_std, new_hidden_states_p),
            (zq_mu, zq_std, new_hidden_states_q),
            dkl_comm,
            (zp_mu, zp_std, new_hidden_states_p),
            (zq_mu, zq_std, new_hidden_states_q),
            dkl_sensors,
            (zp_mu, zp_std, new_hidden_states_p),
            (zq_mu, zq_std, new_hidden_states_q),
            dkls)
        
        
        
if __name__ == "__main__":
    
    args.layers = 0
    args.time_scales = [1]
    
    pvrnn = PVRNN(args = args)
    
    print("\n\nPVRNN: ONE LAYER")
    print(pvrnn)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(pvrnn, 
                                ((episodes, args.layers, args.pvrnn_mtrnn_size) if args.layers > 0 else (1,), 
                                (episodes, 1, args.rgbd_hidden_size + args.comm_hidden_size + args.sensors_hidden_size), 
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