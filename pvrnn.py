#%%
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils import default_args, calculate_dkl, duration, Obs, Inner_States, Action
from utils_submodule import init_weights, episodes_steps, var, sample, model_start
from mtrnn import MTRNN
from submodules import RGBD_IN, Sensors_IN, Voice_IN, Obs_OUT, Wheels_Shoulders_IN



if __name__ == "__main__":
    
    args = default_args
    episodes = args.batch_size ; steps = args.max_steps
    
    

class ZP_ZQ(nn.Module):
    
    def __init__(self, zp_in_features, zq_in_features, out_features, args):
        super(ZP_ZQ, self).__init__()
        
        self.args = args 
            
        # Prior: Previous hidden state and action.  
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
                            
        # Posterior: Include observation.
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
        dkl = calculate_dkl(zp_mu, zp_std, zq_mu, zq_std)
                            
        return(Inner_States(zp, zq, dkl))       



class PVRNN_LAYER(nn.Module):
    
    def __init__(self, time_scale = 1, args = default_args):
        super(PVRNN_LAYER, self).__init__()
        
        self.args = args 
            
        # Prior: Previous hidden state and wheels_shoulders.  
        # Posterior: Include observation.
        self.rgbd_z = ZP_ZQ(
            zp_in_features = self.args.h_w_action_size,
            zq_in_features = self.args.h_w_action_size + self.args.rgbd_encode_size, 
            out_features = self.args.rgbd_state_size, args = self.args)
        
        self.sensors_z = ZP_ZQ(
            zp_in_features = self.args.h_w_action_size,
            zq_in_features = self.args.h_w_action_size + self.args.sensors_encode_size, 
            out_features = self.args.sensors_state_size, args = self.args)
        
        self.father_voice_z = ZP_ZQ(
            zp_in_features = self.args.h_w_action_size,
            zq_in_features = self.args.h_w_action_size + self.args.voice_encode_size, 
            out_features = self.args.voice_state_size, args = self.args)
        
        self.mother_voice_z = ZP_ZQ(
            zp_in_features = self.args.h_w_action_size,
            zq_in_features = self.args.h_w_action_size + self.args.voice_encode_size, 
            out_features = self.args.voice_state_size, args = self.args)
    
        # New hidden state: Previous hidden state, zq value, plus higher-layer hidden state if not top.
        self.mtrnn = MTRNN(
                input_size = self.args.rgbd_state_size + self.args.sensors_state_size + self.args.voice_state_size + self.args.voice_state_size,
                hidden_size = self.args.pvrnn_mtrnn_size, 
                time_constant = time_scale,
                args = self.args)
            
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
            
    def forward(self, prev_hidden_states, obs, prev_action):
        
        def reshape_and_to_dtype(inputs, episodes, steps, dtype=None):
            inputs = inputs.reshape(episodes * steps, inputs.shape[2])
            if dtype:
                inputs = inputs.to(dtype=dtype)
            return inputs
        
        def process_z_func_outputs(zp_inputs, zq_inputs, z_func, episodes, steps, dtype=None):
            zp_inputs = reshape_and_to_dtype(zp_inputs, episodes, steps, dtype)
            zq_inputs = reshape_and_to_dtype(zq_inputs, episodes, steps, dtype)
            inner_states = z_func(zp_inputs, zq_inputs)
            inner_states.dkl = inner_states.dkl.reshape((episodes, steps, inner_states.dkl.shape[1]))
            return(inner_states)
        
        prev_hidden_states = prev_hidden_states.to(self.args.device)
        zp_inputs = torch.cat([prev_hidden_states, prev_action.wheels_shoulders, prev_action.voice_out], dim=-1)
        rgbd_zq_inputs, sensors_zq_inputs, father_voice_zq_inputs, mother_voice_zq_inputs = [torch.cat([zp_inputs, input_data], dim=-1) for input_data in (obs.rgbd, obs.sensors, obs.father_voice, obs.mother_voice)]
        
        episodes, steps = episodes_steps(zp_inputs)
        dtype = torch.float16 if self.args.half else None
        
        rgbd_is = process_z_func_outputs(zp_inputs, rgbd_zq_inputs, self.rgbd_z, episodes, steps, dtype)
        sensors_is = process_z_func_outputs(zp_inputs, sensors_zq_inputs, self.sensors_z, episodes, steps, dtype)
        father_voice_is = process_z_func_outputs(zp_inputs, father_voice_zq_inputs, self.father_voice_z, episodes, steps, dtype)
        mother_voice_is = process_z_func_outputs(zp_inputs, mother_voice_zq_inputs, self.mother_voice_z, episodes, steps, dtype)
        
        mtrnn_inputs_p = torch.cat([rgbd_is.zp, sensors_is.zp, father_voice_is.zp, mother_voice_is.zp], dim=-1)
        mtrnn_inputs_q = torch.cat([rgbd_is.zq, sensors_is.zq, father_voice_is.zq, mother_voice_is.zq], dim=-1)
        
        mtrnn_inputs_p = mtrnn_inputs_p.reshape(episodes, steps, mtrnn_inputs_p.shape[1])
        mtrnn_inputs_q = mtrnn_inputs_q.reshape(episodes, steps, mtrnn_inputs_q.shape[1])
        
        new_hidden_states_p = self.mtrnn(mtrnn_inputs_p, prev_hidden_states)
        new_hidden_states_q = self.mtrnn(mtrnn_inputs_q, prev_hidden_states)
        
        return(new_hidden_states_p, new_hidden_states_q, rgbd_is, sensors_is, father_voice_is, mother_voice_is)
        
        
    
if __name__ == "__main__":
    
    pvrnn_layer = PVRNN_LAYER(time_scale = 1, args = args)
    
    print("\n\nPVRNN LAYER")
    print(pvrnn_layer)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(pvrnn_layer, 
                                ((episodes, 1, args.pvrnn_mtrnn_size), 
                                (episodes, 1, args.rgbd_encode_size),
                                (episodes, 1, args.sensors_encode_size),
                                (episodes, 1, args.voice_encode_size),
                                (episodes, 1, args.wheels_shoulders_encode_size),
                                (episodes, 1, args.voice_encode_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
#%%
    
    
    
class PVRNN(nn.Module):
    
    def __init__(self, args = default_args):
        super(PVRNN, self).__init__()
        
        self.args = args 
        
        self.rgbd_in = RGBD_IN(self.args)
        self.sensors_in = Sensors_IN(self.args)
        self.father_voice_in = Voice_IN(self.args) 
        self.mother_voice_in = Voice_IN(self.args) 
        self.self_voice_in = Voice_IN(self.args) 
        self.wheels_shoulders_in = Wheels_Shoulders_IN(self.args)

        self.pvrnn_layer = PVRNN_LAYER(1, args = self.args)
            
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
            
            
            
    def obs_in(self, obs):
        return(Obs(
            self.rgbd_in(obs.rgbd),
            self.sensors_in(obs.sensors),
            self.father_voice_in(obs.father_voice),
            self.father_voice_in(obs.mother_voice)))
        
        
        
    def action_in(self, action):
        return(Action(
            self.wheels_shoulders_in(action.wheels_shoulders), 
            self.father_voice_in(action.voice_out)))
    
    
            
    # I think voice_out should be here too.
    def predict(self, h, wheels_shoulders):
        h_w_wheels_shoulders = torch.cat([h, wheels_shoulders], dim = -1)
        pred_rgbd, pred_sensors, pred_father_voice, pred_mother_voice = self.predict_obs(h_w_wheels_shoulders)
        return(Obs(pred_rgbd, pred_sensors, pred_father_voice, pred_mother_voice))
    
    
    
        
    def bottom_to_top_step(self, prev_hidden_states, obs, prev_action):
        start_time = duration()
        prev_time = duration()
        
        start, episodes, steps, [prev_hidden_states, rgbd, sensors, father_voice, mother_voice, prev_wheels_shoulders, prev_voice_out] = model_start(
            [(prev_hidden_states, "lin"), (obs.rgbd, "lin"), (obs.sensors, "lin"), (obs.father_voice, "lin"), (obs.mother_voice, "lin"), 
             (prev_action.wheels_shoulders, "lin"), (prev_action.voice_out, "lin")], self.args.device, self.args.half, recurrent = True)
        
        new_hidden_states_p, new_hidden_states_q, rgbd_is, sensors_is, father_voice_is, mother_voice_is = \
            self.pvrnn_layer(
                prev_hidden_states[:,0].unsqueeze(1), 
                Obs(rgbd, sensors, father_voice, mother_voice), Action(prev_wheels_shoulders, prev_voice_out))       
            
        time = duration()
        #if(self.args.show_duration): print("BOTTOM TO TOP STEP:", time - prev_time)
        prev_time = time
                
        return(new_hidden_states_p, new_hidden_states_q, rgbd_is, sensors_is, father_voice_is, mother_voice_is)
    
    
    
    def forward(self, prev_hidden_states, obs, prev_action):
                        
        episodes, steps = episodes_steps(obs.rgbd)
        if(prev_hidden_states == None):
            prev_hidden_states = torch.zeros(episodes, 1, self.args.pvrnn_mtrnn_size)
        
        task_labels = torch.argmax(obs.father_voice[:, :, 0, :], dim=2)
        color_labels = torch.argmax(obs.father_voice[:, :, 1, :], dim=2)
        shape_labels = torch.argmax(obs.father_voice[:, :, 2, :], dim=2)
        labels = torch.stack((task_labels, color_labels, shape_labels), dim = -1)
        
        rgbd_is_list = []
        sensors_is_list = []
        father_voice_is_list = []
        mother_voice_is_list = []
        new_hidden_states_p_list = []
        new_hidden_states_q_list = []
        
        prev_time = duration()
                
        obs = self.obs_in(obs)
        prev_action = self.action_in(prev_action)
                                
        for step in range(steps):
            step_obs = Obs(obs.rgbd[:,step], obs.sensors[:,step], obs.father_voice[:,step], obs.mother_voice[:,step])
            step_action = Action(prev_action.wheels_shoulders[:,step], prev_action.voice_out[:,step])
            new_hidden_states_p, new_hidden_states_q, rgbd_is, sensors_is, father_voice_is, mother_voice_is = \
                self.bottom_to_top_step(prev_hidden_states, step_obs, step_action)
                                
            for l, o in zip(
                [new_hidden_states_p_list, new_hidden_states_q_list, rgbd_is_list, sensors_is_list, father_voice_is_list, mother_voice_is_list],
                [new_hidden_states_p, new_hidden_states_q, rgbd_is, sensors_is, father_voice_is, mother_voice_is]):     
                l.append(o)
                                
            prev_hidden_states = new_hidden_states_q
                        
        lists = [new_hidden_states_p_list, new_hidden_states_q_list, rgbd_is_list, sensors_is_list, father_voice_is_list, mother_voice_is_list]
        for i in range(len(lists)):
            if(isinstance(lists[i][0], torch.Tensor)):
                lists[i] = torch.cat(lists[i], dim=1)
            else:
                zp = torch.stack([inner_states.zp for inner_states in lists[i]], dim=1)
                zq = torch.stack([inner_states.zq for inner_states in lists[i]], dim=1)
                dkl = torch.cat([inner_states.dkl for inner_states in lists[i]], dim=1)
                lists[i] = Inner_States(zp, zq, dkl)
        new_hidden_states_p, new_hidden_states_q, rgbd_is, sensors_is, father_voice_is, mother_voice_is = lists
                
        pred_obs_p = self.predict(new_hidden_states_p[:, :-1], prev_action.wheels_shoulders[:, 1:])
        pred_obs_q = self.predict(new_hidden_states_q[:, :-1], prev_action.wheels_shoulders[:, 1:])
                
        task_labels = labels[:, :, 0].clone().unsqueeze(-1)
        color_labels = labels[:, :, 1].clone().unsqueeze(-1)
        shape_labels = labels[:, :, 2].clone().unsqueeze(-1)
        color_labels[color_labels != 0] = color_labels[color_labels != 0] - 5  # 6-11 -> 1-6
        shape_labels[shape_labels != 0] = shape_labels[shape_labels != 0] - 11  # 12-16 -> 1-5

        labels = torch.cat((task_labels, color_labels, shape_labels), dim=-1)
                        
        return(new_hidden_states_p, new_hidden_states_q, rgbd_is, sensors_is, father_voice_is, mother_voice_is, pred_obs_p, pred_obs_q, labels)
        
        
        
if __name__ == "__main__":
        
    pvrnn = PVRNN(args = args)
    
    print("\n\nPVRNN: ONE LAYER")
    print(pvrnn)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(pvrnn, 
                                ((episodes, 1, args.pvrnn_mtrnn_size), 
                                (episodes, steps+1, args.image_size, args.image_size, 4), 
                                (episodes, steps+1, args.sensors_shape),
                                (episodes, steps+1, args.max_voice_len, args.voice_shape),
                                (episodes, steps+1, args.wheels_shoulders_shape),
                                (episodes, steps+1, args.max_voice_len, args.voice_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

            

# %%