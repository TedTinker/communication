#%%
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils import default_args, dkl, duration, how_many_nans
from utils_submodule import init_weights, episodes_steps, var, sample
from mtrnn import MTRNN
from submodules import RGBD_IN, Sensors_IN, Comm_IN, Obs_OUT, Wheels_Shoulders_IN



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
        kullback_leibler = dkl(zp_mu, zp_std, zq_mu, zq_std)
                            
        return(zp, zq, kullback_leibler)       



class PVRNN_LAYER(nn.Module):
    
    def __init__(self, time_scale = 1, args = default_args):
        super(PVRNN_LAYER, self).__init__()
        
        self.args = args 
            
        # Prior: Previous hidden state and wheels_shoulders.  
        # Posterior: Include observation.
        self.rgbd_z = ZP_ZQ(
            zp_in_features = self.args.h_w_wheels_shoulders_size + self.args.comm_encode_size,
            zq_in_features = self.args.h_w_wheels_shoulders_size + self.args.comm_encode_size + self.args.rgbd_encode_size, 
            out_features = self.args.rgbd_state_size, args = self.args)
        
        self.sensors_z = ZP_ZQ(
            zp_in_features = self.args.h_w_wheels_shoulders_size + self.args.comm_encode_size,
            zq_in_features = self.args.h_w_wheels_shoulders_size + self.args.comm_encode_size + self.args.sensors_encode_size, 
            out_features = self.args.sensors_state_size, args = self.args)
        
        self.comm_z = ZP_ZQ(
            zp_in_features = self.args.h_w_wheels_shoulders_size + self.args.comm_encode_size,
            zq_in_features = self.args.h_w_wheels_shoulders_size + self.args.comm_encode_size + self.args.comm_encode_size, 
            out_features = self.args.comm_state_size, args = self.args)
    
                            
        # New hidden state: Previous hidden state, zq value, plus higher-layer hidden state if not top.
        self.mtrnn = MTRNN(
                input_size = self.args.rgbd_state_size + self.args.sensors_state_size + self.args.comm_state_size,
                hidden_size = self.args.pvrnn_mtrnn_size, 
                time_constant = time_scale,
                args = self.args)
            
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
            
    def forward(self, prev_hidden_states, rgbd=None, sensors=None, comm=None, prev_wheels_shoulders=None, prev_comm_out=None):
        def reshape_and_to_dtype(inputs, episodes, steps, dtype=None):
            inputs = inputs.reshape(episodes * steps, inputs.shape[2])
            if dtype:
                inputs = inputs.to(dtype=dtype)
            return inputs
        
        def process_z_func_outputs(zp_inputs, zq_inputs, z_func, episodes, steps, dtype=None):
            zp_inputs = reshape_and_to_dtype(zp_inputs, episodes, steps, dtype)
            zq_inputs = reshape_and_to_dtype(zq_inputs, episodes, steps, dtype)
            zp, zq, kullback_leibler = z_func(zp_inputs, zq_inputs)
            kullback_leibler = kullback_leibler.reshape((episodes, steps, kullback_leibler.shape[1]))
            return(zp, zq, kullback_leibler)
        
        """how_many_nans(prev_hidden_states, "PVRNN layer, prev_hidden_states")
        how_many_nans(rgbd, "PVRNN layer, rgbd")
        how_many_nans(comm, "PVRNN layer, comm")
        how_many_nans(sensors, "PVRNN layer, sensors")
        how_many_nans(prev_wheels_shoulders, "PVRNN layer, prev_wheels_shoulders")
        how_many_nans(prev_comm_out, "PVRNN layer, prev_comm_out")"""
        
        prev_hidden_states = prev_hidden_states.to(self.args.device)
        zp_inputs = torch.cat([prev_hidden_states, prev_wheels_shoulders, prev_comm_out], dim=-1)
        rgbd_zq_inputs, sensors_zq_inputs, comm_zq_inputs = [torch.cat([zp_inputs, input_data], dim=-1) for input_data in (rgbd, sensors, comm)]
        
        """how_many_nans(rgbd_zq_inputs, "PVRNN layer, rgbd_zq_inputs")
        how_many_nans(comm_zq_inputs, "PVRNN layer, comm_zq_inputs")
        how_many_nans(sensors_zq_inputs, "PVRNN layer, sensors_zq_inputs")"""
        
        episodes, steps = episodes_steps(zp_inputs)
        dtype = torch.float16 if self.args.half else None
        
        rgbd_zp, rgbd_zq, rgbd_dkl = process_z_func_outputs(zp_inputs, rgbd_zq_inputs, self.rgbd_z, episodes, steps, dtype)
        sensors_zp, sensors_zq, sensors_dkl = process_z_func_outputs(zp_inputs, sensors_zq_inputs, self.sensors_z, episodes, steps, dtype)
        comm_zp, comm_zq, comm_dkl = process_z_func_outputs(zp_inputs, comm_zq_inputs, self.comm_z, episodes, steps, dtype)
                
        """how_many_nans(rgbd_zp, "PVRNN layer, rgbd_zp")
        how_many_nans(rgbd_zq, "PVRNN layer, rgbd_zq")
        how_many_nans(rgbd_dkl, "PVRNN layer, rgbd_dkl")
        how_many_nans(comm_zp, "PVRNN layer, comm_zp")
        how_many_nans(comm_zq, "PVRNN layer, comm_zq")
        how_many_nans(comm_dkl, "PVRNN layer, comm_dkl")
        how_many_nans(sensors_zp, "PVRNN layer, sensors_zp")
        how_many_nans(sensors_zq, "PVRNN layer, sensors_zq")
        how_many_nans(sensors_dkl, "PVRNN layer, sensors_dkl")"""
        
        mtrnn_inputs_p = torch.cat([rgbd_zp, sensors_zp, comm_zp], dim=-1)
        mtrnn_inputs_q = torch.cat([rgbd_zq, sensors_zq, comm_zq], dim=-1)
        
        mtrnn_inputs_p = mtrnn_inputs_p.reshape(episodes, steps, mtrnn_inputs_p.shape[1])
        mtrnn_inputs_q = mtrnn_inputs_q.reshape(episodes, steps, mtrnn_inputs_q.shape[1])
        
        new_hidden_states_p = self.mtrnn(mtrnn_inputs_p, prev_hidden_states)
        new_hidden_states_q = self.mtrnn(mtrnn_inputs_q, prev_hidden_states)
        
        """how_many_nans(new_hidden_states_p, "PVRNN layer, new_hidden_states_p")
        how_many_nans(new_hidden_states_q, "PVRNN layer, new_hidden_states_q")
        how_many_nans(rgbd_dkl, "PVRNN layer, rgbd_dkl")
        how_many_nans(comm_dkl, "PVRNN layer, comm_dkl")
        how_many_nans(sensors_dkl, "PVRNN layer, sensors_dkl")"""
        
        return(new_hidden_states_p, new_hidden_states_q, rgbd_dkl, sensors_dkl, comm_dkl, comm_zq)
        
        
    
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
                                (episodes, 1, args.comm_encode_size),
                                (episodes, 1, args.wheels_shoulders_encode_size),
                                (episodes, 1, args.comm_encode_size))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
#%%
    
class PVRNN(nn.Module):
    
    def __init__(self, args = default_args):
        super(PVRNN, self).__init__()
        
        self.args = args 
        
        self.rgbd_in = RGBD_IN(self.args)
        self.sensors_in = Sensors_IN(self.args)
        self.comm_in = Comm_IN(self.args)
        self.wheels_shoulders_in = Wheels_Shoulders_IN(self.args)
        self.comm_out_in = Comm_IN(self.args)

        self.pvrnn_layer = PVRNN_LAYER(1, args = self.args)
            
        self.predict_obs = Obs_OUT(args)
        
        
        
        self.apply(init_weights)
        self.to(self.args.device)
        if(self.args.half):
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)
        
    def predict(self, h, wheels_shoulders):
        h_w_wheels_shoulders = torch.cat([h, wheels_shoulders], dim = -1)
        pred_rgbd, pred_sensors, pred_comm = self.predict_obs(h_w_wheels_shoulders)
        return(pred_rgbd, pred_sensors, pred_comm)
        
    def bottom_to_top_step(self, prev_hidden_states, rgbd = None, sensors = None, comm = None, prev_wheels_shoulders = None, prev_comm_out = None):
        start_time = duration()
        prev_time = duration()
        
        if(prev_hidden_states != None and len(prev_hidden_states.shape) == 2): 
            prev_hidden_states = prev_hidden_states.unsqueeze(1)
        if(rgbd != None and len(rgbd.shape) == 2): 
            rgbd = rgbd.unsqueeze(1)
        if(sensors != None and len(sensors.shape) == 2): 
            sensors = sensors.unsqueeze(1)
        if(comm != None and len(comm.shape) == 2): 
            comm = comm.unsqueeze(1)
        if(prev_wheels_shoulders != None and len(prev_wheels_shoulders.shape) == 2): 
            prev_wheels_shoulders = prev_wheels_shoulders.unsqueeze(1)
        if(prev_comm_out != None and len(prev_comm_out.shape) == 2): 
            prev_comm_out = prev_comm_out.unsqueeze(1)
                                    
        new_hidden_states_p, new_hidden_states_q, rgbd_dkl, sensors_dkl, comm_dkl, comm_zq = \
            self.pvrnn_layer(
                prev_hidden_states[:,0].unsqueeze(1), 
                rgbd, sensors, comm, prev_wheels_shoulders, prev_comm_out)
            
        time = duration()
        if(self.args.show_duration): print("BOTTOM TO TOP STEP:", time - prev_time)
        prev_time = time
                
        return(new_hidden_states_p, new_hidden_states_q, rgbd_dkl, sensors_dkl, comm_dkl, comm_zq)
    
    def forward(self, prev_hidden_states, rgbd, sensors, comm_in, prev_wheels_shoulders, prev_comm_out):
                
        task_labels = torch.argmax(comm_in[:, :, 0, :], dim=2)
        color_labels = torch.argmax(comm_in[:, :, 1, :], dim=2)
        shape_labels = torch.argmax(comm_in[:, :, 2, :], dim=2)
        labels = torch.stack((task_labels, color_labels, shape_labels), dim = -1)
        
        rgbd_dkl_list = []
        sensors_dkl_list = []
        comm_dkl_list = []
        new_hidden_states_p_list = []
        new_hidden_states_q_list = []
        comm_zq_list = []
        
        """how_many_nans(prev_hidden_states, "PVRNN, prev_hidden_states")
        how_many_nans(rgbd, "PVRNN, rgbd 1")
        how_many_nans(comm_in, "PVRNN, comm_in 1")
        how_many_nans(sensors, "PVRNN, sensors 1")
        how_many_nans(prev_wheels_shoulders, "PVRNN, prev_wheels_shoulders 1")
        how_many_nans(prev_comm_out, "PVRNN, prev_comm_out 1")"""
        
        prev_time = duration()
                
        episodes, steps = episodes_steps(rgbd)
        if(prev_hidden_states == None):
            prev_hidden_states = torch.zeros(episodes, 1, self.args.pvrnn_mtrnn_size)
        rgbd = self.rgbd_in(rgbd)
        sensors = self.sensors_in(sensors)
        comm_in = self.comm_in(comm_in)
        
        prev_wheels_shoulders = self.wheels_shoulders_in(prev_wheels_shoulders)
        prev_comm_out = self.comm_out_in(prev_comm_out)
        
        """how_many_nans(rgbd, "PVRNN, rgbd 2")
        how_many_nans(comm_in, "PVRNN, comm_in 2")
        how_many_nans(sensors, "PVRNN, sensors 2")
        how_many_nans(prev_wheels_shoulders, "PVRNN, prev_wheels_shoulders 2")
        how_many_nans(prev_comm_out, "PVRNN, prev_comm_out 2")"""
                                
        for step in range(steps):
            new_hidden_states_p, new_hidden_states_q, rgbd_dkl, sensors_dkl, comm_dkl, comm_zq = \
            self.bottom_to_top_step(
                prev_hidden_states, rgbd[:,step], sensors[:,step], comm_in[:,step], 
                prev_wheels_shoulders[:,step], prev_comm_out[:,step])
            """how_many_nans(new_hidden_states_p, f"PVRNN, new_hidden_states_p step {step}")
            how_many_nans(new_hidden_states_q, f"PVRNN, new_hidden_states_q step {step}")
            how_many_nans(rgbd_dkl, f"PVRNN, rgbd_dkl step {step}")
            how_many_nans(comm_dkl, f"PVRNN, comm_dkl step {step}")
            how_many_nans(sensors_dkl, f"PVRNN, sensors_dkl step {step}")"""
                                
            for l, o in zip(
                [new_hidden_states_p_list, new_hidden_states_q_list, rgbd_dkl_list, sensors_dkl_list, comm_dkl_list, comm_zq_list],
                [new_hidden_states_p, new_hidden_states_q, rgbd_dkl, sensors_dkl, comm_dkl, comm_zq.unsqueeze(1)]):     
                l.append(o)
                                
            prev_hidden_states = new_hidden_states_q
                        
        lists = [new_hidden_states_p_list, new_hidden_states_q_list, rgbd_dkl_list, sensors_dkl_list, comm_dkl_list, comm_zq_list]
        for i in range(len(lists)):
            lists[i] = torch.cat(lists[i], dim=1)
        new_hidden_states_p, new_hidden_states_q, rgbd_dkl, sensors_dkl, comm_dkl, comm_zq = lists
                
        pred_rgbd_q, pred_sensors_q, pred_comm_q = self.predict(new_hidden_states_q[:, :-1], prev_wheels_shoulders[:, 1:])
        
        """how_many_nans(pred_rgbd_q, "PVRNN, pred_rgbd_q 2")
        how_many_nans(pred_comm_q, "PVRNN, pred_comm_q 2")
        how_many_nans(pred_sensors_q, "PVRNN, pred_sensors_q 2")"""
                
        task_labels = labels[:, :, 0].clone().unsqueeze(-1)
        color_labels = labels[:, :, 1].clone().unsqueeze(-1)
        shape_labels = labels[:, :, 2].clone().unsqueeze(-1)
        #task_labels[task_labels == 0] = 1
        color_labels[color_labels != 0] = color_labels[color_labels != 0] - 5  # 6-11 -> 1-6
        shape_labels[shape_labels != 0] = shape_labels[shape_labels != 0] - 11  # 12-16 -> 1-5

        # Combine the filtered labels back into a single tensor
        labels = torch.cat((task_labels, color_labels, shape_labels), dim=-1)
                        
        return(new_hidden_states_p, new_hidden_states_q, rgbd_dkl, sensors_dkl, comm_dkl, pred_rgbd_q, pred_sensors_q, pred_comm_q, comm_zq, labels)
        
        
        
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
                                (episodes, steps+1, args.max_comm_len, args.comm_shape),
                                (episodes, steps+1, args.wheels_shoulders_shape),
                                (episodes, steps+1, args.max_comm_len, args.comm_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

    """
    args.layers = 5
    
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
                                (episodes, steps+1, args.wheels_shoulders_shape),
                                (episodes, steps+1, args.max_comm_len, args.comm_shape))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    """

            

# %%