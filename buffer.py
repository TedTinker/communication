#%%

import numpy as np

from utils import default_args
from utils_submodule import pad_zeros



class VariableBuffer:
    def __init__(self, shape = (1,), before_and_after = False, args = default_args):
        self.args = args
        self.shape = shape
        self.data = np.zeros((self.args.capacity, self.args.max_steps + (1 if before_and_after else 0)) + self.shape, dtype='float32')

    def reset_episode(self, episode_ptr):
        self.data[episode_ptr] = 0

    def push(self, episode_ptr, time_ptr, value):
        if self.shape == (1,): self.data[episode_ptr, time_ptr]    = value
        else:                  self.data[episode_ptr, time_ptr, :] = value

    def sample(self, indices):
        return self.data[indices]



class RecurrentReplayBuffer:
    def __init__(self, args = default_args):
        self.args = args
        self.capacity = self.args.capacity
        self.max_episode_len = self.args.max_steps
        self.num_episodes = 0

        self.rgbd = VariableBuffer(
            shape = (self.args.image_size, self.args.image_size, 4), 
            before_and_after = True, 
            args = self.args)
        self.sensors = VariableBuffer(
            shape = (self.args.sensors_shape,), 
            before_and_after = True, 
            args = self.args)
        self.father_comm_in = VariableBuffer(
            shape = (self.args.max_comm_len, self.args.comm_shape,), 
            before_and_after = True, 
            args = self.args)
        self.mother_comm_in = VariableBuffer(
            shape = (self.args.max_comm_len, self.args.comm_shape,), 
            before_and_after = True, 
            args = self.args)
        self.action = VariableBuffer(
            shape = (self.args.action_shape,), 
            args = self.args)
        self.comm_out = VariableBuffer(
            shape = (self.args.max_comm_len, self.args.comm_shape,), 
            args = self.args)
        self.reward = VariableBuffer(args = self.args)
        self.done = VariableBuffer(args = self.args)
        self.mask = VariableBuffer(args = self.args)

        self.episode_ptr = 0
        self.time_ptr = 0

    def push(
            self, 
            rgbd,
            sensors,
            father_comm_in, 
            mother_comm_in,
            action, 
            comm_out, 
            reward, 
            next_rgbd,
            next_sensors,
            next_father_comm_in, 
            next_mother_comm_in,
            done):
        
                
        if self.time_ptr == 0:
            for buffer in [
                    self.rgbd,
                    self.sensors,
                    self.father_comm_in, 
                    self.mother_comm_in,
                    self.action, 
                    self.comm_out, 
                    self.reward, 
                    self.done, 
                    self.mask]:
                buffer.reset_episode(self.episode_ptr)

        father_comm_in = pad_zeros(father_comm_in, self.args.max_comm_len)
        mother_comm_in = pad_zeros(mother_comm_in, self.args.max_comm_len)
        next_father_comm_in = pad_zeros(next_father_comm_in, self.args.max_comm_len)
        next_mother_comm_in = pad_zeros(next_mother_comm_in, self.args.max_comm_len)
        
        self.rgbd.push(self.episode_ptr, self.time_ptr, rgbd)
        self.sensors.push(self.episode_ptr, self.time_ptr, sensors)
        self.father_comm_in.push(self.episode_ptr, self.time_ptr, father_comm_in)
        self.mother_comm_in.push(self.episode_ptr, self.time_ptr, mother_comm_in)
        self.action.push(self.episode_ptr, self.time_ptr, action)
        self.comm_out.push(self.episode_ptr, self.time_ptr, comm_out)
        self.reward.push(self.episode_ptr, self.time_ptr, reward)
        self.done.push(self.episode_ptr, self.time_ptr, done)
        self.mask.push(self.episode_ptr, self.time_ptr, 1.0)

        self.time_ptr += 1
        if done or self.time_ptr >= self.max_episode_len:
            self.rgbd.push(self.episode_ptr, self.time_ptr, next_rgbd)
            self.sensors.push(self.episode_ptr, self.time_ptr, next_sensors)
            self.father_comm_in.push(self.episode_ptr, self.time_ptr, next_father_comm_in)
            self.mother_comm_in.push(self.episode_ptr, self.time_ptr, next_mother_comm_in)
            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0
            self.num_episodes = min(self.num_episodes + 1, self.capacity)

    def sample(self, batch_size, random_sample = True):
        if(self.num_episodes == 0): return(False)
        if(random_sample):
            if(self.num_episodes < batch_size):
                indices = np.random.choice(self.num_episodes, self.num_episodes, replace=False)
            else:
                indices = np.random.choice(self.num_episodes, batch_size, replace=False)
        else:
            indices = [i for i in range(batch_size)]
        batch = (
            self.rgbd.sample(indices),
            self.sensors.sample(indices),
            self.father_comm_in.sample(indices),
            self.mother_comm_in.sample(indices),
            self.action.sample(indices),
            self.comm_out.sample(indices),
            self.reward.sample(indices),
            self.done.sample(indices),
            self.mask.sample(indices))
        return batch