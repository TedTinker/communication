#%%

import numpy as np

from utils_submodule import pad_zeros



class VariableBuffer:
    def __init__(self, args, shape = (1,), before_and_after = False):
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
    def __init__(self, args):
        self.args = args
        self.capacity = self.args.capacity
        self.max_episode_len = self.args.max_steps
        self.num_episodes = 0

        self.vision = VariableBuffer(
            shape = (self.args.image_size, self.args.image_size, 4), 
            before_and_after = True, 
            args = self.args)
        self.touch = VariableBuffer(
            shape = (self.args.touch_shape + self.args.joint_aspects,), 
            before_and_after = True, 
            args = self.args)
        self.command_voice = VariableBuffer(
            shape = (self.args.max_voice_len, self.args.voice_shape,), 
            before_and_after = True, 
            args = self.args)
        self.report_voice = VariableBuffer(
            shape = (self.args.max_voice_len, self.args.voice_shape,), 
            before_and_after = True, 
            args = self.args)
        self.wheels_joints = VariableBuffer(
            shape = (self.args.wheels_joints_shape,), 
            args = self.args)
        self.voice_out = VariableBuffer(
            shape = (self.args.max_voice_len, self.args.voice_shape,), 
            args = self.args)
        self.reward = VariableBuffer(args = self.args)
        self.done = VariableBuffer(args = self.args)
        self.mask = VariableBuffer(args = self.args)

        self.episode_ptr = 0
        self.time_ptr = 0

    def push(
            self, 
            vision,
            touch,
            command_voice, 
            report_voice,
            wheels_joints, 
            voice_out, 
            reward, 
            next_vision,
            next_touch,
            next_command_voice, 
            next_report_voice,
            done):
        
                
        if self.time_ptr == 0:
            for buffer in [
                    self.vision,
                    self.touch,
                    self.command_voice, 
                    self.report_voice,
                    self.wheels_joints, 
                    self.voice_out, 
                    self.reward, 
                    self.done, 
                    self.mask]:
                buffer.reset_episode(self.episode_ptr)

        command_voice = pad_zeros(command_voice, self.args.max_voice_len)
        next_command_voice = pad_zeros(next_command_voice, self.args.max_voice_len)
        self.vision.push(self.episode_ptr, self.time_ptr, vision)
        self.touch.push(self.episode_ptr, self.time_ptr, touch)
        self.command_voice.push(self.episode_ptr, self.time_ptr, command_voice)
        self.report_voice.push(self.episode_ptr, self.time_ptr, report_voice)
        self.wheels_joints.push(self.episode_ptr, self.time_ptr, wheels_joints)
        self.voice_out.push(self.episode_ptr, self.time_ptr, voice_out)
        self.reward.push(self.episode_ptr, self.time_ptr, reward)
        self.done.push(self.episode_ptr, self.time_ptr, done)
        self.mask.push(self.episode_ptr, self.time_ptr, 1.0)

        self.time_ptr += 1
        if done or self.time_ptr >= self.max_episode_len:
            self.vision.push(self.episode_ptr, self.time_ptr, next_vision)
            self.touch.push(self.episode_ptr, self.time_ptr, next_touch)
            self.command_voice.push(self.episode_ptr, self.time_ptr, next_command_voice)
            self.report_voice.push(self.episode_ptr, self.time_ptr, next_report_voice)
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
            self.vision.sample(indices),
            self.touch.sample(indices),
            self.command_voice.sample(indices),
            self.report_voice.sample(indices),
            self.wheels_joints.sample(indices),
            self.voice_out.sample(indices),
            self.reward.sample(indices),
            self.done.sample(indices),
            self.mask.sample(indices))
        return batch