#%% 

import os
import pickle
import torch
from copy import deepcopy

from utils import wait_for_button_press
from processor import Processor
from agent import Agent 
from plotting_episodes import plot_step

hyper_parameters = "ef"
agent_num = "0001"
epochs = "070000"
saved_file = "saved_deigo"



print("\n\nLoading...", end = " ")
    
with open(f'{saved_file}/{hyper_parameters}/agents/args.pickle', 'rb') as file:
    args = pickle.load(file)
print("Loaded!\n\n")

print("Making arena...", end = " ")
agent = Agent(GUI = True, args = args)
print("Made arena!")

agent.load_agent(load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pth.gz')

episodes = 0
wins = 0



def dream_step(self,
    prev_obs_pred_1, prev_action_1, hq_1, 
    prev_obs_pred_2, prev_action_2, hq_2, sleep_time = None):

    with torch.no_grad():
        self.eval()
        parenting = self.processor.parenting
                    
                    
            
        def agent_step():
            real_obs = self.processor.obs(1)
            obs = prev_obs_pred_1
            if(obs == None):
                obs = real_obs
            prev_action = prev_action_1
            partner_prev_voice_out = prev_action_2.voice_out
            hq = hq_1
                            
            obs.father_voice = obs.father_voice.one_hots.unsqueeze(0).unsqueeze(0) 
            obs.mother_voice = obs.mother_voice.one_hots.unsqueeze(0).unsqueeze(0) 

            hp, hq, rgbd_is, sensors_is, father_voice_is, mother_voice_is = self.forward.bottom_to_top_step(
                hq, self.forward.obs_in(obs), self.forward.action_in(prev_action))

            action, _, _ = self.actor(hq.detach(), parenting) 
            values = []
            for i in range(self.args.critics):
                value = self.critics[i](action, hq.detach()) 
                values.append(round(value.item(), 3))
                
            pred_obs_p = self.forward.predict(hp, self.forward.wheels_joints_in(action.wheels_joints)) 
            pred_obs_q = self.forward.predict(hq, self.forward.wheels_joints_in(action.wheels_joints)) 
            
            return(real_obs, pred_obs_p, pred_obs_q, action, hp, hq, values, rgbd_is, sensors_is, father_voice_is, mother_voice_is)
        
        
        
        real_obs_1, pred_obs_p_1, pred_obs_q_1, action_1, hp_1, hq_1, values_1, rgbd_is_1, sensors_is_1, father_voice_is_1, mother_voice_is_1 = agent_step()

        # This action is happening, but agent is hallucinating.
        reward, done, win = self.processor.step(action_1.wheels_joints[0,0].clone(), None, sleep_time = sleep_time)
        
        def next_agent_step():
            
            next_obs = self.processor.obs(1)
            obs = pred_obs_q_1
            action = action_1
            partner_voice_out = None
            
            next_obs.father_voice = next_obs.father_voice.one_hots.unsqueeze(0).unsqueeze(0)
            next_obs.mother_voice = next_obs.mother_voice.one_hots.unsqueeze(0).unsqueeze(0)

            to_push = To_Push(obs, action, reward, next_obs, done)     
            return(next_obs, to_push)
        
        next_obs_1, to_push_1 = next_agent_step()
        
    torch.cuda.empty_cache()
    
    return(
        real_obs_1, pred_obs_p_1, pred_obs_q_1, action_1, values_1, hp_1.squeeze(1), hq_1.squeeze(1), rgbd_is_1, sensors_is_1, father_voice_is_1, mother_voice_is_1,
        reward, done, win, to_push_1)


    
def save_dreams(self, swapping = False, test = False, sleep_time = None, for_display = False):        
    with torch.no_grad():
        self.processor = self.processors[self.processor_name]
        self.processor.begin(test = test)       
        parenting = self.processor.parenting

        common_keys = [
            "obs", "action", 
            "birds_eye", "reward", "critic_predictions", "prior_predictions", "posterior_predictions", 
            "rgbd_dkl", "sensors_dkl", "father_voice_dkl", "mother_voice_dkl"]
        episode_dict = {}
        for agent_id in [0, 1]:
            for key in common_keys:
                episode_dict[f"{key}_{agent_id}"] = []
        episode_dict["reward"] = []
        episode_dict["processor"] = self.processor
        episode_dict["goal"] = self.processor.goal
        
        done = False
        
        done, complete_reward, steps, \
            (to_push_list_1, prev_action_1, hq_1), \
            (to_push_list_2, prev_action_2, hq_2) = self.start_episode()
                
        hp_1 = deepcopy(hq_1)
        hp_2 = deepcopy(hp_1)
        
        
                        
        def save_step(step, hp, hq, wheels_joints):
            birds_eye = self.processor.arena_1.photo_from_above()
            obs = self.processor.obs()
            
            episode_dict[f"obs_1"].append(obs) 
            episode_dict[f"birds_eye_1"].append(birds_eye[:,:,0:3])
                
                
            
        def display(step, done = False, stopping = False, wait = True):
            if(for_display):
                print(f"\n{self.processor.goal.human_text}", end = " ")
                print("STEP:", step)
                plot_step(step, episode_dict, last_step = done, saving = False)
                if(not self.processor.parenting and not stopping):
                    display(step, stopping = True)
                if(wait): 
                    WAITING = wait_for_button_press()
                    
                    
        
        pred_obs_q_1 = None 
        pred_obs_q_2 = None
        for step in range(self.args.max_steps + 1):
            save_step(step, hp_1, hq_1, wheels_joints = prev_action_1.wheels_joints)    
            if(not self.processor.parenting):
                save_step(step, hp_2, hq_2, wheels_joints = prev_action_2.wheels_joints)  
                
            display(step)
            
            real_obs_1, pred_obs_p_1, pred_obs_q_1, prev_action_1, values_1, hp_1, hq_1, rgbd_is_1, sensors_is_1, father_voice_is_1, mother_voice_is_1, \
                    reward, done, win, to_push_1 = self.dream_step(
                        pred_obs_q_1, prev_action_1, hq_1,
                        pred_obs_q_2, prev_action_2, hq_2, sleep_time) 
                    
            episode_dict["reward"].append(str(round(reward, 3)))
            
            def update_episode_dict(index, pred_obs_p, pred_obs_q, prev_action, rgbd_is, sensors_is, father_voice_is, mother_voice_is, values, reward):
                episode_dict[f"action_{index}"].append(prev_action)
                episode_dict[f"rgbd_dkl_{index}"].append(rgbd_is.dkl.sum().item())
                episode_dict[f"sensors_dkl_{index}"].append(sensors_is.dkl.sum().item())
                episode_dict[f"father_voice_dkl_{index}"].append(father_voice_is.dkl.sum().item())
                episode_dict[f"mother_voice_dkl_{index}"].append(mother_voice_is.dkl.sum().item())
                episode_dict[f"critic_predictions_{index}"].append(values)
                episode_dict[f"reward_{index}"].append(str(round(reward, 3)))
                                
                pred_obs_p.father_voice = get_goal_from_one_hots(pred_obs_p.father_voice)
                pred_obs_q.father_voice = get_goal_from_one_hots(pred_obs_q.father_voice)
                
                pred_obs_p.mother_voice = get_goal_from_one_hots(pred_obs_p.mother_voice)
                pred_obs_q.mother_voice = get_goal_from_one_hots(pred_obs_q.mother_voice)
                
                episode_dict[f"prior_predictions_1"].append(pred_obs_p)
                episode_dict[f"posterior_predictions_1"].append(pred_obs_q)
                
                

            update_episode_dict(1, pred_obs_p_1, pred_obs_q_1, prev_action_1, rgbd_is_1, sensors_is_1, father_voice_is_1, mother_voice_is_1, values_1, reward)
            if not self.processor.parenting:
                update_episode_dict(2, pred_obs_p_2, pred_obs_q_2, prev_action_2, rgbd_is_2, sensors_is_2, father_voice_is_2, mother_voice_is_2, values_2, reward_2)
            
            if(done):
                save_step(step, hp_1, hq_1, wheels_joints = prev_action_1.wheels_joints)    
                if(not self.processor.parenting):
                    save_step(step, hp_2, hq_2, wheels_joints = prev_action_2.wheels_joints) 
                display(step + 1, done = True, wait = False)
                self.processor.done()
                break
        
        if(for_display):
            return(win)
        else:
            pass 



agent.dream_step = dream_step.__get__(agent)
agent.save_dreams = save_dreams.__get__(agent)

print("Ready to go!")



#%%

    #1,  # Watch
    #2,  # Push
    #3,  # Pull
    #4,  # Left
    #5   # Right   
    
"""agent.processors = {0 : Processor(
    agent.arena_1, agent.arena_2,
    tasks_and_weights = [(3, 1)], 
    objects = 2, 
    colors = [0, 1, 2, 3, 4, 5], 
    shapes = [0, 1, 2, 3, 4], 
    parenting = True, 
    args = agent.args)}"""
    
agent.processors = {0 : Processor(
    agent.args, agent.arena_1, agent.arena_2,
    tasks_and_weights = [(2, 1)], 
    objects = 2, 
    colors = [2], 
    shapes = [0], 
    parenting = True)}

agent.processor_name = 0

agent.save_dreams(test = None, sleep_time = 1, for_display = True)


# %%

agent.processor.done()
# %%
