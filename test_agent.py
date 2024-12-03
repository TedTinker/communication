#%%
import os
import pickle

from processor import Processor
from agent import Agent 

hyper_parameters = "ej_t1_3"
agent_num = 3
epochs = 50000
saved_file = "saved_deigo"



print("\n\nLoading...", end = " ")
with open(f'{saved_file}/{hyper_parameters}/plot_dict.pickle', 'rb') as file:
    plot_dict = pickle.load(file)
    agent_lists = plot_dict["agent_lists"]
    args = plot_dict["args"]
    args.left_duration = 1 
    args.right_duration = 1
print("Loaded!\n\n")

print("Making arena...", end = " ")
agent = Agent(GUI = True, args = args)
print("Made arena!")

agent.forward = agent_lists["forward"]
agent.actor = agent_lists["actor"]
for i in range(agent.args.critics):
    agent.critics[i] = agent_lists["critic"]
    agent.critic_targets[i] = agent_lists["critic"]
    
these_parameters = agent_lists[f"{agent_num}_{epochs}"]
agent.load_state_dict(state_dict = these_parameters)

episodes = 0
wins = 0
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
    agent.arena_1, agent.arena_2,
    tasks_and_weights = [(2, 1)], 
    objects = 2, 
    colors = [2], 
    shapes = [0], 
    parenting = True, 
    args = agent.args)}

agent.processor_name = 0


episodes += 1
win = agent.save_episodes(test = None, sleep_time = 1, for_display = True)
if(win): 
    wins += 1
print(f"\tWIN RATE: {round(100 * (wins / episodes), 2)}% \t ({wins} wins out of {episodes} episodes)")




# %%