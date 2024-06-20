#%%
import os
import pickle

from task import Task, Task_Runner
from agent import Agent 



hyper_parameters = "ef_2"
agent_num = 1
epochs = 5000
saved_file = "saved_deigo"



print("\n\nLoading...", end = " ")
with open(f'{saved_file}/{hyper_parameters}/plot_dict.pickle', 'rb') as file:
    plot_dict = pickle.load(file)
    
    agent_lists = plot_dict["agent_lists"]
    args = plot_dict["args"]
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

agent.tasks = {0 : Task(actions = [1], objects = 2, colors = [0, 1, 2, 3, 4, 5], shapes = [0,], parenting = True, args = agent.args)}
agent.task_runners = {task_name : Task_Runner(task, agent.arena_1, agent.arena_2) for i, (task_name, task) in enumerate(agent.tasks.items())}
agent.task_name = 0
agent.task = agent.task_runners[agent.task_name]

episodes = 0
wins = 0

while(True):
    episodes += 1
    win = agent.save_episodes(test = False, sleep_time = 2, for_display = True)
    if(win): 
        wins += 1
    print(f"\tWIN RATE: {round(100 * (wins / episodes), 2)}%")
    WAITING = input("WAITING")
# %%