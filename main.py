#%%

import os
import pickle, torch, random
import numpy as np
from multiprocessing import Process, Queue, set_start_method
from time import sleep 
from math import floor

from utils import args, folder, duration, estimate_total_duration, print
from agent import Agent

print("\nname:\n{}".format(args.arg_name))
print("\nagents: {}. previous_agents: {}.".format(args.agents, args.previous_agents))



def train(q, i):
    seed = args.init_seed + i
    np.random.seed(seed) 
    random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    
    if(str(args.device) != "cpu"):
        num_gpus = torch.cuda.device_count()
        gpu_id = i % num_gpus
        args.device = torch.device(f"cuda:{gpu_id}")
        
    num_cores = os.cpu_count()
    cpu_id = i % num_cores
    args.cpu = cpu_id
    
    print(f"\nagent {i}: cpu {cpu_id}\n")
    
    if(args.load_agents):
        with open(folder + "/agents/agent_" + str(i).zfill(3) + ".pickle", "rb") as handle: 
            agent = pickle.load(handle)   
        agent.start_physics()
        agent.args = args
    else:
        agent = Agent(args = args, i = i)
    agent.training(q)

if __name__ == '__main__':
    set_start_method('spawn')
    queue = Queue()
    
    processes = []
    for worker_id in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents):
        process = Process(target=train, args=(queue, worker_id))
        processes.append(process)
        process.start()
    
    progress_dict      = {i : "0"  for i in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents)}
    prev_progress_dict = {i : None for i in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents)}
    
    while any(process.is_alive() for process in processes) or not queue.empty():
        while not queue.empty():
            worker_id, progress_percentage = queue.get()
            progress_dict[worker_id] = progress_percentage
    
        if any(progress_dict[key] != prev_progress_dict[key] for key in progress_dict.keys()):
            prev_progress_dict = progress_dict.copy()
            string = "" ; hundreds = 0
            values = list(progress_dict.values()) ; values.sort()
            so_far = duration()
            lowest = float(values[0])
            estimated_total = estimate_total_duration(lowest)
            if(estimated_total == "?:??:??"): to_do = "?:??:??"
            else:                                   to_do = estimated_total - so_far
            values = [str(floor(100 * float(value))).ljust(3, " ") for value in values]
            for value in values:
                if(value != "100"): string += " " + value
                else:               hundreds += 1 
            if(hundreds > 0): string += " ##" + " 100" * hundreds
            string = "{} ({} left):".format(so_far, to_do) + string
            if(hundreds == 0): string += " ##"
            string = string.rstrip() + "."
            print(string)
        sleep(1)
    
    for process in processes:
        process.join()
    
    print("\nDuration: {}. Done!".format(duration()))
    # %%
