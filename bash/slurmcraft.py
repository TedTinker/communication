#%%
from copy import deepcopy
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("--comp",         type=str,  default = "deigo")
parser.add_argument("--agents",       type=int,  default = 10)
parser.add_argument("--arg_list",     type=str,  default = [])
try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

if(type(args.arg_list) != list): args.arg_list = json.loads(args.arg_list)
combined = "___{}___".format("+".join(args.arg_list))    

import os 
try:    os.chdir("communication/bash")
except: pass



from itertools import product
def expand_args(name, args):
    combos = [{}]
    complex = False
    for key, value in args.items():
        if(type(value) != list):
            for combo in combos:
                combo[key] = value
        else: 
            complex = True
            if(value[0]) == "num_min_max": 
                num, min_val, max_val = value[1]
                num = int(num)
                min_val = float(min_val)
                max_val = float(max_val)
                value = [min_val + i*((max_val - min_val) / (num - 1)) for i in range(num)]
            new_combos = []
            for v in value:
                temp_combos = deepcopy(combos)
                for combo in temp_combos: 
                    combo[key] = v        
                    new_combos.append(combo)   
            combos = new_combos  
    if(complex and name[-1] != "_"): name += "_"
    return(name, combos)

slurm_dict = {
    "d"    : {}, 
    "e"    : {"alpha" : "None"},
    "n"    : {                  "curiosity" : "naive"},
    "en"   : {"alpha" : "None", "curiosity" : "naive"},
    "f"    : {                  "curiosity" : "free",  "beta" : .05},
    "ef"   : {"alpha" : "None", "curiosity" : "free",  "beta" : .05},
    }

def get_args(name):
    s = "" 
    for key, value in slurm_dict[name].items(): s += "--{} {} ".format(key, value)
    return(s)

def add_this(name, args):
    keys, values = [], []
    for key, value in slurm_dict.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        new_key = key + "_" + name 
        new_value = deepcopy(value)
        for arg_name, arg in args.items():
            if(type(arg) != list or type(arg[0]) != dict): new_value[arg_name] = arg
            else:
                for if_arg_name, if_arg in arg[0].items():
                    if(if_arg_name in value and value[if_arg_name] == if_arg):
                        new_value[arg_name] = arg[1]
        slurm_dict[new_key] = new_value



new_slurm_dict = {}
for key, value in slurm_dict.items():
    key, combos = expand_args(key, value)
    if(len(combos) == 1): new_slurm_dict[key] = combos[0] 
    else:
        for i, combo in enumerate(combos): new_slurm_dict[key + str(i+1)] = combo
        
slurm_dict = new_slurm_dict

def all_like_this(this): 
    if(this in ["break", "empty_space"]): result = [this]
    elif(this[-1] != "_"):                result = [this]
    else: result = [key for key in slurm_dict.keys() if key.startswith(this) and key[len(this):].isdigit()]
    return(json.dumps(result))
            


        
if(__name__ == "__main__" and args.arg_list == []):
    for key, value in slurm_dict.items(): print(key, ":", value,"\n")

max_cpus = 36
if(__name__ == "__main__" and args.arg_list != []):
    
    if(args.comp == "deigo"):
        partition = \
"""
#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G
"""

    if(args.comp == "saion"):
        partition = \
"""
#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G
"""

    for name in args.arg_list:
        if(name in ["break", "empty_space"]): pass 
        else:
            with open("main_{}.slurm".format(name), "w") as f:
                f.write(
"""
{}
#SBATCH --ntasks={}
module load singularity
singularity exec maze.sif python communication/main.py --comp {} --arg_name {} {} --agents $agents_per_job --previous_agents $previous_agents
""".format(partition, max_cpus, args.comp, name, get_args(name))[2:])
            


    with open("finish_dicts.slurm", "w") as f:
        f.write(
"""
{}
module load singularity
singularity exec maze.sif python communication/finish_dicts.py --comp {} --arg_title {} --arg_name finishing_dictionaries
""".format(partition, args.comp, combined)[2:])
        
    with open("plotting.slurm", "w") as f:
        f.write(
"""
{}
module load singularity
singularity exec maze.sif python communication/plotting.py --comp {} --arg_title {} --arg_name plotting
""".format(partition, args.comp, combined)[2:])
# %%

