import os, pickle, re

from utils import duration, args, print, save_file

print("name:\n{}".format(args.arg_name))

os.chdir(save_file)
folders = [item for item in os.listdir() if os.path.isdir(item)]
folders.sort()
try: folders.remove("thesis_pics")                                                                                                                                                                          
except: pass                                                                                                                                                                                
print("\n{} folders.".format(len(folders)))

plot_dict_pattern = re.compile(r'^plot_dict_\d{3}\.pickle$')
min_max_dict_pattern = re.compile(r'^min_max_dict_\d{3}\.pickle$')

for folder in folders:
    plot_dict = {} ; min_max_dict = {}
    files = os.listdir(folder) ; files.sort()
    print("{} files in folder {}.".format(len(files), folder))
    
    filtered_files = [file for file in files if file != "agents"]
    all_files_match = all(plot_dict_pattern.match(file) or min_max_dict_pattern.match(file) for file in filtered_files)
    if(not all_files_match):
        print("Skipping.")
    else:
        for file in filtered_files:                                                                                                                                                                          
            if(file.split("_")[0] == "plot"): d = plot_dict    ; plot = True 
            if(file.split("_")[0] == "min"):  d = min_max_dict ; plot = False
            with open(folder + "/" + file, "rb") as handle: 
                saved_d = pickle.load(handle)          
            for key in saved_d.keys(): 
                if(not key in d): d[key] = []
                if(key in ["args", "arg_title", "arg_name", "all_task_names"]): d[key] = saved_d[key]
                else:  d[key].append(saved_d[key])
                
        episode_dicts = {}
        for d in plot_dict["episode_dicts"]: 
            episode_dicts.update(d)
        plot_dict["episode_dicts"] = episode_dicts
        
        agent_lists = {}
        for d in plot_dict["agent_lists"]: agent_lists.update(d)
        plot_dict["agent_lists"] = agent_lists
            
        for key in min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "all_task_names", "values_for_composition", "episode_dicts", "agent_lists", "spot_names", "steps", "goal_action"]):
                minimum = None ; maximum = None
                for min_max in min_max_dict[key]:
                    if(any(item in [None, "not_used"] for item in min_max)): 
                        pass
                    else:
                        if(  minimum == None):      minimum = min_max[0]
                        elif(minimum > min_max[0]): minimum = min_max[0]
                        if(  maximum == None):      maximum = min_max[1]
                        elif(maximum < min_max[1]): maximum = min_max[1]
                min_max_dict[key] = (minimum, maximum)
                
        files = os.listdir(folder) ; files.sort()
        filtered_files = [file for file in files if file != "agents"]
        for file in filtered_files:                
            os.remove(folder + "/" + file)         

        with open(folder + "/min_max_dict.pickle", "wb") as handle:
            pickle.dump(min_max_dict, handle)
                
        with open(folder + "/plot_dict.pickle", "wb") as handle:
            pickle.dump(plot_dict, handle)         
            
        print(f"Finished {folder}.")
    
print("\nDuration: {}. Done!\n".format(duration()))