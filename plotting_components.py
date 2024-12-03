#%%
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from utils import args, duration, load_dicts, print, task_map, color_map, shape_map

print("name:\n{}\n".format(args.arg_name),)

dpi = 50

task_names = ['SILENCE', 'WATCH', 'PUSH', 'PULL', 'LEFT', 'RIGHT']
color_names = ['RED', 'GREEN', 'BLUE', 'CYAN', 'PINK', 'YELLOW']
shape_names = ['PILLAR', 'POLE', 'DUMBBELL', 'DELTA', 'HOURGLASS']

lda_task = LDA(n_components = len(task_names) - 1)
lda_color = LDA(n_components = len(color_names) - 1)
lda_shape = LDA(n_components = len(shape_names) - 1)



def color_based_on_title(task_name):
    task_str, color_str, shape_str = task_name.split('_')
    task_index = task_names.index(task_str)
    color_index = color_names.index(color_str)
    shape_index = shape_names.index(shape_str)
    
    task_normalized = task_index / (len(task_names) - 1)
    color_normalized = color_index / (len(color_names) - 1)
    shape_normalized = shape_index / (len(shape_names) - 1)
    
    r = int(task_normalized * 255)
    g = int(color_normalized * 255)
    b = int(shape_normalized * 255)
    
    color = f'rgb({r}, {g}, {b})'
    
    #print(task_name, "\t", color)
    return color



def add_jitter(values, jitter_amount=0.1):
    noise = np.random.uniform(-jitter_amount, jitter_amount, size=values.shape)
    return values + noise



def plot_interactive_3d(plot_dict, file_name="plot"):
    args = plot_dict["args"]
    for agent_num, values_for_composition in enumerate(plot_dict["component_data"]):
        if(values_for_composition != {}):
            print("\nAGENT NUM", agent_num)
            for epochs, (father_voice_zq, labels, all_mask, father_voice_zq_filtered, labels_filtered, all_mask_filtered) in values_for_composition.items():
                print("EPOCHS", epochs)
                
                #print(father_voice_zq.shape, labels.shape, all_mask.shape, father_voice_zq_filtered.shape, labels_filtered.shape, all_mask_filtered.shape)
                
                #print("\n\n")
                #print(labels)
                #print("\n\n")
                #print(labels_filtered)
                #print("\n\n")

                labels = labels.reshape(-1, labels.shape[-1])
                labels = labels[all_mask.reshape(-1).astype(bool)]
                
                labels_filtered = labels_filtered.reshape(-1, labels_filtered.shape[-1])
                labels_filtered = labels_filtered[all_mask_filtered.reshape(-1).astype(bool)]
                                
                def plot_lda(this, this_filtered, type_name):
                    
                    try: os.mkdir(f"thesis_pics/composition/{args.arg_name}/lda")
                    except: pass
                    try: os.mkdir(f"thesis_pics/composition/{args.arg_name}/lda/{type_name}")
                    except: pass
                    
                    all_processor_names = plot_dict["all_processor_names"]
                    all_processor_names_no_free_play = [processor_name for processor_name in all_processor_names if processor_name.split("_")[0] != "SILENCE"]
                    
                    episode_indexes = [0]
                    episode_index = 0             
                    for mask in all_mask: 
                        episode_len = int(mask.sum().item())
                        episode_indexes.append(episode_len + episode_index)
                        episode_index = episode_indexes[-1]
                        
                    episode_indexes_no_free_play = [0]
                    episode_index_no_free_play = 0
                    for mask in all_mask_filtered: 
                        episode_len = int(mask.sum().item())
                        episode_indexes_no_free_play.append(episode_len + episode_index)
                        episode_index_no_free_play = episode_indexes_no_free_play[-1]           
                                                                        
                    this = this.reshape(-1, this.shape[-1])
                    this = this[all_mask.reshape(-1).astype(bool)]
                    
                    this_filtered = this_filtered.reshape(-1, this_filtered.shape[-1])
                    this_filtered = this_filtered[all_mask_filtered.reshape(-1).astype(bool)]            
                                                                                                            
                    lda_task.fit(this, labels[:,0])
                    lda_color.fit(this_filtered, labels_filtered[:,1])
                    lda_shape.fit(this_filtered, labels_filtered[:,2])
                    
                    task_probs = lda_task.predict_proba(this)
                    color_probs = lda_color.predict_proba(this_filtered)
                    shape_probs = lda_shape.predict_proba(this_filtered)
                                                                                                                                                           
                    all_pred_tasks = []
                    all_pred_colors = []
                    all_pred_shapes = []
                    
                    fig = go.Figure()
                
                    for i, processor_name in enumerate(all_processor_names):
                        start_idx = episode_indexes[i]
                        end_idx = episode_indexes[i + 1]
                        these_task_probs = task_probs[start_idx:end_idx]
                        pred_tasks = np.argmax(these_task_probs, axis = 1)
                        all_pred_tasks.append(pred_tasks)
                        
                    for i, processor_name in enumerate(all_processor_names_no_free_play):               
                        start_idx = episode_indexes_no_free_play[i]
                        end_idx = episode_indexes_no_free_play[i + 1]
                        
                        these_color_probs = color_probs[start_idx:end_idx]
                        pred_colors = np.argmax(these_color_probs, axis = 1)
                        all_pred_colors.append(pred_colors)
                        #print(pred_colors.shape)
                        pc = np.array([item for sublist in all_pred_colors for item in sublist])
                        #print("\t", pc.shape)
                        
                        these_shape_probs = shape_probs[start_idx:end_idx]        
                        pred_shapes = np.argmax(these_shape_probs, axis = 1)
                        all_pred_shapes.append(pred_shapes)

                    output_file = f"thesis_pics/composition/{args.arg_name}/lda/{type_name}/{file_name}_epoch_{epochs}_{type_name}_agent_num_{agent_num}_{args.arg_name}_lda"
                    output_file_2d = output_file + ".png"
                    
                    all_pred_tasks = np.array([item for sublist in all_pred_tasks for item in sublist])
                    all_pred_colors = np.array([item for sublist in all_pred_colors for item in sublist])
                    all_pred_shapes = np.array([item for sublist in all_pred_shapes for item in sublist])
                                        
                    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
                    fig.suptitle(f'LDA for {args.arg_name} using {type_name} in epoch {epochs}', fontsize=16)
                    
                    size = 1 
                    jitter = .3
                    
                    jittered_labels = add_jitter(labels, jitter)
                    jittered_labels_filtered = add_jitter(labels_filtered, jitter)
                    all_pred_tasks = add_jitter(all_pred_tasks, jitter)
                    all_pred_colors = add_jitter(all_pred_colors, jitter)
                    all_pred_shapes = add_jitter(all_pred_shapes, jitter)
                    
                    #print("Tasks:",  len(jittered_labels[:,0]),          all_pred_tasks.shape)
                    #print("Colors:", len(jittered_labels_filtered[:,1]), all_pred_colors.shape)
                    #print("Shapes:", len(jittered_labels_filtered[:,2]), all_pred_shapes.shape)
                    #print("\n\n")
                    
                    axs[0].scatter(jittered_labels[:,0] + 1, all_pred_tasks, c = "black", s = size)
                    axs[0].set_title('Tasks')
                    axs[0].set_xlabel('Real Labels')
                    axs[0].set_ylabel('LDA Predictions')
                    axs[0].set_xticks([1, 2, 3, 4, 5, 6])  
                    axs[0].set_xticklabels(task_names)  
                    axs[0].set_yticks([0, 1, 2, 3, 4, 5]) 
                    axs[0].set_yticklabels(task_names) 
                    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right")
                    plt.setp(axs[0].get_yticklabels(), rotation=0, ha="right")

                    axs[1].scatter(jittered_labels_filtered[:,1], all_pred_colors, c = "black", s = size)
                    axs[1].set_title('Colors')
                    axs[1].set_xlabel('Real Labels')
                    axs[1].set_ylabel('LDA Predictions')
                    axs[1].set_xticks([1, 2, 3, 4, 5, 6])  
                    axs[1].set_xticklabels(color_names)  
                    axs[1].set_yticks([0, 1, 2, 3, 4, 5]) 
                    axs[1].set_yticklabels(color_names) 
                    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")
                    plt.setp(axs[1].get_yticklabels(), rotation=0, ha="right")

                    axs[2].scatter(jittered_labels_filtered[:,2], all_pred_shapes, c = "black", s = size)
                    axs[2].set_title('Shapes')
                    axs[2].set_xlabel('Real Labels')
                    axs[2].set_ylabel('LDA Predictions')
                    axs[2].set_xticks([1, 2, 3, 4, 5])  
                    axs[2].set_xticklabels(shape_names)  
                    axs[2].set_yticks([0, 1, 2, 3, 4]) 
                    axs[2].set_yticklabels(shape_names) 
                    plt.setp(axs[2].get_xticklabels(), rotation=45, ha="right")
                    plt.setp(axs[2].get_yticklabels(), rotation=0, ha="right")

                    plt.tight_layout()
                    plt.savefig(output_file_2d)
                    plt.close()
                    
                    print(f"LDA for {args.arg_name} using {type_name} in epoch {epochs} plotted.")
                    
                for this, this_filtered, type_name in [
                    (father_voice_zq, father_voice_zq_filtered, "father_voice_zq")]:
                    plot_lda(this, this_filtered, type_name)
                    
                #print("MAKING PCA")
                #for this, type_name in [
                #    (hq, "hq"), 
                #    (zp_zq_dkls["father_voice"].zp, "father_voice_zp"),
                #    (zp_zq_dkls["father_voice"].zq, "father_voice_zq")]:
                #    plot_pca(this, type_name)



def plots(plot_dicts):
    for i, plot_dict in enumerate(plot_dicts):
        args = plot_dict["args"]
        try: os.mkdir(f"thesis_pics/composition/{args.arg_name}")
        except: pass
        print(f"\nStarting {args.arg_name}.")
        
        plot_interactive_3d(plot_dict)

        print(f"Finished {args.arg_name}.")
        print(f"Duration: {duration()}")
        
    print(f"\tFinished composition.")



if(os.getcwd().split("/")[-1] != "communication"): os.chdir("communication")
os.chdir(f"saved_{args.comp}")
try: os.mkdir("thesis_pics/composition")
except: pass
    
plot_dicts, min_max_dict, complete_order = load_dicts(args)
plots(plot_dicts)
print(f"\nDuration: {duration()}. Done!")
# %%