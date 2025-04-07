#%%
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
#import umap

from utils import args, duration, load_dicts, print, task_map, color_map, shape_map



print("name:\n{}\n".format(args.arg_name),)

dpi = 50

task_names = ['WATCH', 'BE NEAR', 'TOUCH THE TOP', 'PUSH FORWARD', 'PUSH LEFT', 'PUSH RIGHT']
color_names = ['RED', 'GREEN', 'BLUE', 'CYAN', 'PINK', 'YELLOW']
shape_names = ['PILLAR', 'POLE', 'DUMBBELL', 'CONE', 'HOURGLASS']

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



"""def plot_interactive_3d(plot_dict, file_name="plot"):
    args = plot_dict["args"]
    for agent_num, values_for_composition in enumerate(plot_dict["component_data"]):
        if(values_for_composition != {}):
            print("\nAGENT NUM", agent_num)
            for epochs, comp_dict in values_for_composition.items():
                print("EPOCHS", epochs)

                labels = comp_dict["labels"]
                non_zero_mask = comp_dict["non_zero_mask"]
                all_mask = comp_dict["all_mask"]
                vision_zq = comp_dict["vision_zq"]
                touch_zq = comp_dict["touch_zq"]
                command_voice_zq = comp_dict["command_voice_zq"]
                report_voice_zq = comp_dict["report_voice_zq"]
                
                labels_filtered = labels[non_zero_mask]
                all_mask_filtered = all_mask[non_zero_mask]
                vision_zq_filtered = vision_zq[non_zero_mask]
                touch_zq_filtered = touch_zq[non_zero_mask]
                command_voice_zq_filtered = command_voice_zq[non_zero_mask]
                report_voice_zq_filtered = report_voice_zq[non_zero_mask]
                
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
                    
                    episode_indexes = [0]
                    episode_index = 0             
                    for mask in all_mask: 
                        episode_len = int(mask.sum().item())
                        episode_indexes.append(episode_len + episode_index)
                        episode_index = episode_indexes[-1]     
                                                                        
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
                        
                        these_color_probs = color_probs[start_idx:end_idx]
                        pred_colors = np.argmax(these_color_probs, axis = 1)
                        all_pred_colors.append(pred_colors)
                        pc = np.array([item for sublist in all_pred_colors for item in sublist])
                        
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
                    
                    print(jittered_labels.shape, all_pred_tasks.shape, all_pred_colors.shape, all_pred_shapes.shape)
                    axs[0].scatter(jittered_labels[:,0] + 1, all_pred_tasks, c = "black", s = size)
                    axs[0].set_title('Tasks')
                    axs[0].set_xlabel('Real Labels')
                    axs[0].set_ylabel('LDA Predictions')
                    axs[0].set_xticks([2, 3, 4, 5, 6, 7])  
                    axs[0].set_xticklabels(task_names)  
                    axs[0].set_yticks([0, 1, 2, 3, 4, 5]) 
                    axs[0].set_yticklabels(task_names) 
                    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right")
                    plt.setp(axs[0].get_yticklabels(), rotation=0, ha="right")

                    axs[1].scatter(jittered_labels_filtered[:,1], all_pred_colors, c = "black", s = size)
                    axs[1].set_title('Colors')
                    axs[1].set_xlabel('Real Labels')
                    axs[1].set_ylabel('LDA Predictions')
                    axs[1].set_xticks([2, 3, 4, 5, 6, 7])  
                    axs[1].set_xticklabels(color_names)  
                    axs[1].set_yticks([0, 1, 2, 3, 4, 5]) 
                    axs[1].set_yticklabels(color_names) 
                    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")
                    plt.setp(axs[1].get_yticklabels(), rotation=0, ha="right")

                    axs[2].scatter(jittered_labels_filtered[:,2], all_pred_shapes, c = "black", s = size)
                    axs[2].set_title('Shapes')
                    axs[2].set_xlabel('Real Labels')
                    axs[2].set_ylabel('LDA Predictions')
                    axs[2].set_xticks([2, 3, 4, 5, 6])  
                    axs[2].set_xticklabels(shape_names)  
                    axs[2].set_yticks([0, 1, 2, 3, 4]) 
                    axs[2].set_yticklabels(shape_names) 
                    plt.setp(axs[2].get_xticklabels(), rotation=45, ha="right")
                    plt.setp(axs[2].get_yticklabels(), rotation=0, ha="right")
                    plt.tight_layout()
                    plt.savefig(output_file_2d)
                    plt.close()
                    
                    print(f"LDA for {args.arg_name} using {type_name} in epoch {epochs} plotted.")
                    
                for type_name, this, this_filtered in [("command_voice_zq", command_voice_zq, command_voice_zq_filtered)]:
                    plot_lda(this, this_filtered, type_name)"""
                    
                    
                    
def plot_dimension_reduction(plot_dict, file_name = "plot"):
    args = plot_dict["args"]
    for agent_num, values_for_composition in enumerate(plot_dict["component_data"]):
        if(values_for_composition != {}):
            print("\nAGENT NUM", agent_num)
            for epochs, comp_dict in values_for_composition.items():
                print("EPOCHS", epochs)

                non_zero_mask = comp_dict["non_zero_mask"]
                all_mask = comp_dict["all_mask"]
                all_mask_filtered = all_mask[non_zero_mask]
                
                labels = comp_dict["labels"]
                labels_filtered = labels[non_zero_mask]
                
                labels = labels.reshape(-1, labels.shape[-1])
                labels = labels[all_mask.reshape(-1).astype(bool)]
                labels_filtered = labels_filtered.reshape(-1, labels_filtered.shape[-1])
                labels_filtered = labels_filtered[all_mask_filtered.reshape(-1).astype(bool)]
                
                vision_zq = comp_dict["vision_zq"]
                vision_zq_filtered = vision_zq[non_zero_mask]
                
                vision_zq = vision_zq.reshape(-1, vision_zq.shape[-1])
                vision_zq = vision_zq[all_mask.reshape(-1).astype(bool)]
                vision_zq_filtered = vision_zq_filtered.reshape(-1, vision_zq_filtered.shape[-1])
                vision_zq_filtered = vision_zq_filtered[all_mask_filtered.reshape(-1).astype(bool)]     
                
                touch_zq = comp_dict["touch_zq"]
                touch_zq_filtered = touch_zq[non_zero_mask]
                
                touch_zq = touch_zq.reshape(-1, touch_zq.shape[-1])
                touch_zq = touch_zq[all_mask.reshape(-1).astype(bool)]
                touch_zq_filtered = touch_zq_filtered.reshape(-1, touch_zq_filtered.shape[-1])
                touch_zq_filtered = touch_zq_filtered[all_mask_filtered.reshape(-1).astype(bool)]       
                
                command_voice_zq = comp_dict["command_voice_zq"]
                command_voice_zq_filtered = command_voice_zq[non_zero_mask]
                
                command_voice_zq = command_voice_zq.reshape(-1, command_voice_zq.shape[-1])
                command_voice_zq = command_voice_zq[all_mask.reshape(-1).astype(bool)]
                command_voice_zq_filtered = command_voice_zq_filtered.reshape(-1, command_voice_zq_filtered.shape[-1])
                command_voice_zq_filtered = command_voice_zq_filtered[all_mask_filtered.reshape(-1).astype(bool)]  
                    
                report_voice_zq = comp_dict["report_voice_zq"]
                report_voice_zq_filtered = report_voice_zq[non_zero_mask]
                
                report_voice_zq = report_voice_zq.reshape(-1, report_voice_zq.shape[-1])
                report_voice_zq = report_voice_zq[all_mask.reshape(-1).astype(bool)]
                report_voice_zq_filtered = report_voice_zq_filtered.reshape(-1, report_voice_zq_filtered.shape[-1])
                report_voice_zq_filtered = report_voice_zq_filtered[all_mask_filtered.reshape(-1).astype(bool)]  
                
                tasks = labels_filtered[:, 0]
                colors = labels_filtered[:, 1]
                shapes = labels_filtered[:, 2]
                unique_tasks = np.unique(tasks)
                unique_colors = np.unique(colors)
                unique_shapes = np.unique(shapes)
                
                task_mapping = {
                    'WATCH': '#FF0000',
                    'BE NEAR': '#00FF00',
                    'TOUCH THE TOP': '#0000FF',
                    'PUSH FORWARD': '#00FFFF',
                    'PUSH LEFT': '#FF00FF',
                    'PUSH RIGHT': '#FFFF00'}
                
                color_mapping = {
                    'RED': '#FF0000',
                    'GREEN': '#00FF00',
                    'BLUE': '#0000FF',
                    'CYAN': '#00FFFF',
                    'MAGENTA': '#FF00FF',
                    'YELLOW': '#FFFF00'}
                
                shape_mapping = {
                    'PILLAR': '#FF0000',
                    'POLE': '#00FF00',
                    'DUMBBELL': '#0000FF',
                    'CONE': '#00FFFF',
                    'HOURGLASS': '#FF00FF'}
                
                
                
                pca = PCA(n_components=2, random_state=42)
                isomap = Isomap(n_components=2, n_neighbors=10)  # You can tune `n_neighbors`

                
                
                def plot_reduction(this, this_filtered, type_name, reducer, method_name):
                    try:
                        os.makedirs(f"thesis_pics/composition/{args.arg_name}/{method_name}/{type_name}", exist_ok=True)
                    except Exception as e:
                        print(f"Directory creation failed: {e}")          

                    this_2d = reducer.fit_transform(this)

                    output_file = f"thesis_pics/composition/{args.arg_name}/{method_name}/{type_name}/{file_name}_epoch_{epochs}_{type_name}_agent_num_{agent_num}_{args.arg_name}_{method_name}"
                    output_file_2d = output_file + ".png"

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, constrained_layout=True)

                    def plot_by_attribute(ax, data_2d, labels, unique_labels, label_map, color_mapping, title):
                        for label in unique_labels:
                            idx = labels == label
                            ax.scatter(
                                data_2d[idx, 0], data_2d[idx, 1],
                                label=label_map[label].name,
                                alpha=0.6,
                                color=color_mapping[label_map[label].name]
                            )
                        ax.set_title(title)
                        ax.set_xlabel("Component 1")
                        if title == "Task":
                            ax.set_ylabel("Component 2")
                        ax.legend(
                            title=title,
                            fontsize=8,
                            title_fontsize=8,
                            loc="upper left",
                            bbox_to_anchor=(1.02, 1),
                            borderaxespad=0,
                            ncol=1,
                            handletextpad=0.4,
                            columnspacing=0.7,
                            borderpad=0.3,
                            markerscale=0.8
                        )
                        ax.grid(True)

                    plot_by_attribute(axes[0], this_2d, tasks, unique_tasks, task_map, task_mapping, "Task")
                    plot_by_attribute(axes[1], this_2d, colors, unique_colors, color_map, color_mapping, "Color")
                    plot_by_attribute(axes[2], this_2d, shapes, unique_shapes, shape_map, shape_mapping, "Shape")

                    plt.suptitle(f"{method_name.upper()} of Command-Voice Posterior z_q\nAgent {agent_num}, epoch {epochs}", fontsize=16)
                    plt.savefig(output_file_2d, bbox_inches="tight")
                    plt.close()
                    
                    
                    
                for type_name, this, this_filtered in [("command_voice_zq", command_voice_zq, command_voice_zq_filtered)]:
                    plot_reduction(this, this_filtered, type_name, pca, "pca")
                    #plot_reduction(this, this_filtered, type_name, isomap, "isomap")
                    


def plots(plot_dicts):
    for i, plot_dict in enumerate(plot_dicts):
        args = plot_dict["args"]
        try: os.mkdir(f"thesis_pics/composition/{args.arg_name}")
        except: pass
        print(f"\nStarting {args.arg_name}.")
        
        #plot_interactive_3d(plot_dict)
        plot_dimension_reduction(plot_dict)

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