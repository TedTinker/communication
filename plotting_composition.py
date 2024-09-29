#%%
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from utils import args, duration, load_dicts, print, action_map, color_map, shape_map

print("name:\n{}\n".format(args.arg_name),)

dpi = 50

action_names = ['FREEPLAY', 'WATCH', 'PUSH', 'PULL', 'LEFT', 'RIGHT']
color_names = ['RED', 'GREEN', 'BLUE', 'CYAN', 'PINK', 'YELLOW']
shape_names = ['PILLAR', 'POLE', 'DUMBBELL', 'DELTA', 'HOURGLASS']

lda_action = LDA(n_components = len(action_names) - 1)
lda_color = LDA(n_components = len(color_names) - 1)
lda_shape = LDA(n_components = len(shape_names) - 1)



def color_based_on_title(task_name):    
    action_str, color_str, shape_str = task_name.split('_')
    action_index = action_names.index(action_str)
    color_index = color_names.index(color_str)
    shape_index = shape_names.index(shape_str)
    
    action_normalized = action_index / (len(action_names) - 1)
    color_normalized = color_index / (len(color_names) - 1)
    shape_normalized = shape_index / (len(shape_names) - 1)
    
    r = int(action_normalized * 255)
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
    for agent_num, values_for_composition in enumerate(plot_dict["values_for_composition"]):
        if(values_for_composition != {}):
            print("\nAGENT NUM", agent_num)
            for epochs, (hq, zp_zq_dkls, labels, all_mask) in values_for_composition.items():
                print("EPOCHS", epochs)
                
                freeplay_mask_for_mask = labels[:, 0, 0] != 1
                labels = labels.view(-1, 3)
                labels = labels[all_mask.view(-1).bool()].detach().cpu().numpy()  
                freeplay_mask = labels[:, 0] != 1
                    
                """
                print("1", hq.shape)
                print("2", zp_zq_dkls["father_comm"].zp.shape)
                print("3", zp_zq_dkls["father_comm"].zq.shape)
                print("4", zp_zq_dkls["father_comm"].dkl.shape)
                print("5", all_mask.shape)
                print("6", labels)
                """
                
                
                                
                def plot_lda(this, type_name, labels = labels):
                    
                    try: os.mkdir(f"thesis_pics/composition/{args.arg_name}/lda")
                    except: pass
                    try: os.mkdir(f"thesis_pics/composition/{args.arg_name}/lda/{type_name}")
                    except: pass
                    
                    # PROBLEM: NEED TO TRACK FREE-PLAY! 
                    all_task_names = plot_dict["all_task_names"]
                    
                    episode_indexes = [0]
                    episode_index = 0
                    for task_name, mask in zip(all_task_names, all_mask): 
                        episode_len = int(mask.sum().item())
                        episode_indexes.append(episode_len + episode_index)
                        episode_index = episode_indexes[-1]
                        
                    all_task_names_no_free_play = [task_name for task_name in all_task_names if task_name.split("_")[0] != "FREEPLAY"]
                    all_mask_no_free_play = all_mask[freeplay_mask_for_mask]
                    
                    episode_indexes_no_free_play = [0]
                    episode_index_no_free_play = 0
                    for task_name, mask in zip(all_task_names_no_free_play, all_mask_no_free_play): 
                        episode_len = int(mask.sum().item())
                        episode_indexes_no_free_play.append(episode_len + episode_index_no_free_play)
                        episode_index_no_free_play = episode_indexes_no_free_play[-1]
                                                
                    this = this.view(-1, this.shape[-1])
                    this = this[all_mask.view(-1).bool()].detach().cpu().numpy()
                    this_no_free_play = this[freeplay_mask]
                    labels_no_free_play = labels[freeplay_mask]
                            
                    lda_action.fit(this, labels[:,0])
                    lda_color.fit(this_no_free_play, labels_no_free_play[:,1])
                    lda_shape.fit(this_no_free_play, labels_no_free_play[:,2])
                    
                    action_probs = lda_action.predict_proba(this)
                    color_probs = lda_color.predict_proba(this_no_free_play)
                    shape_probs = lda_shape.predict_proba(this_no_free_play)
                                                            
                    all_pred_actions = []
                    all_pred_colors = []
                    all_pred_shapes = []
                    
                    fig = go.Figure()
                
                    for i, task_name in enumerate(all_task_names):
                        start_idx = episode_indexes[i]
                        end_idx = episode_indexes[i + 1]
                        these_action_probs = action_probs[start_idx:end_idx]
                        pred_actions = np.argmax(these_action_probs, axis = 1)
                        all_pred_actions.append(pred_actions)
                        
                    for i, task_name in enumerate(all_task_names_no_free_play):               
                        start_idx = episode_indexes_no_free_play[i]
                        end_idx = episode_indexes_no_free_play[i + 1]
                        
                        these_color_probs = color_probs[start_idx:end_idx]
                        pred_colors = np.argmax(these_color_probs, axis = 1)
                        all_pred_colors.append(pred_colors)
                        
                        these_shape_probs = shape_probs[start_idx:end_idx]        
                        pred_shapes = np.argmax(these_shape_probs, axis = 1)
                        all_pred_shapes.append(pred_shapes)
                                            
                    non_freeplay_index = 0
                    for i, task_name in enumerate(all_task_names):
                        if(task_name.split("_")[0] != "FREEPLAY"):
                            
                            pred_actions = all_pred_actions[i]
                            pred_colors = all_pred_colors[non_freeplay_index]
                            pred_shapes = all_pred_shapes[non_freeplay_index]
                            non_freeplay_index += 1
                                                    
                            fig.add_trace(go.Scatter3d(
                                x=pred_actions, 
                                y=pred_colors, 
                                z=pred_shapes, 
                                mode='markers',
                                marker=dict(
                                    size=8,  # Marker size
                                    color=color_based_on_title(task_name),  # Colors can be any valid sequence
                                    opacity=0.8
                                ),
                            showlegend=False 
                            ))

                    # Set labels and title
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(
                                title='Actions', 
                                tickvals=[0, 1, 2, 3, 4],  # Corresponding numeric values
                                ticktext=action_names,          # Corresponding action labels
                                range=[-1, 6]
                            ),
                            yaxis=dict(
                                title='Colors', 
                                tickvals=[0, 1, 2, 3, 4, 5],  # Corresponding numeric values
                                ticktext=color_names,               # Corresponding color labels
                                range=[-1, 6]
                            ),
                            zaxis=dict(
                                title='Shapes', 
                                tickvals=[0, 1, 2, 3, 4],  # Corresponding numeric values
                                ticktext=shape_names,           # Corresponding shape labels
                                range=[-1, 5]
                            )
                        ),
                        title=f"LDA after {epochs} epochs",
                        autosize=True,
                        height=800,
                        width=1000,
                    )
                    
                    output_file = f"thesis_pics/composition/{args.arg_name}/lda/{type_name}/{file_name}_epoch_{epochs}_{type_name}_agent_num_{agent_num}_{args.arg_name}_lda"
                    output_file_2d = output_file + ".png"
                    output_file_3d = output_file + "_3d.html"
                    fig.write_html(output_file_3d)
                    
                    all_pred_actions = np.array([item for sublist in all_pred_actions for item in sublist])
                    all_pred_colors = np.array([item for sublist in all_pred_colors for item in sublist])
                    all_pred_shapes = np.array([item for sublist in all_pred_shapes for item in sublist])
                    
                    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
                    fig.suptitle(f'LDA for {args.arg_name} using {type_name} in epoch {epochs}', fontsize=16)
                    
                    size = 1 
                    jitter = .3
                    
                    labels = add_jitter(labels, jitter)
                    labels_no_free_play = add_jitter(labels_no_free_play, jitter)
                    all_pred_actions = add_jitter(all_pred_actions, jitter)
                    all_pred_colors = add_jitter(all_pred_colors, jitter)
                    all_pred_shapes = add_jitter(all_pred_shapes, jitter)
                    
                    axs[0].scatter(labels[:,0], all_pred_actions, c = "black", s = size)
                    axs[0].set_title('Actions')
                    axs[0].set_xlabel('Real Labels')
                    axs[0].set_ylabel('LDA Predictions')
                    axs[0].set_xticks([1, 2, 3, 4, 5, 6])  
                    axs[0].set_xticklabels(action_names)  
                    axs[0].set_yticks([0, 1, 2, 3, 4, 5]) 
                    axs[0].set_yticklabels(action_names) 
                    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right")
                    plt.setp(axs[0].get_yticklabels(), rotation=0, ha="right")

                    axs[1].scatter(labels_no_free_play[:,1], all_pred_colors, c = "black", s = size)
                    axs[1].set_title('Colors')
                    axs[1].set_xlabel('Real Labels')
                    axs[1].set_ylabel('LDA Predictions')
                    axs[1].set_xticks([1, 2, 3, 4, 5, 6])  
                    axs[1].set_xticklabels(color_names)  
                    axs[1].set_yticks([0, 1, 2, 3, 4, 5]) 
                    axs[1].set_yticklabels(color_names) 
                    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")
                    plt.setp(axs[1].get_yticklabels(), rotation=0, ha="right")

                    axs[2].scatter(labels_no_free_play[:,2], all_pred_shapes, c = "black", s = size)
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
                    
                    
                    
                def plot_pca(this, type_name, labels=labels):
                    
                    try: os.mkdir(f"thesis_pics/composition/{args.arg_name}/pca")
                    except: pass
                    try: os.mkdir(f"thesis_pics/composition/{args.arg_name}/pca/{type_name}")
                    except: pass
                    
                    all_task_names = plot_dict["all_task_names"]
                    episode_indexes = [0]
                    episode_index = 0
                    for task_name, mask in zip(all_task_names, all_mask): 
                        episode_len = int(mask.sum().item())
                        episode_indexes.append(episode_len + episode_index)
                        episode_index = episode_indexes[-1]
                    
                    # Reshape and filter the data
                    this = this.view(-1, this.shape[-1])
                    this = this[all_mask.view(-1).bool()].detach().cpu().numpy()
                    
                    # Apply PCA
                    pca = PCA(n_components=3)
                    pca_result = pca.fit_transform(this)
                    
                    # Separate the PCA components
                    pca_action = pca_result[:, 0]  # First principal component
                    pca_color = pca_result[:, 1]   # Second principal component
                    pca_shape = pca_result[:, 2]   # Third principal component
                    
                    fig = go.Figure()

                    for i, task_name in enumerate(all_task_names):
                        
                        start_idx = episode_indexes[i]
                        end_idx = episode_indexes[i + 1]
                        
                        these_action_pca = pca_action[start_idx:end_idx]
                        these_color_pca = pca_color[start_idx:end_idx]
                        these_shape_pca = pca_shape[start_idx:end_idx]
                        
                        fig.add_trace(go.Scatter3d(
                            x=these_action_pca, 
                            y=these_color_pca, 
                            z=these_shape_pca, 
                            mode='markers',
                            marker=dict(
                                size=8,  # Marker size
                                color=color_based_on_title(task_name),  # Colors can be any valid sequence
                                opacity=0.8
                            ),
                            showlegend=False 
                        ))

                    # Set labels and title
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(
                                title='PCA 1', 
                                range=[pca_action.min()-1, pca_action.max()+1]
                            ),
                            yaxis=dict(
                                title='PCA 2', 
                                range=[pca_color.min()-1, pca_color.max()+1]
                            ),
                            zaxis=dict(
                                title='PCA 3', 
                                range=[pca_shape.min()-1, pca_shape.max()+1]
                            )
                        ),
                        title=f"PCA after {epochs} epochs",
                        autosize=True,
                        height=800,
                        width=1000,
                    )
                    
                    output_file = f"thesis_pics/composition/{args.arg_name}/pca/{type_name}/{file_name}_epoch_{epochs}_{type_name}_agent_num_{agent_num}_{args.arg_name}"
                    output_file_2d = output_file + ".png"
                    output_file_3d = output_file + "_3d.html"
                    fig.write_html(output_file_3d)
                    
                    
                    
                    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
                    fig.suptitle(f'PCA for {args.arg_name} using {type_name} in epoch {epochs}', fontsize=16)
                    
                    size = 1 
                    
                    axs[0].scatter(pca_action, [0] * len(pca_action), c = "black", s = size)
                    axs[0].set_title('Actions')

                    axs[1].scatter(pca_color, [0] * len(pca_color), c = "black", s = size)
                    axs[1].set_title('Colors')

                    axs[2].scatter(pca_shape, [0] * len(pca_color), c = "black", s = size)
                    axs[2].set_title('Shape')

                    plt.tight_layout()
                    plt.savefig(output_file_2d)
                    plt.close()
                    
                    print(f"PCA for {args.arg_name} using {type_name} in epoch {epochs} plotted.")
                    
                    
                    
                print("MAKING LDA")
                for this, type_name in [
                    (hq, "hq"), 
                    (zp_zq_dkls["father_comm"].zp, "father_comm_zp"),
                    (zp_zq_dkls["father_comm"].zq, "father_comm_zq")]:
                    plot_lda(this, type_name)
                    
                #print("MAKING PCA")
                #for this, type_name in [
                #    (hq, "hq"), 
                #    (zp_zq_dkls["father_comm"].zp, "father_comm_zp"),
                #    (zp_zq_dkls["father_comm"].zq, "father_comm_zq")]:
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