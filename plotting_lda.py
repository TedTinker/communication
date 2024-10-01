#%%
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
import plotly.graph_objects as go

from utils import args, duration, load_dicts, print, action_map, color_map, shape_map

print("name:\n{}\n".format(args.arg_name),)

dpi = 50

action_names = ['FREEPLAY', 'WATCH', 'PUSH', 'PULL', 'LEFT', 'RIGHT']
color_names = ['RED', 'GREEN', 'BLUE', 'CYAN', 'PINK', 'YELLOW']
shape_names = ['PILLAR', 'POLE', 'DUMBBELL', 'DELTA', 'HOURGLASS']



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



def plot_interactive_3d(plot_dict, file_name="lda_plot.html"):
    """
    Creates an interactive 3D plot and saves it as an HTML file for click-and-drag navigation.

    Parameters:
    plot_dict (dict): Dictionary containing the data for plotting.
    file_name (str): Name of the output HTML file.
    """
    for agent_num, agent in enumerate(plot_dict["lda_transformations"]):
        if(agent != {}):
            print("AGENT NUM", agent_num)

            for epochs, task_tensors in agent.items():
                
                print("EPOCHS", epochs)
                fig = go.Figure()
                
                for task_name, (action_probs, color_probs, shape_probs) in task_tensors.items():
                                
                    actions = np.argmax(action_probs, axis=1)
                    colors = np.argmax(color_probs, axis=1)
                    shapes = np.argmax(shape_probs, axis=1)
                                    
                    actions = np.array(actions).flatten()
                    colors = np.array(colors).flatten()
                    shapes = np.array(shapes).flatten()
                                    
                    fig.add_trace(go.Scatter3d(
                        x=actions, 
                        y=colors, 
                        z=shapes, 
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
                            tickvals=[0, 1, 2, 3, 4, 5],  # Corresponding numeric values
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

                # Save the plot as an interactive HTML file
                output_file = f"thesis_pics/lda/{file_name}_epoch_{epochs}_agent_num_{agent_num}_{plot_dict['arg_name']}.html"
                fig.write_html(output_file)

                print(f"Interactive plot saved as {output_file}")



def plots(plot_dicts):
    try: os.mkdir("thesis_pics/lda")
    except: pass
  
    for i, plot_dict in enumerate(plot_dicts):
        print(f"\nStarting {plot_dict['arg_name']}.")
        
        plot_interactive_3d(plot_dict)

        print(f"Finished {plot_dict['arg_name']}.")
        print(f"Duration: {duration()}")
        
    print(f"\tFinished LDA.")

    
    
plot_dicts, min_max_dict, complete_order = load_dicts(args)
plots(plot_dicts)
print(f"\nDuration: {duration()}. Done!")
# %%
