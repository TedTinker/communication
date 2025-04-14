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
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn 
import torch.optim as optim
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from utils import args, duration, load_dicts, print, task_map, color_map, shape_map



print("name:\n{}\n".format(args.arg_name),)

epochs_for_classification = 50



task_mapping = {
    'WATCH': '#FF0000',         # Red
    'BE NEAR': '#00FF00',       # Green
    'TOUCH THE TOP': '#0000FF', # Blue
    'PUSH FORWARD': '#00FFFF',  # Cyan
    'PUSH LEFT': '#FF00FF',     # Magenta
    'PUSH RIGHT': '#FFFF00'}    # Yellow

color_mapping = {
    'RED': '#FF0000',           
    'GREEN': '#00FF00',
    'BLUE': '#0000FF',
    'CYAN': '#00FFFF',
    'MAGENTA': '#FF00FF',
    'YELLOW': '#FFFF00'}

shape_mapping = {
    'PILLAR': '#FF0000',        # Red
    'POLE': '#00FF00',          # Greed
    'DUMBBELL': '#0000FF',      # Blue
    'CONE': '#00FFFF',          # Cyan
    'HOURGLASS': '#FF00FF'}     # Magenta



class Classifier(nn.Module):

    def __init__(self, this):
        super(Classifier, self).__init__()
        
        this_shape = this.shape[1]
        
        self.seq = nn.Sequential(
            nn.Linear(
                in_features = this_shape, 
                out_features = this_shape),
            nn.PReLU(),
            nn.Linear(
                in_features = this_shape, 
                out_features = this_shape),
            nn.PReLU(),
            nn.Linear(
                in_features = this_shape, 
                out_features = 32))
        self.task  = nn.Linear(32, 6)
        self.color = nn.Linear(32, 6)
        self.shape = nn.Linear(32, 5)

    def forward(self, this):
        embedding = self.seq(this)
        return(embedding, self.task(embedding), self.color(embedding), self.shape(embedding))
    
    
    
def train_to_classify(this, labels, epochs, dim=3):
    this = torch.from_numpy(this)
    labels = torch.from_numpy(labels)
    classifier = Classifier(this)
    optimizer = optim.Adam(classifier.parameters(), lr=.001)
    classifier.train()
    for e in range(epochs):
        print(e, end = " ")
        optimizer.zero_grad()
        _, task, color, shape = classifier(this)
        task_loss  = F.cross_entropy(task, labels[:, 0] - 1)
        color_loss = F.cross_entropy(color, labels[:, 1])
        shape_loss = F.cross_entropy(shape, labels[:, 2])
        loss = task_loss + color_loss + shape_loss
        loss.backward()
        optimizer.step()
    print()
    classifier.eval()
    embedding, _, _, _ = classifier(this)
    reduced = PCA(n_components=dim, random_state=42).fit_transform(embedding.detach().cpu().numpy())
    return(reduced)
    
            
                    
                    
def plot_dimension_reduction(plot_dict, file_name = "plot"):
    args = plot_dict["args"]
    for agent_num, values_for_composition in enumerate(plot_dict["component_data"]):
        if(values_for_composition != {}):
            print("\nAGENT NUM", agent_num)
            for epochs, comp_dict in values_for_composition.items():
                print("EPOCHS", epochs)

                all_mask = comp_dict["all_mask"].astype(bool)                
                max_episode_len = all_mask.shape[1]
                all_mask = all_mask.reshape(-1, all_mask.shape[-1]).squeeze()
                
                one_episode = np.arange(max_episode_len)
                steps = np.broadcast_to(one_episode.reshape(1, max_episode_len, 1), (180, max_episode_len, 1))
                steps = steps.reshape(-1, steps.shape[-1]).squeeze()
                steps = steps[all_mask]
                                
                def process_component(key):
                    data = comp_dict[key]
                    data = data.reshape(-1, data.shape[-1])
                    data = data[all_mask]
                    return data

                components = ["labels", "vision_zq", "touch_zq", "command_voice_zq", "report_voice_zq", "hq"]
                results = {}
                for key in components:
                    results[key] = process_component(key)
                
                tasks = results["labels"][:, 0]
                colors = results["labels"][:, 1]
                shapes = results["labels"][:, 2]
                unique_tasks = np.unique(tasks)
                unique_colors = np.unique(colors)
                unique_shapes = np.unique(shapes)
                

                
                def plot_2d_reduction(type_name, method_name):
                    
                    try:
                        os.makedirs(f"thesis_pics/composition/{args.arg_name}/{method_name}/{type_name}", exist_ok=True)
                    except Exception as e:
                        print(f"Directory creation failed: {e}")     
                        
                    this = results[type_name] 
                    this = train_to_classify(this, results["labels"], epochs_for_classification, dim = 2)

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

                    plot_by_attribute(axes[0], this, tasks,  unique_tasks,  task_map,  task_mapping,  "Task")
                    plot_by_attribute(axes[1], this, colors, unique_colors, color_map, color_mapping, "Color")
                    plot_by_attribute(axes[2], this, shapes, unique_shapes, shape_map, shape_mapping, "Shape")

                    plt.suptitle(f"{method_name.upper()} of {type_name}\nAgent {agent_num}, epoch {epochs}", fontsize=16)
                    output_file = f"thesis_pics/composition/{args.arg_name}/{method_name}/{type_name}/{file_name}_epoch_{epochs}_{type_name}_agent_num_{agent_num}_{args.arg_name}_{method_name}.png"
                    plt.savefig(output_file, bbox_inches="tight")
                    plt.close()
                        
                        
                    
                # For some reason, YELLOW is gray in these.
                def plot_3d_reduction(type_name, method_name):
                    
                    try:
                        os.makedirs(f"thesis_pics/composition/{args.arg_name}/{method_name}/{type_name}", exist_ok=True)
                    except Exception as e:
                        print(f"Directory creation failed: {e}")
                        
                    this = results[type_name]
                    this = train_to_classify(this, results["labels"], epochs_for_classification, dim = 3)
                    
                    x, y, z = this[:, 0], this[:, 1], this[:, 2]
                    trace_list = []

                    def get_colors(labels, label_map, mapping, unique_labels, transparent = False, verbose = False):
                        #for label in labels:
                            #if(verbose):
                            #    print(list(mapping.keys()))
                            #    print(label_map[label].name)
                            #    print(mapping[label_map[label].name] if label in unique_labels else '#D3D3D3')
                        return [mapping[label_map[label].name] if label in unique_labels else '#D3D3D3' for label in labels]

                    def get_opacities(labels, unique_labels):
                        return .7

                    # Create 3D scatter traces
                    def make_trace(label_type, label_data, label_map, mapping, unique_set, visible=True, verbose = False):
                        return go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='markers',
                            visible=visible,
                            marker=dict(
                                size=4,
                                color=get_colors(label_data, label_map, mapping, unique_set, verbose = verbose),
                                opacity=get_opacities(label_data, unique_set)
                            ),
                            text=[f"{label_type}: {label_map[l].name}" for l in label_data],
                            hoverinfo='text',
                            name=label_type
                        )

                    # Main data traces
                    trace_task  = make_trace("Task",  tasks,  task_map,  task_mapping,  unique_tasks,  visible=True)
                    trace_color = make_trace("Color", colors, color_map, color_mapping, unique_colors, visible=False, verbose = True)
                    trace_shape = make_trace("Shape", shapes, shape_map, shape_mapping, unique_shapes, visible=False)

                    trace_list += [trace_task, trace_color, trace_shape]

                    # Add dummy legend traces for the current mode
                    def make_legend_traces(label_map, mapping, current_label_type):
                        return [
                            go.Scatter3d(
                                x=[None], y=[None], z=[None],
                                mode='markers',
                                marker=dict(size=6, color=color),
                                name=key,
                                showlegend=True,
                                visible=(current_label_type == 'Task')  # toggle visibility
                            )
                            for key, color in mapping.items()
                        ]

                    trace_list += make_legend_traces(task_map,  task_mapping,  'Task')
                    trace_list += make_legend_traces(color_map, color_mapping, 'Color')
                    trace_list += make_legend_traces(shape_map, shape_mapping, 'Shape')

                    # Buttons to toggle between modes
                    view_buttons = [
                        dict(label="Task", method="update", args=[{"visible": [True, False, False] + [True]*6 + [False]*6 + [False]*5},
                                                                    {"title": "3D PCA - Task"}]),
                        dict(label="Color", method="update", args=[{"visible": [False, True, False] + [False]*6 + [True]*6 + [False]*5},
                                                                    {"title": "3D PCA - Color"}]),
                        dict(label="Shape", method="update", args=[{"visible": [False, False, True] + [False]*6 + [False]*6 + [True]*5},
                                                                    {"title": "3D PCA - Shape"}]),
                    ]

                    # Build the figure
                    fig = go.Figure(data=trace_list)

                    fig.update_layout(
                        updatemenus=[
                            dict(
                                type="buttons",
                                direction="right",
                                buttons=view_buttons,
                                x=0.1,
                                y=1.2,
                                showactive=True
                            )
                        ],
                        scene=dict(
                            xaxis_title="PC1",
                            yaxis_title="PC2",
                            zaxis_title="PC3"
                        ),
                        title=f"3D PCA of {type_name} | Agent {agent_num}, Epoch {epochs}"
                    )

                    # Save to HTML
                    output_file = (
                        f"thesis_pics/composition/{args.arg_name}/{method_name}/{type_name}/"
                        f"{file_name}_epoch_{epochs}_{type_name}_agent_num_{agent_num}_{args.arg_name}_{method_name}_3d.html")
                    fig.write_html(output_file)

                                    
                                    
                for type_name in ["hq"]:
                    #plot_2d_reduction(type_name, "pca_nn")
                    plot_3d_reduction(type_name, "pca_3d_nn")
                    


def plots(plot_dicts):
    for i, plot_dict in enumerate(plot_dicts):
        args = plot_dict["args"]
        try: os.mkdir(f"thesis_pics/composition/{args.arg_name}")
        except: pass
        print(f"\nStarting {args.arg_name}.")        
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