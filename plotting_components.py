#%%
import os
import random
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
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
from collections import defaultdict

from utils import testing_combos, args, duration, load_dicts, print, task_map, color_map, shape_map
from utils_submodule import  init_weights


print("name:\n{}\n".format(args.arg_name),)

epochs_for_classification = 50



task_mapping_color = {
    'WATCH':            '#FF0000',  # Red
    'BE NEAR':          '#00FF00',  # Green
    'TOUCH THE TOP':    '#0000FF',  # Blue
    'PUSH FORWARD':     '#00DDDD',  # Cyan
    'PUSH LEFT':        '#FF00FF',  # Magenta
    'PUSH RIGHT':       '#DDDD00'}  # Yellow

task_mapping_letter = {
    'WATCH':            "W",
    'BE NEAR':          "B",
    'TOUCH THE TOP':    "T",
    'PUSH FORWARD':     "F",
    'PUSH LEFT':        "L",
    'PUSH RIGHT':       "R"}

color_mapping_color = {
    'RED':              '#FF0000',           
    'GREEN':            '#00FF00',
    'BLUE':             '#0000FF',
    'CYAN':             '#00DDDD',
    'MAGENTA':          '#FF00FF',
    'YELLOW':           '#DDDD00'}

def darken_hex_color(hex_color, factor=0.8):
    """
    Darkens the given hex color by the specified factor.
    Factor should be between 0 (black) and 1 (no change).
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    return f'#{r:02X}{g:02X}{b:02X}'

color_mapping_color_dark = {name: darken_hex_color(hex_code) for name, hex_code in color_mapping_color.items()}

shape_mapping_color = {
    'PILLAR':           '#FF0000',  # Red
    'POLE':             '#00FF00',  # Green
    'DUMBBELL':         '#0000FF',  # Blue
    'CONE':             '#00DDDD',  # Cyan
    'HOURGLASS':        '#FF00FF'}  # Magenta

shape_mapping_marker = {
    'PILLAR':           mpimg.imread('pybullet_data/shapes/pillar.png'),   
    'POLE':             mpimg.imread('pybullet_data/shapes/pole.png'),   
    'DUMBBELL':         mpimg.imread('pybullet_data/shapes/dumbbell.png'),    
    'CONE':             mpimg.imread('pybullet_data/shapes/cone.png'),   
    'HOURGLASS':        mpimg.imread('pybullet_data/shapes/hourglass.png')}   

"""shape_mapping_marker = {
    'PILLAR':           'o',   
    'POLE':             's',   
    'DUMBBELL':         'P',    
    'CONE':             '^',   
    'HOURGLASS':        'X'}   """
        
    
    
def colorize_marker_image(marker_img, hex_color, alpha = .3):
    hex_color = hex_color.lstrip('#')
    target_rgb = np.array([
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0])
    colorized = marker_img.copy()
    mask = colorized[..., 3] > 0 
    colorized[mask, 0] = target_rgb[0]
    colorized[mask, 1] = target_rgb[1]
    colorized[mask, 2] = target_rgb[2]
    colorized[mask, 3] = alpha
    return colorized

shape_mapping_colored_marker = {}
for shape_name, shape_marker in shape_mapping_marker.items():
    shape_mapping_colored_marker[shape_name] = {}
    shape_mapping_colored_marker[shape_name]["BLACK"] = colorize_marker_image(shape_marker, "#000000")
    for color_name, color_color in color_mapping_color.items():
        shape_mapping_colored_marker[shape_name][color_name] = colorize_marker_image(shape_marker, color_color)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        

class Classifier(nn.Module):

    def __init__(self, data):
        super(Classifier, self).__init__()
        
        data_shape = data.shape[1]
        
        self.seq = nn.Sequential(
            nn.Linear(
                in_features = data_shape, 
                out_features = data_shape),
            nn.PReLU(),
            nn.Linear(
                in_features = data_shape, 
                out_features = data_shape),
            nn.PReLU(),
            nn.Linear(
                in_features = data_shape, 
                out_features = 32))
        self.task  = nn.Linear(32, 6)
        self.color = nn.Linear(32, 6)
        self.shape = nn.Linear(32, 5)
        self.apply(init_weights)

    def forward(self, data):
        embedding = self.seq(data)
        return(embedding, self.task(embedding), self.color(embedding), self.shape(embedding))
    
    
    
def train_to_classify_2d(data, labels, epochs, method):
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    
    reduced_dict = {}
    
    for classes in [("task", "color"), ("task", "shape"), ("color", "shape")]:    
        print(classes)
        set_seed(42)
        classifier = Classifier(data)
        optimizer = optim.Adam(classifier.parameters(), lr=.001)
        classifier.train()
        for e in range(epochs):
            print(e, end = " ")
            optimizer.zero_grad()
            _, task, color, shape = classifier(data)
            task_loss  = F.cross_entropy(task, labels[:, 0] - 1)    * (1 if "task" in classes else 0)
            color_loss = F.cross_entropy(color, labels[:, 1])       * (1 if "color" in classes else 0)
            shape_loss = F.cross_entropy(shape, labels[:, 2])       * (1 if "shape" in classes else 0)
            loss = task_loss + color_loss + shape_loss
            loss.backward()
            optimizer.step()
        print()
        classifier.eval()
        embedding, _, _, _ = classifier(data)
        if(method == "pca"):
            reduced = PCA(n_components=2, random_state=42).fit_transform(embedding.detach().cpu().numpy())
        if(method == "tsne"):
            reduced = TSNE(n_components=2, perplexity=10, random_state=42, learning_rate="auto", init = "pca").fit_transform(embedding.detach().cpu().numpy())
        if(method == "umap"):
            pass # PLEEEAAASE!
        
        reduced_dict["_".join(classes)] = reduced
        
    return(reduced_dict)



def train_to_classify_3d(data, labels, epochs, method):
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    classifier = Classifier(data)
    optimizer = optim.Adam(classifier.parameters(), lr=.001)
    classifier.train()
    for e in range(epochs):
        print(e, end = " ")
        optimizer.zero_grad()
        _, task, color, shape = classifier(data)
        task_loss  = F.cross_entropy(task, labels[:, 0] - 1)
        color_loss = F.cross_entropy(color, labels[:, 1])
        shape_loss = F.cross_entropy(shape, labels[:, 2])
        loss = task_loss + color_loss + shape_loss
        loss.backward()
        optimizer.step()
    print()
    classifier.eval()
    embedding, _, _, _ = classifier(data)
    if(method == "pca"):
        reduced = PCA(n_components=3, random_state=42).fit_transform(embedding.detach().cpu().numpy())
    if(method == "tsne"):
        reduced = TSNE(n_components=3, perplexity=10, random_state=42).fit_transform(embedding.detach().cpu().numpy())
    if(method == "umap"):
        pass # PLEEEAAASE!
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
                is_test = [tuple(label) in testing_combos for label in results["labels"]] # Not sure if this is right.
                unique_tasks = np.unique(tasks)
                unique_colors = np.unique(colors)
                unique_shapes = np.unique(shapes)
                
                
                
                def plot_2d_reduction(type_name, method_name, method):
                    
                    try:
                        os.makedirs(f"thesis_pics/composition/{args.arg_name}/{method_name}/{type_name}", exist_ok=True)
                    except Exception as e:
                        print(f"Directory creation failed: {e}")     
                        
                    data = results[type_name] 
                    reduced_dict = train_to_classify_2d(data, results["labels"], epochs_for_classification, method)
                    
                    reduced_task_color  = reduced_dict["task_color"]
                    reduced_task_shape  = reduced_dict["task_shape"]
                    reduced_color_shape = reduced_dict["color_shape"]

                    dpi = 400  
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi = dpi, sharex=False, sharey=False, constrained_layout=True)

                    def plot_by_attribute(ax, reduced, title):
                        min_x = min([r[0] for r in reduced])
                        max_x = max([r[0] for r in reduced])
                        min_y = min([r[1] for r in reduced])
                        max_y = max([r[1] for r in reduced])
                        
                        s = 20
                        test_s = 60
                                                
                        test_coords = [point for is_t, point in zip(is_test, reduced) if is_t]
                        if test_coords:
                            test_xs, test_ys = zip(*test_coords)  # unzip x and y values
                            ax.scatter(test_xs, test_ys, facecolors='none', edgecolors="#000000", linewidths=0.5, marker='o', alpha=.5, s=test_s, zorder=0, linestyle='dotted')
                            
                        grouped_points = defaultdict(list)
                        grouped_letters = {}
                        
                        for task, color, shape, (x, y) in zip(tasks, colors, shapes, reduced):
                            key = (task_map[task].name, color_map[color].name, shape_map[shape].name)
                            grouped_points[key].append((x, y))
                            if key not in grouped_letters:
                                grouped_letters[key] = task_mapping_letter[task_map[task].name]
                        
                        for (task_name, color_name, shape_name), coords in grouped_points.items():
                            letter = grouped_letters[(task_name, color_name, shape_name)]
                            text_color_val = color_mapping_color_dark[color_name]
                            color_val = color_mapping_color[color_name]
                            marker = shape_mapping_marker[shape_name]
                            xs, ys = zip(*coords)  # unzip x and y values
                            
                            

                            if(title == "Task and Color"):
                                ax.scatter(xs, ys, color=text_color_val, marker=f'${letter}$', alpha=0.4, s=s, edgecolor='none', zorder=2)

                            if(title == "Task and Shape"):
                                for (x, y) in zip(xs, ys):
                                    colored_marker = shape_mapping_colored_marker[shape_name]["BLACK"]
                                    imagebox = OffsetImage(colored_marker, zoom=0.2)  # Adjust zoom as needed
                                    ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
                                    ax.add_artist(ab)
                                ax.scatter(xs, ys, color="black", marker=f'${letter}$', alpha=0.4, s=s, edgecolor='none', zorder=2)
                                
                            if(title == "Color and Shape"):
                                for (x, y) in zip(xs, ys):
                                    colored_marker = shape_mapping_colored_marker[shape_name][color_name]
                                    imagebox = OffsetImage(colored_marker, zoom=0.2)  # Adjust zoom as needed
                                    ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
                                    ax.add_artist(ab)
                                
                            
                                       
                        if title == "Color and Shape":
                            legend_elements = []

                            for task in unique_tasks:
                                name = task_map[task].name
                                letter = task_mapping_letter[name]
                                legend_elements.append(Line2D([0], [0], marker=f'${letter}$', linestyle='None', 
                                                            label=f"{name}",
                                                            color='black'))

                            for color in unique_colors:
                                name = color_map[color].name
                                color_val = color_mapping_color[name]
                                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                            label=f"{name}",
                                                            markerfacecolor=color_val, markersize=6))

                            for shape in unique_shapes:
                                name = shape_map[shape].name
                                marker = shape_mapping_marker[name]
                                legend_elements.append(Line2D([0], [0], marker="", linestyle='None', color='black', 
                                                            label=f"{name}"))
                                
                            ax.legend(
                                handles=legend_elements,
                                fontsize=8,
                                loc="upper left",
                                bbox_to_anchor=(1.02, 1),
                                borderaxespad=0,
                                ncol=1,
                                handletextpad=0.4,
                                columnspacing=0.7,
                                borderpad=0.3,
                                markerscale=0.8)         
                                            
                        ax.set_title(title)
                        ax.set_xlabel("Component 1")
                        if title == "Task and Color":
                            ax.set_ylabel("Component 2")
                        ax.set_xticks([])      # remove x-axis ticks
                        ax.set_yticks([])      # remove y-axis ticks
                        ax.set_xticklabels([]) # remove x-axis tick labels
                        ax.set_yticklabels([]) # remove y-axis tick labels
                        ax.set_xlim([min_x, max_x])
                        ax.set_ylim([min_y, max_y])
                        ax.grid(False)

                    plot_by_attribute(axes[0], reduced_task_color,  "Task and Color")
                    plot_by_attribute(axes[1], reduced_task_shape,  "Task and Shape")
                    plot_by_attribute(axes[2], reduced_color_shape, "Color and Shape")

                    plt.suptitle(f"{method_name.upper()} of {type_name}\nAgent {agent_num}, epoch {epochs}", fontsize=16)
                    output_file = f"thesis_pics/composition/{args.arg_name}/{method_name}/{type_name}/{file_name}_epoch_{epochs}_{type_name}_agent_num_{agent_num}_{args.arg_name}_{method_name}.png"
                    plt.savefig(output_file, bbox_inches="tight")
                    plt.close()
                        
                        
                    
                def plot_3d_reduction(type_name, method_name, method):
                    
                    try:
                        os.makedirs(f"thesis_pics/composition/{args.arg_name}/{method_name}/{type_name}", exist_ok=True)
                    except Exception as e:
                        print(f"Directory creation failed: {e}")
                        
                    data = results[type_name]
                    reduced = train_to_classify_3d(data, results["labels"], epochs_for_classification, method)
                    
                    x, y, z = reduced[:, 0], reduced[:, 1], reduced[:, 2]
                    trace_list = []

                    def get_colors(labels, label_map, mapping, unique_labels, transparent = False, verbose = False):
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
                    trace_task  = make_trace("Task",  tasks,  task_map,  task_mapping_color,  unique_tasks,  visible=True)
                    trace_color = make_trace("Color", colors, color_map, color_mapping_color, unique_colors, visible=False, verbose = True)
                    trace_shape = make_trace("Shape", shapes, shape_map, shape_mapping_color, unique_shapes, visible=False)

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

                    trace_list += make_legend_traces(task_map,  task_mapping_color,  'Task')
                    trace_list += make_legend_traces(color_map, color_mapping_color, 'Color')
                    trace_list += make_legend_traces(shape_map, shape_mapping_color, 'Shape')

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
                    plot_2d_reduction(type_name, "pca_nn", "pca")
                    #plot_3d_reduction(type_name, "pca_3d_nn", "pca")
                    #plot_3d_reduction(type_name, "tsne_3d_nn", "tsne")
                    


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