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

epochs_for_classification = 10
lr = .01
weight_decay = .0005
noise_size = .1



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
                out_features = 32))
        self.task  = nn.Linear(32, 6)
        self.color = nn.Linear(32, 6)
        self.shape = nn.Linear(32, 5)
        self.apply(init_weights)

    def forward(self, data):
        embedding = self.seq(data)
        if self.training:
            std = embedding.std(unbiased=False)
            noise = torch.randn_like(embedding) * (noise_size * std)
            embedding = embedding + noise
        return(embedding, self.task(embedding), self.color(embedding), self.shape(embedding))
    
    
    
set_seed(42)
task_color_classifier = Classifier(torch.zeros((1, 256)))
set_seed(42)
task_shape_classifier = Classifier(torch.zeros((1, 256)))
set_seed(42)    
color_shape_classifier = Classifier(torch.zeros((1, 256)))


    
def train_to_classify_2d(data, labels, epochs, classifier_pca_dict, agent_num, args):
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
            
    all_task_losses = {}
    all_color_losses = {}
    all_shape_losses = {}
    all_total_losses = {}

    for classes in [("task", "color"), ("task", "shape"), ("color", "shape")]:
        set_seed(42)
        classifier = Classifier(data)
        optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
        classifier.train()

        task_losses, color_losses, shape_losses, total_losses = [], [], [], []

        for e in range(epochs_for_classification):
            print(e, end=" ")
            optimizer.zero_grad()
            _, task, color, shape = classifier(data)
            task_loss  = F.cross_entropy(task, labels[:, 0] - 1) * (1 if "task" in classes else 0)
            color_loss = F.cross_entropy(color, labels[:, 1])    * (1 if "color" in classes else 0)
            shape_loss = F.cross_entropy(shape, labels[:, 2])    * (1 if "shape" in classes else 0)
            loss = task_loss + color_loss + shape_loss

            task_losses.append(task_loss.item())
            color_losses.append(color_loss.item())
            shape_losses.append(shape_loss.item())
            total_losses.append(loss.item())

            loss.backward()
            optimizer.step()
        print()

        classifier.eval()
        embedding, _, _, _ = classifier(data)
        pca = PCA(n_components=2, random_state=42)
        pca.fit(embedding.detach().cpu().numpy())
        classifier_pca_dict[epochs][classes] = (classifier, pca)

        class_str = "_".join(classes)
        all_task_losses[class_str] = task_losses
        all_color_losses[class_str] = color_losses
        all_shape_losses[class_str] = shape_losses
        all_total_losses[class_str] = total_losses

    # Determine y-axis limits
    all_losses_combined = list(all_task_losses.values()) + list(all_color_losses.values()) + list(all_shape_losses.values()) + list(all_total_losses.values())
    y_min = min(min(losses) for losses in all_losses_combined)
    y_max = max(max(losses) for losses in all_losses_combined)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    class_keys = list(all_total_losses.keys())

    for i, class_str in enumerate(class_keys):
        epochs_range = range(epochs_for_classification)
        axs[i].plot(epochs_range, all_task_losses[class_str], label='Task Loss')
        axs[i].plot(epochs_range, all_color_losses[class_str], label='Color Loss')
        axs[i].plot(epochs_range, all_shape_losses[class_str], label='Shape Loss')
        axs[i].plot(epochs_range, all_total_losses[class_str], label='Total Loss', linestyle='--', color='black')
        axs[i].set_title(f"All Losses for {class_str}")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Loss")
        axs[i].set_ylim(-.1, y_max + .1)
    axs[0].legend(
        loc='upper right',
        fontsize='xx-small')

    plt.tight_layout()
    plt.savefig(f"thesis_pics/composition/{args.arg_name}/{agent_num}/loss_values/epochs_{epochs}.png")
    plt.close()

    return classifier_pca_dict


def use_classifier_2d(data, classifier_pca):
    data = torch.from_numpy(data)
    reduced_dict = {}
    for classes in [("task", "color"), ("task", "shape"), ("color", "shape")]:  
        classifier, pca = classifier_pca[classes]
        classifier.eval()
        embedding, _, _, _ = classifier(data)
        reduced = pca.transform(embedding.detach().cpu().numpy())
        reduced_dict["_".join(classes)] = reduced
    return(reduced_dict)



def plot_by_epoch_vs_epoch(data_dict, data_epochs, classifier_pca_dict, classifier_pca_epochs, agent_num, args):
    data, labels = data_dict[data_epochs]
    tasks = labels[:, 0]
    colors = labels[:, 1]
    shapes = labels[:, 2]
    unique_tasks = np.unique(tasks)
    unique_colors = np.unique(colors)
    unique_shapes = np.unique(shapes)
    classifier_pca = classifier_pca_dict[classifier_pca_epochs]
    
    reduced_dict = use_classifier_2d(data, classifier_pca)
    
    reduced_task_color  = reduced_dict["task_color"]
    reduced_task_shape  = reduced_dict["task_shape"]
    reduced_color_shape = reduced_dict["color_shape"]

    dpi = 400  
    zoom = .4
    s = 40
    test_s = 120
    
    fig, axes = plt.subplots(
        1, 4, 
        figsize=(19, 6), 
        dpi=dpi, 
        sharex=False, 
        sharey=False, 
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.4]}  # Make the last (legend) plot thinner
    )

    def plot_by_attribute(ax, reduced, title):
        min_x = min([r[0] for r in reduced])
        max_x = max([r[0] for r in reduced])
        min_y = min([r[1] for r in reduced])
        max_y = max([r[1] for r in reduced])
                                
        grouped_points = defaultdict(list)
        grouped_letters = {}
        
        for task, color, shape, (x, y) in zip(tasks, colors, shapes, reduced):
            key = (task, color, shape)
            grouped_points[key].append((x, y))
            if key not in grouped_letters:
                grouped_letters[key] = task_mapping_letter[task_map[task].name]
        
        for (task, color, shape), coords in grouped_points.items():
            task_name, color_name, shape_name = task_map[task].name, color_map[color].name, shape_map[shape].name
            letter = grouped_letters[(task, color, shape)]
            text_color_val = color_mapping_color_dark[color_name]
            color_val = color_mapping_color[color_name]
            marker = shape_mapping_marker[shape_name]
            xs, ys = zip(*coords)  # unzip x and y values
            x = sum(xs) / len(xs)
            y = sum(ys) / len(ys)
            
            if((task, color, shape) in testing_combos):
                ax.scatter(x, y, facecolors='none', edgecolors="#000000", linewidths=0.5, marker='o', alpha=.5, s=test_s, zorder=0, linestyle='dotted')

            colored_marker = shape_mapping_colored_marker[shape_name][color_name]
            imagebox = OffsetImage(colored_marker, zoom=zoom)  # Adjust zoom as needed
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
            ax.add_artist(ab)
            ax.scatter(x, y, color=text_color_val, marker=f'${letter}$', alpha=0.4, s=s, edgecolor='none', zorder=2)
                            
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
    
    ax = axes[3]
    zoom = .4
    s = 60
    test_s = 120
    
    y = 10
    for task_name, letter in task_mapping_letter.items():
        ax.scatter(.1, y, color="black", marker=f'${letter}$', alpha=0.4, s=s, edgecolor='none')
        ax.text(.3, y, s = task_name, horizontalalignment='left', verticalalignment='center', fontsize = 10)
        y -= .5
        
    for color_name, color in color_mapping_color.items():
        ax.scatter(.1, y, facecolors=color, edgecolors='none', s=test_s, alpha=0.4)
        ax.text(.3, y, s = color_name, horizontalalignment='left', verticalalignment='center', fontsize = 10)
        y -= .5
        
    for shape_name, marker in shape_mapping_marker.items():
        colored_marker = shape_mapping_colored_marker[shape_name]["BLACK"]
        imagebox = OffsetImage(colored_marker, zoom=zoom)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (.1, y), frameon=False, alpha=0.4, zorder=1)
        ax.add_artist(ab)
        ax.text(.3, y, s = shape_name, horizontalalignment='left', verticalalignment='center', fontsize = 10)
        y -= .5
        
    ax.scatter(.1, y, facecolors='none', edgecolors="#000000", linewidths=0.5, marker='o', alpha=.5, s=test_s, zorder=0, linestyle='dotted')
    ax.text(.3, y, s = "TEST", horizontalalignment='left', verticalalignment='center', fontsize = 10)
        
    ax.set_xticks([])      # remove x-axis ticks
    ax.set_yticks([])      # remove y-axis ticks
    ax.set_xticklabels([]) # remove x-axis tick labels
    ax.set_yticklabels([]) # remove y-axis tick labels
    ax.set_xlim([0, .8])
    ax.set_ylim([1, 10.5])

    plt.suptitle(f"Tests for Generalization\nAgent {agent_num}, data epoch {data_epochs}, classifier_pca_epochs {classifier_pca_epochs}", fontsize=16)
    os.makedirs(f"thesis_pics/composition/{args.arg_name}/{agent_num}/pca_{classifier_pca_epochs}", exist_ok = True)
    plt.savefig(f"thesis_pics/composition/{args.arg_name}/{agent_num}/pca_{classifier_pca_epochs}/data_{data_epochs}.png", bbox_inches="tight")
    plt.close()

                    
                    
def plot_dimension_reduction(plot_dict):
    args = plot_dict["args"]
    for agent_num, values_for_composition in enumerate(plot_dict["composition_data"]):

        if(values_for_composition != {}):
            print("\nAGENT NUM", agent_num)
            classifier_pca_dict = {}
            data_dict = {}
            
            try:
                os.makedirs(f"thesis_pics/composition/{args.arg_name}/{agent_num}/loss_values", exist_ok=True)
            except Exception as e:
                print(f"Directory creation failed: {e}")     
        
            for epochs, comp_dict in values_for_composition.items():
                print("EPOCHS", epochs)
                classifier_pca_dict[epochs] = {}

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

                components = ["labels", "hq"]
                results = {}
                for key in components:
                    results[key] = process_component(key)                    
                
                labels = results["labels"]
                                                    
                try:
                    os.makedirs(f"thesis_pics/composition/{args.arg_name}/{agent_num}", exist_ok=True)
                except Exception as e:
                    print(f"Directory creation failed: {e}")     
                    
                data = results["hq"] 
                data_dict[epochs] = (data, labels)

                classifier_pca_dict = train_to_classify_2d(data, labels, epochs, classifier_pca_dict, agent_num, args)
                    
            for data_epochs in values_for_composition.keys():
                for classifier_epochs in values_for_composition.keys():
                    try:
                        os.makedirs(f"communication/saved_deigo/thesis_pics/composition/{args.arg_name}/{agent_num}/pca_{classifier_epochs}", exist_ok=True)
                    except Exception as e:
                        print(f"Directory creation failed: {e}")     
                    
                    print(f"DATA: {data_epochs}. Classifier: {classifier_epochs}")
                    plot_by_epoch_vs_epoch(data_dict, data_epochs, classifier_pca_dict, classifier_epochs, agent_num, args)
                        
                        
                            
                    


def plots(plot_dicts):
    for i, plot_dict in enumerate(plot_dicts):
        args = plot_dict["args"]
        try: os.mkdir(f"saved_deigo/thesis_pics/composition/{args.arg_name}")
        except: pass
        print(f"\nStarting {args.arg_name}.")        
        plot_dimension_reduction(plot_dict)
        print(f"Finished {args.arg_name}.")
        print(f"Duration: {duration()}")
        
    print(f"\tFinished composition.")



if(os.getcwd().split("/")[-1] != "communication"): 
    os.chdir("communication")
os.chdir(f"saved_{args.comp}")
os.makedirs("thesis_pics/composition", exist_ok=True)
    
plot_dicts, min_max_dict, complete_order = load_dicts(args)
plots(plot_dicts)
print(f"\nDuration: {duration()}. Done!")
# %%