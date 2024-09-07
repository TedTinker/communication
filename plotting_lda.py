#%%
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
import plotly.graph_objects as go

from utils import args, duration, load_dicts, print

print("name:\n{}\n".format(args.arg_name),)

dpi = 50



def plots(plot_dicts):
  
    for i, plot_dict in enumerate(plot_dicts):
        print(f"\nStarting {plot_dict['arg_name']}.")
        
        # These don't look good!
        # LDA 
        try: os.mkdir("thesis_pics/lda")
        except: pass
        
        def plot_interactive_3d(plot_dict, file_name="lda_plot.html"):
            """
            Creates an interactive 3D plot and saves it as an HTML file for click-and-drag navigation.

            Parameters:
            plot_dict (dict): Dictionary containing the data for plotting.
            file_name (str): Name of the output HTML file.
            """
            for agent_num, agent in enumerate(plot_dict["lda_transformations"]):
                for epochs, (actions, colors, shapes) in agent.items():
                    # Flatten the data if it's in nested arrays
                    actions = np.array(actions).flatten()
                    colors = np.array(colors).flatten()
                    shapes = np.array(shapes).flatten()

                    # Debug: Print the ranges
                    print(f"Actions Range: {actions.min()} to {actions.max()}")
                    print(f"Colors Range: {colors.min()} to {colors.max()}")
                    print(f"Shapes Range: {shapes.min()} to {shapes.max()}")

                    fig = go.Figure()

                    # Create the 3D scatter plot
                    fig.add_trace(go.Scatter3d(
                        x=actions, 
                        y=colors, 
                        z=shapes, 
                        mode='markers',
                        marker=dict(
                            size=8,  # Marker size
                            color=colors,  # Colors can be any valid sequence
                            opacity=0.8,
                            colorscale='Viridis'  # Optional colorscale
                        )
                    ))

                    # Set labels and title
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(title='Actions', range=[actions.min(), actions.max()]),
                            yaxis=dict(title='Colors', range=[colors.min(), colors.max()]),
                            zaxis=dict(title='Shapes', range=[shapes.min(), shapes.max()])
                        ),
                        title=f"LDA after {epochs} epochs",
                        autosize=True,
                        height=800,
                        width=1000,
                    )

                    # Save the plot as an interactive HTML file
                    output_file = f"thesis_pics/lda/{file_name}_epoch_{epochs}_agent_num_{agent_num}.html"
                    fig.write_html(output_file)

                    print(f"Interactive plot saved as {output_file}")



        plot_interactive_3d(plot_dict)

            
        print(f"\tFinished LDA.")
        print(f"Finished {plot_dict['arg_name']}.")
        print(f"Duration: {duration()}")
    
    

plot_dicts, min_max_dict, complete_order = load_dicts(args)
plots(plot_dicts)
print(f"\nDuration: {duration()}. Done!")
# %%
