#%%
import pyvista as pv
import numpy as np
import panel as pn
from matplotlib.colors import to_rgb

pn.extension('vtk')

# Simulated data
N = 500
this_3d = np.random.randn(N, 3)
tasks = np.random.randint(0, 6, N)
task_labels = ['WATCH', 'BE NEAR', 'TOUCH THE TOP', 'PUSH FORWARD', 'PUSH LEFT', 'PUSH RIGHT']
task_colors = {
    'WATCH': '#FF0000',
    'BE NEAR': '#00FF00',
    'TOUCH THE TOP': '#0000FF',
    'PUSH FORWARD': '#00FFFF',
    'PUSH LEFT': '#FF00FF',
    'PUSH RIGHT': '#FFFF00'
}

# Create PyVista point cloud
points = pv.PolyData(this_3d)

# Set default scalar color mapping (task labels)
task_names = np.array([task_labels[t] for t in tasks])
points['task'] = task_names

def get_color_array(selected_labels):
    colors = []
    for label in task_names:
        if label in selected_labels:
            color = np.array(to_rgb(task_colors[label])) * 255
        else:
            color = np.array([180, 180, 180])  # Gray for hidden
        colors.append(color)
    return np.array(colors)

# Panel widgets
label_selector = pn.widgets.RadioButtonGroup(
    name='View by', options=['Task'], value='Task'
)

checkbox = pn.widgets.CheckBoxGroup(
    name='Show Tasks', value=task_labels, options=task_labels
)

# Initial rendering
plotter = pv.Plotter()
actor = plotter.add_points(
    this_3d,
    scalars=get_color_array(checkbox.value),
    rgb=True,
    point_size=8,
    render_points_as_spheres=True
)

vtk_pane = pn.pane.VTK(plotter.ren_win, sizing_mode='stretch_width', height=600)

# Update function
def update_view(event=None):
    actor.GetProperty().SetColor(1, 1, 1)
    new_colors = get_color_array(checkbox.value)
    actor.GetMapper().GetInput().GetPointData().SetScalars(pv.pyvista_ndarray(new_colors))
    vtk_pane.param.trigger('object')

checkbox.param.watch(update_view, 'value')

layout = pn.Row(
    pn.Column(label_selector, checkbox),
    vtk_pane
)

layout.servable()
