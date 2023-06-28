import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


x_labels = ['Cylinder', 'Pole', 'Flat\nBottom', 'Flat\nTop', 'Cone', 'Inverse\nCone']
y_labels = ['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow']

data = np.array(
    [
     [1,1,1,1,0,0],
     [0,1,1,1,1,0],
     [0,0,1,1,1,1],
     [1,0,0,1,1,1],
     [1,1,0,0,1,1],
     [1,1,1,0,0,1]])

colors = ['red', 'green']
cmap = plt.cm.colors.ListedColormap(colors)

fig, ax = plt.subplots(figsize=(5,5))
cax = ax.matshow(data, cmap=cmap)

ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

legend_elements = [Patch(facecolor='green', edgecolor='k', label='Training'),
                   Patch(facecolor='red', edgecolor='k', label='Testing')]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))


plt.show()
plt.close()