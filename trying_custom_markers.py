#%%

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import numpy as np

# Load your custom marker image
custom_marker = mpimg.imread('pybullet_data/shapes/cone.png')  # Must be black on transparent background

# Example data
xs = np.random.rand(10)
ys = np.random.rand(10)

fig, ax = plt.subplots()

for (x, y) in zip(xs, ys):
    imagebox = OffsetImage(custom_marker, zoom=1)  # Adjust zoom as needed
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.show()