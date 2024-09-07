#%% 
import torch

# Example dimensions
batch_size = 100  # Number of sequences
feature_size = 50  # Number of features (similar to the dimension of one-hot vectors or activations)

# Generate a random tensor (values between 0 and 1)
activations = torch.rand(batch_size, feature_size)
print(activations.shape)



from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Option 1: PCA for dimensionality reduction to 3D
pca = PCA(n_components=3)
activations_3d_pca = pca.fit_transform(activations.detach().cpu().numpy())
print(activations_3d_pca.shape)

# Option 2: t-SNE for dimensionality reduction to 3D
tsne = TSNE(n_components=3, random_state=42)
activations_3d_tsne = tsne.fit_transform(activations.detach().cpu().numpy())
print(activations_3d_tsne.shape)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot PCA or t-SNE results (depending on the method you chose)
ax.scatter(activations_3d_pca[:, 0], activations_3d_pca[:, 1], activations_3d_pca[:, 2], c='blue', marker='o')

# Set labels and show plot
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title("PCA 3D Visualization of Layer Activations")
plt.show()



# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot PCA or t-SNE results (depending on the method you chose)
ax.scatter(activations_3d_tsne[:, 0], activations_3d_tsne[:, 1], activations_3d_tsne[:, 2], c='blue', marker='o')

# Set labels and show plot
ax.set_xlabel('TSNE 1')
ax.set_ylabel('TSNE 2')
ax.set_zlabel('TSNE 3')
plt.title("TSNE 3D Visualization of Layer Activations")
plt.show()
# %%

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

batch_size = 128
activations = torch.rand(batch_size, feature_size)

# Example with labeled data (each sentence has an associated action, color, and shape)
# Assuming labels for actions, colors, and shapes are integers or categorical labels
labels = torch.randint(0, 5, (batch_size,))  # Dummy labels (0, 1, 2 for action, color, shape)

# Perform LDA (Supervised dimensionality reduction)
lda = LDA(n_components=3)
lda.fit(activations.detach().cpu().numpy(), labels.detach().cpu().numpy())
activations_3d_lda = lda.transform(activations.detach().cpu().numpy())

# Visualize the LDA results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(activations_3d_lda[:, 0], activations_3d_lda[:, 1], activations_3d_lda[:, 2], c=labels)
ax.set_xlabel('LDA 1')
ax.set_ylabel('LDA 2')
ax.set_zlabel('LDA 3')
plt.title("LDA 3D Visualization of Activations based on Labels")
plt.show()

# %%
d = {}

d[0] = (1, 2, 3)

for epochs, (actions, colors, shapes) in d.items():
    print(epochs, actions, colors, shapes)
# %%
