#%% 
import numpy as np
import matplotlib.pyplot as plt

# Define the range of x-values
x_values = np.linspace(1, 50000, 50000)

# Define parameters for the sigmoid function
L = 1       # maximum value of the curve
x0 = 25000  # midpoint of the sigmoid
ks = [.0001, 0.0002, .0003, .0005]  # steepness of the curve

# Sigmoid function
for k in ks:
    y_values = 1 - L / (1 + np.exp(-k * (x_values - x0)))

    # Plot the sigmoid curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values)
    plt.title("Sigmoid Curve from x = 1 to 50000, k = " + str(k))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()

# %%
