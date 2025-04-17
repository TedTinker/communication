#%%
import matplotlib.pyplot as plt
import numpy as np
%matplotlib qt

plt.figure()
ax = plt.subplot(1, 1, 1)

N = 100
x = np.array([i for i in range(N)])
y = np.random.randn(*x.shape)

line = plt.plot(x, y)[0]
plt.ion()
plt.pause(0.01)

for i in range(200):
	y = np.random.randn(*x.shape)
	line.set_ydata(y)
	ax.set_title(str(i))
	plt.pause(0.01)

plt.ioff()
plt.show()