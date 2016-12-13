import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


C = scio.loadmat('confusion.mat')
C = C['C']
C = C[0, 0::2]

digits = np.array([0, 1, 2, 3]) + 0.50
labels = ['1', '4', '5','7']

plt.figure()

plt.subplot(221)
plt.pcolormesh(C[0])
plt.colorbar()
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('No Binarize | No Crop', y=1.08)

plt.subplot(222)
plt.pcolormesh(C[1])
plt.colorbar()
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('No Binarize | Crop', y=1.08)

plt.subplot(223)
plt.pcolormesh(C[2])
plt.colorbar()
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Binarize | No Crop', y=1.08)

plt.subplot(224)
plt.pcolormesh(C[3])
plt.colorbar()
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Binarize | Crop', y=1.08)

plt.show()