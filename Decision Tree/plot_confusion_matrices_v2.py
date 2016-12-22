from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import pickle

'''
r|b|c
0 0 0
0 0 1
0 1 0
0 1 1
1 0 0
1 0 1
1 1 0
1 1 1
'''

import pickle

R1 = pickle.load(open('entropy_Trial_1.p', 'rb'))
R1 = R1['confusion_matrix']
R2 = pickle.load(open('entropy_cropped_Trial_1.p', 'rb'))
R2 = R2['confusion_matrix']
R3 = pickle.load(open('entropy_binarized_Trial_1.p', 'rb'))
R3 = R3['confusion_matrix']
R4 = pickle.load(open('entropy_cropped_binarized_Trial_1.p', 'rb'))
R4 = R4['confusion_matrix']

R5 = pickle.load(open('gini_Trial_1.p', 'rb'))
R5 = R5['confusion_matrix']
R6 = pickle.load(open('gini_cropped_Trial_1.p', 'rb'))
R6 = R6['confusion_matrix']
R7 = pickle.load(open('gini_binarized_Trial_1.p', 'rb'))
R7 = R7['confusion_matrix']
R8 = pickle.load(open('gini_cropped_binarized_Trial_1.p', 'rb'))
R8 = R8['confusion_matrix']


digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 0.50
labels = ['1','2','3', '4','5', '6', '7', '8', '9', '0']

K = 500
N = 100
T0 = 10
Td = 10

cmap = 'YlOrRd'

plt.figure()

plt.subplot(221)
plt.pcolormesh(R1)
plt.colorbar()
plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Entropy | No Binarize | No Crop')

plt.subplot(222)
plt.pcolormesh(R2)
plt.colorbar()
plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Entropy | No Binarize | Crop')

plt.subplot(223)
plt.pcolormesh(R3)
plt.colorbar()
plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Entropy | Binarize | No Crop')

plt.subplot(224)
plt.pcolormesh(R4)
plt.colorbar()
plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Entropy | Binarize | Crop')

plt.tight_layout()
plt.show()


plt.figure()

plt.subplot(221)
plt.pcolormesh(R5)
plt.colorbar()
plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Gini | No Binarize | No Crop')

plt.subplot(222)
plt.pcolormesh(R6)
plt.colorbar()
plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Gini | No Binarize | Crop')

plt.subplot(223)
plt.pcolormesh(R7)
plt.colorbar()
plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Gini | Binarize | No Crop')

plt.subplot(224)
plt.pcolormesh(R8)
plt.colorbar()
plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()
plt.xticks(digits, labels)
plt.yticks(digits, labels)
plt.title('Gini | Binarize | Crop')

plt.tight_layout()
plt.show()