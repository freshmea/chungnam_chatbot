# 1 import library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

X_train = np.load("pytorch/data/data.npy")
print(X_train.shape, X_train.dtype, X_train)

# 2 gmm generate
gmm = GaussianMixture(n_components=4)
gmm.fit(X_train)

print(gmm.means_, "\n", gmm.covariances_)
print(len(np.linspace(-1, 6, 100)))
X, Y = np.meshgrid(np.linspace(-1, 6, 100), np.linspace(-1, 6, 100))
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(100, 100)

plt.contour(X, Y, Z, levels=np.logspace(0, 2, 30))
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.show()
