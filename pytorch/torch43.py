# 1 import libraries
import numpy as np
from sklearn.datasets import load_digits
from minisom import MiniSom
from pylab import plot, axis, show, pcolor, colorbar, bone

digits = load_digits()
data = digits.data
labels = digits.target

# 2 MiniSom algorithm
som = MiniSom(16, 16, 64, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 10000)

bone()
pcolor(som.distance_map().T)
colorbar()
show()

# 3 labels setting, color setting, plot
for i in range(10):
    labels[labels == str(i)] = i

makers = list("ov138spxD*")
color = list("rgbymck")
color.append((0, 0.1, 0.8))
color.append((1, 0.5, 0))
color.append((1, 1, 0.3))

for cnt, xx in enumerate(data):
    w = som.winner(xx)
    plot(
        w[0] + 0.5,
        w[1] + 0.5,
        makers[labels[cnt]],
        markerfacecolor="None",
        markeredgecolor=color[labels[cnt]],
        markersize=12,
        markeredgewidth=2,
    )
show()
