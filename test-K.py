import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *
import sys

ax = plt.subplot(projection="3d")

X = np.linspace(0, 2 * np.pi, 50)
Y = np.linspace(0, 2 * np.pi, 50)
X, Y = np.meshgrid(X, Y)

# f2 = lambda x, y: u(np.array([x, y]))

Z = np.vectorize(K)(X, Y)

surf = ax.plot_surface(
    X,
    Y,
    Z,
    rstride=1,
    cstride=1,
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False,
    alpha=0.9,
)
plt.colorbar(surf, shrink=0.5, aspect=10)

# if x is not None:
#     ax.scatter(x[0], x[1], u(x), 'r.')

plt.show()
