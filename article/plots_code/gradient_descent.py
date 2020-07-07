import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from pylab import rcParams
# rcParams['figure.figsize'] = 15, 10
# http://mpastell.com/2013/05/02/matplotlib_colormaps/


def gradient_decent(x, f, r=0.07, e=0.01):
    X = np.array([np.linspace(-50, 100), np.linspace(-50, 100)])
    X = np.meshgrid(X[0], X[1])
    z = f(X)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('$a$')
    ax.set_ylabel('$b$')
    ax.set_zlabel(r'$z = g(a, b)$')
    plt.title('Gradient Descent')
    ax.plot_surface(X[0], X[1], z, linewidth=0.1,
                    antialiased=True, alpha=0.7, cmap="CMRmap")

    i, t = 0, x - r * f(x, gradient=True)
    while (np.abs(f(t) - f(x)) >= e):
        ax.scatter(t[0], t[1], f(t) + 5, c='k', marker='.')
        # plt.pause(0.001)
        x = t
        i, t = i + 1, x - r * f(x, gradient=True)
    return x


def f(x, gradient=False):
    return np.array([2 * x[0], 2 * x[1]]) if gradient else x[0]**2 + x[1]**2


gradient_decent(np.array([100, 100]), f)
plt.show()
