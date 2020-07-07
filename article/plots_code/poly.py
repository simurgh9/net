import numpy as np
import matplotlib.pyplot as plt


def g(x, drv=False):
    return 2 * x if drv else x**2


x = np.linspace(-10, 10)
y = g(x)
eta = 0.1

plt.xlim(-10, 10), plt.ylim(0, 20)
plt.xlabel('$x$'), plt.ylabel('$g(x)$')
plt.title('Descending per Derivative.')
plt.plot(x, y, 'k', label='Function: $g(x)$.')

t, T = 10, []
while g(t) > 0:
    T.append(t)
    t -= eta * g(t, drv=True)

t, Z = -10, []
while g(t) > 0:
    Z.append(t)
    t -= eta * g(t, drv=True)

plt.plot(T, [g(t) for t in T], 'r ^', label='Negative Nudges to $x$.')
plt.plot(Z, [g(z) for z in Z], 'g ^', label='Positive Nudges to $x$.')

plt.legend()
plt.show()
