from network import Network
import matplotlib.pylab as plt
import numpy as np

N, n = 700, 10
X = np.array([list(map(int, '{0:010b}'.format(e))) for e in range(2**n)])
y = np.array([int(e % 2 == 0) for e in range(2**n)])
X, y, X_test, y_test = X[:N], y[:N], X[N:], y[N:]

np.random.seed(0)
net = Network(X, y, structure=[10, 3, 2], epochs=20, bt_size=8, eta=0.3)

# train
net.tango()

# test
predictions = np.array([net.predict(x.flatten()) for x in X_test])
acc = np.sum(predictions == y_test) / len(y_test)
print('Network Accuracy: {}'.format(acc))

# plot
fig, (ax1, ax2) = plt.subplots(2)
ax1.set_ylabel('$W_{3,10} \in \mathcal{W}$')
ax1.set_xticks(range(10))
ax1.set_yticks(range(3))
ax1.imshow(net.Wb[0][0], cmap='bwr')

ax2.set_ylabel('$W_{2,3} \in \mathcal{W}$')
ax2.set_xticks(range(3))
ax2.set_yticks(range(2))
im = ax2.imshow(net.Wb[0][1], cmap='bwr')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.suptitle('Plotted weight matrices', fontsize=16)

plt.show()
