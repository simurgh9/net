from mnsit_handwritten_digits import MNSIT
from network import Network
import numpy as np

# MNSIT Data
mn = MNSIT(path='../mnsit_data/')
# mn.plot_image(999, source='training')
train_X, test_X, train_y, test_y = mn.get_data()

# Network
np.random.seed(0)
net = Network(train_X,
              train_y,
              structure=[784, 32, 10],
              epochs=100,
              bt_size=256)
net.load_weights_biases(path='weights_biases.npy')

# train
# net.tango()
# net.save_weights_biases(path='weights_biases_new.npy')

# test
predictions = np.array([net.predict(x.flatten()) for x in test_X])
acc = np.sum(predictions == test_y) / len(test_y)
print('Network Accuracy: {}'.format(acc))
