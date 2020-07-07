"""
A multi-layer Perceptron implementation in Python three.
Copyright (C) 2020  Tashfeen, Ahmad

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np


class Network:

    def __init__(self, X, y, structure, epochs=20, bt_size=32, eta=0.3):
        # labels to one-hot array
        self.X, self.y = X, np.eye(len(set(y)))[y.reshape(-1)]
        self.structure = structure
        self.epochs, self.bt_size = epochs, bt_size
        self.eta = 0.3
        self.L = len(structure)
        self.Wb = self.random_weights_biases()
        self.W, self.b = self.Wb

    def random_weights_biases(self, sigma=1, mu=0):
        W = np.empty(self.L - 1, dtype=object)
        b = np.empty(self.L - 1, dtype=object)
        for i in range(self.L - 1):
            c, r = self.structure[i], self.structure[i + 1]
            W[i] = sigma * np.random.randn(r, c) + mu
            b[i] = sigma * np.random.randn(r) + mu

        Wb = np.empty(2, dtype=object)
        Wb[0], Wb[1] = W, b  # all weights and biases
        return Wb

    def batches(self):
        shuffle_ind = np.arange(len(self.X))
        np.random.shuffle(shuffle_ind)
        shuffle_X, shuffle_y = self.X[shuffle_ind], self.y[shuffle_ind]
        i, num_batches = 0, int(len(shuffle_X) / self.bt_size)
        for i in range(num_batches - 1):
            l, u = i * self.bt_size, (i + 1) * self.bt_size
            mini_batch_X = shuffle_X[l:u]
            mini_batch_y = shuffle_y[l:u]
            yield zip(mini_batch_X, mini_batch_y)
        mini_batch_X = shuffle_X[(i + 1) * self.bt_size:]
        mini_batch_y = shuffle_y[(i + 1) * self.bt_size:]
        yield zip(mini_batch_X, mini_batch_y)

    def tango(self, print_steps=False):  # train
        for epoch in range(self.epochs):
            error = self.SGD(print_steps)
            print('* Epoch: {:>4}, Error: {:>3.5}'.format(epoch, error))

    def SGD(self, print_steps):  # Stochastic Gradient Descent
        step, step_error_sum = 0, 0
        for mini_batch in self.batches():
            gradient, step_error = self.average_gradient(mini_batch)
            step_error_sum += step_error
            self.Wb = self.Wb - (self.eta * gradient)
            self.W, self.b = self.Wb
            if print_steps:
                to_print = 'SGD step: {:>7}, Error: {:>3.5}'
                print(to_print.format(step, step_error))
            step += 1
        return step_error_sum / step

    def average_gradient(self, mini_batch):
        g_sum, error_sum = self.backpropagation(*next(mini_batch))
        for x, y in mini_batch:
            batch_gradient, error = self.backpropagation(x, y)
            error_sum += error
            g_sum += batch_gradient
        return g_sum / self.bt_size, error_sum / self.bt_size

    def backpropagation(self, x, y):
        outputs, activations = self.forward_pass(x)
        gradient = self.backward_pass(outputs, activations, y)
        return gradient, self.error(y, activations[-1])

    def forward_pass(self, example, keep_track=True):
        input_layer = example.flatten()
        # if we only want the output of the network
        if keep_track is False:
            for W, b in zip(self.W, self.b):
                input_layer = self.sigmoid(np.dot(W, input_layer) + b)
            return input_layer
        outputs = np.empty(shape=self.L - 1, dtype=np.object)  # z^(l)
        activations = np.empty(shape=self.L, dtype=np.object)  # a^(l)
        activations[0] = input_layer
        for W, b, l in zip(self.W, self.b, range(self.L - 1)):
            outputs[l] = np.dot(W, input_layer) + b
            activations[l + 1] = self.sigmoid(outputs[l])
            input_layer = activations[l + 1]
        return outputs, activations

    def backward_pass(self, outputs, activations, y):
        gradient_W = np.empty(shape=self.L - 1, dtype=np.object)
        gradient_b = np.empty(shape=self.L - 1, dtype=np.object)
        z, a = outputs[-1], activations[-1]  # z^L, a^L
        delta = -2 * (y - a) * self.sigmoid(z, derivative=True)  # delta^L eq 2
        delta = delta.reshape((1, len(delta)))
        for l in range(self.L - 1, 0, -1):
            a_prev = activations[l - 1]
            a_prev = a_prev.reshape((len(a_prev), 1)).T
            pC_w = np.dot(delta.T, a_prev)  # eq 1 or 4
            pC_b = delta.flatten()  # eq 3 or 6
            gradient_W[l - 1], gradient_b[l - 1] = pC_w, pC_b
            if l == 1:
                break
            z, a = outputs[l - 2], activations[l - 1]
            delta = np.dot(delta, self.W[l - 1]) * self.sigmoid(z, derivative=True)  # eq 5
        gradient = np.empty(shape=2, dtype=np.object)
        gradient[0], gradient[1] = gradient_W, gradient_b
        return gradient

    def error(self, y, a):
        return np.sum(np.square(y - a))

    def sigmoid(self, x, derivative=False):
        s = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
        return s(x) * (1 - s(x)) if derivative else s(x)

    def predict(self, input_layer):
        output_layer = self.forward_pass(input_layer, False)
        return output_layer.argmax()

    def save_weights_biases(self, path='./weights_biases.npy'):
        return np.save(path, self.Wb)

    def load_weights_biases(self, path='./weights_biases.npy'):
        self.Wb = np.load(path, allow_pickle=True)
        self.W, self.b = self.Wb
        return True

    def __repr__(self):
        ret = ''
        for l, W, b in zip(self.structure, self.W, self.b):
            ret += '({}: W{} + b{})\n'.format(l, W.shape, b.shape)
        return ret

    def __str__(self):
        return self.__repr__()
