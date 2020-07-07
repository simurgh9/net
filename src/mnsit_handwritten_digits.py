"""
Downloads, reshapes and returns the MNSIT data as Numpy ndarrays.
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
import os
import gzip
import requests
import numpy as np
import matplotlib.pyplot as plt


class MNSIT:
    DOWNLOAD_ADDRESS = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'  # training set images (9912422 bytes)
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'  # training set labels (28881 bytes)
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'  # test set images (1648877 bytes)
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'  # test set labels (4542 bytes)

    def __init__(self, path=None):
        self.files = [
            self.TRAIN_IMAGES, self.TEST_IMAGES, self.TRAIN_LABELS,
            self.TEST_LABELS
        ]
        self.path = path
        if self.path is None:
            self.path = os.path.join('.', 'mnsit_data', '')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.download()
        self.x, self.tx, self.y, self.ty = self.load()

    def get_data(self):
        return self.x, self.tx, self.y, self.ty

    def plot_image(self, i, source='training'):
        if source == 'training':
            if i > self.x.shape[0]:
                raise IndexError(
                    'Index {} out of bounds in {} examples.'.format(
                        i, self.x.shape[0]))
            image = np.array(self.x[i], dtype='float')
            plt.imshow(image, cmap='gray_r')
            plt.title('Label: {}'.format(self.y[i]))
            plt.show()
            return
        elif source == 'testing':
            if i > self.tx.shape[0]:
                raise IndexError(
                    'Index {} out of bounds in {} examples.'.format(
                        i, self.tx.shape[0]))
            image = np.array(self.tx[i], dtype='float')
            plt.imshow(image, cmap='gray')
            plt.title('Label: {}'.format(self.ty[i]))
            plt.show()
            return
        else:
            raise ValueError('Source must either be "testing" or "training".')

    def download(self, path=None):
        path = self.path if path is None else path
        for fl_name in self.files:
            responce = requests.get(self.DOWNLOAD_ADDRESS + fl_name)
            file = open(path + fl_name, 'wb')
            file.write(responce.content)

    def load(self, path=None):
        path = self.path if path is None else path
        ret = [None] * 4
        for fl_name, i in zip(self.files, range(4)):
            f = gzip.open(path + fl_name, 'rb')
            arr = bytearray(f.read())
            # magic_number = int.from_bytes(arr[:4], byteorder='big')
            num_examples = int.from_bytes(arr[4:8], byteorder='big')
            if i < 2:
                rows = int.from_bytes(arr[8:12], byteorder='big')
                cols = int.from_bytes(arr[12:16], byteorder='big')
                ret[i] = np.array(arr[16:]).reshape(num_examples, rows, cols)
            else:
                ret[i] = np.array(arr[8:])
        return tuple(ret)
