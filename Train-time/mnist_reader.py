from os import path
import gzip
import struct
import pickle
import numpy as np

class MnistReader(object):
    def __init__(self, data_path):
        if not path.isfile(data_path):
            raise FileNotFoundError

        mnist_file = gzip.open(data_path, "rb")
        training_data, validation_data, test_data = pickle.load(mnist_file, encoding="latin1")
        mnist_file.close()

        self._training_data = training_data
        self._validation_data = validation_data
        self._test_data = test_data

    def get_train_data(self, n_samples=None, noise=None, alpha=0, delta=0):
        if not n_samples:
            n_samples = self._training_data[0].shape[0]
        else:
            n_samples = min(n_samples, self._training_data[0].shape[0])

        if noise is None:
            return self._training_data[0][0:n_samples], self._training_data[1][0:n_samples]
        elif noise == 'u':
            clean_X = self._training_data[0][0:n_samples]
            clean_y = self._training_data[1][0:n_samples]

            rand_idx = np.random.randint(n_samples, high=self._training_data[0].shape[0], size=alpha*n_samples)
            noisy_X = self._training_data[0][rand_idx]
            noisy_y = np.random.randint(0, high=10, size=alpha*n_samples)

            train_X = np.concatenate((clean_X, noisy_X))
            train_y = np.concatenate((clean_y, noisy_y))
            #print(train_X.shape)
            #print(train_y.shape)

            return train_X, train_y

        return None

    def get_validation_data(self, n_samples=None):
        if not n_samples:
            n_samples = self._validation_data[0].shape[0]
        else:
            n_samples = min(n_samples, self._validation_data[0].shape[0])

        return self._validation_data[0][0:n_samples], self._validation_data[1][0:n_samples]

    def get_test_data(self, n_samples=None):
        if not n_samples:
            n_samples = self._test_data[0].shape[0]
        else:
            n_samples = min(n_samples, self._test_data[0].shape[0])

        return self._test_data[0][0:n_samples], self._test_data[1][0:n_samples]
