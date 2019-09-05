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

        train_X, train_y = training_data
        self._train_samples = {i: None for i in range(10)}
        for i in range(10):
            train_i = np.where(train_y == i)[0]
            self._train_samples[i] = train_X[train_i, :]

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

            rand_idx = np.random.randint(self._training_data[0].shape[0], size=alpha*n_samples)
            noisy_X = self._training_data[0][rand_idx]
            noisy_y = np.random.randint(10, size=alpha*n_samples)

            train_X = np.concatenate((clean_X, noisy_X))
            train_y = np.concatenate((clean_y, noisy_y))

            return train_X, train_y
        elif noise == 's':
            confusion_matrix = np.loadtxt("./results/confusion.dat")

            similarities = np.argsort(confusion_matrix, axis=1)
            similarities = np.flip(similarities, axis=1)

            deltas = delta * (-np.arange(0, 10, 1) + 5) / 4
            probabilities = 1 / 10 + deltas / (1 + alpha)
            probabilities[0] = 1 / 10

            clean_X = self._training_data[0][0:n_samples]
            clean_y = self._training_data[1][0:n_samples]

            # For every clean sample with label i, add alpha samples with label i and random label from 0 to 9
            train_X = clean_X
            train_y = clean_y
            # real_train_y = clean_y
            for i in range(10):
                # First, get number of samples with label i
                num_samples = np.count_nonzero(clean_y == i)
                num_noisy = num_samples * alpha

                # Get num_noisy samples
                sample_indices = np.random.randint(0, high=self._train_samples[i].shape[0], size=num_noisy)
                noisy_X = self._train_samples[i][sample_indices, :]

                # Labels are randomly assigned according to probability distribution given by the delta factor
                p = np.zeros(probabilities.shape)
                p[similarities[i]] = probabilities
                noisy_y = np.random.choice(10, size=num_noisy, p=p)

                train_X = np.concatenate((train_X, noisy_X))
                train_y = np.concatenate((train_y, noisy_y))

                # real_train_y = np.concatenate((real_train_y, np.ones(num_noisy) * i))

            return train_X, train_y #, real_train_y

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
