from os import path
import gzip
import struct
import pickle
import numpy as np
import theano

class CifarReader(object):
    def __init__(self, data_path):
        # Check if necessary files actually exist
        if not path.isdir(data_path):
            raise FileNotFoundError

        train_batch_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
        validation_batch_name = "data_batch_5"
        test_batch_name = "test_batch"

        # Load data from batch files
        train_batches = []
        validation_batch = None
        test_batch = None

        for batch_file in train_batch_names:
            with open(data_path + batch_file, 'rb') as fo:
                train_batches.append(pickle.load(fo, encoding='bytes'))

        with open(data_path + validation_batch_name, 'rb') as fo:
            validation_batch = pickle.load(fo, encoding='bytes')

        with open(data_path + test_batch_name, 'rb') as fo:
            test_batch = pickle.load(fo, encoding='bytes')

        # Load sample data
        train_data = []
        train_labels = []
        for batch in range(len(train_batch_names)):
            train_data.extend(train_batches[batch][b'data'])
            train_labels.extend(train_batches[batch][b'labels'])

        validation_data = validation_batch[b'data']
        validation_labels = validation_batch[b'labels']

        test_data = test_batch[b'data']
        test_labels = test_batch[b'labels']

        # Convert samples and labels to numpy arrays and store them as member variables
        self._train_X = np.array(train_data, dtype=theano.config.floatX)
        self._train_y = np.array(train_labels)

        self._valid_X = np.array(validation_data, dtype=theano.config.floatX)
        self._valid_y = np.array(validation_labels)

        self._test_X = np.array(test_data, dtype=theano.config.floatX)
        self._test_y = np.array(test_labels)

        self._num_train = self._train_X.shape[0]
        self._num_valid = self._valid_X.shape[0]
        self._num_test = self._test_X.shape[0]

    def get_train_data(self, n_samples=None, noise=None, alpha=0, delta=0):
        if not n_samples:
            n_samples = self._num_train
        else:
            n_samples = min(n_samples, self._num_train)

        if noise is None:
            return self._train_X[0:n_samples], self._train_y[0:n_samples]
        elif noise == 'u':
            clean_X = self._train_X[0:n_samples]
            clean_y = self._train_y[0:n_samples]

            rand_idx = np.random.randint(self._num_train, size=alpha*n_samples)
            noisy_X = self._train_X[rand_idx]
            noisy_y = np.random.randint(10, size=alpha*n_samples)

            train_X = np.concatenate((clean_X, noisy_X))
            train_y = np.concatenate((clean_y, noisy_y))

            return train_X, train_y

        return self._train_X[0:n_samples], self._train_y[0:n_samples]

    def get_validation_data(self, n_samples=None):
        if not n_samples:
            n_samples = self._num_valid
        else:
            n_samples = min(n_samples, self._num_valid)

        return self._valid_X[0:n_samples], self._valid_y[0:n_samples]

    def get_test_data(self, n_samples=None):
        if not n_samples:
            n_samples = self._num_test
        else:
            n_samples = min(n_samples, self._num_test)

        return self._test_X[0:n_samples], self._test_y[0:n_samples]
