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

        batch_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        meta_name = "batches.meta"
        test_batch_name = "test_batch"

        for batch_file in batch_names:
            if not path.isfile(data_path + batch_file):
                raise FileNotFoundError

        if not path.isfile(data_path + meta_name) or not path.isfile(data_path + test_batch_name):
            raise FileNotFoundError

        # Load data from batch files
        data_batches = []
        meta = None
        test_batch = None

        for batch_file in batch_names:
            with open(data_path + batch_file, 'rb') as fo:
                data_batches.append(pickle.load(fo, encoding='bytes'))

        with open(data_path + meta_name, 'rb') as fo:
            meta = pickle.load(fo, encoding='bytes')

        with open(data_path + test_batch_name, 'rb') as fo:
            test_batch = pickle.load(fo, encoding='bytes')

        # Merge samples and labels from all data batches
        # Use the first 45.000 samples for training and the rest (5.000 samples) for validation
        train_data = []
        train_labels = []
        for batch in range(len(batch_names)):
            train_data.extend(data_batches[batch][b'data'])
            train_labels.extend(data_batches[batch][b'labels'])

        validation_data = train_data[45000:]
        validation_labels = train_labels[45000:]
        del train_data[45000:]
        del train_labels[45000:]

        # Also load test samples and test labels from the test batch
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
