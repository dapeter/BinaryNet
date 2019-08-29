import sys
import os
import time
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import pickle
import gzip

import binary_net

from collections import OrderedDict

from mnist_reader import MnistReader

import csv
import argparse
import datetime


def run(binary=False, noise=None, nalpha=0, result_path=None):
    # BN parameters
    batch_size = 128  # default: 100
    print("batch_size = " + str(batch_size))

    # alpha is the exponential moving average factor
    alpha = .1  # default: .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4  # default: 1e-4
    print("epsilon = " + str(epsilon))

    # MLP parameters
    num_units = 300  # default: 4096
    print("num_units = " + str(num_units))
    n_hidden_layers = 2  # default: 3
    print("n_hidden_layers = " + str(n_hidden_layers))

    # Training parameters
    num_epochs = 500  # default: 1000
    print("num_epochs = " + str(num_epochs))

    # Dropout parameters
    dropout_in = .2  # default: .2
    print("dropout_in = " + str(dropout_in))
    dropout_hidden = .5  # default: .5
    print("dropout_hidden = " + str(dropout_hidden))

    # BinaryOut
    if binary:
        activation = binary_net.binary_tanh_unit
        print("activation = binary_net.binary_tanh_unit")
    else:
        activation = lasagne.nonlinearities.tanh
        print("activation = lasagne.nonlinearities.tanh")

    # BinaryConnect
    print("binary = " + str(binary))
    stochastic = False  # default: False
    print("stochastic = " + str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.  # default: 1.
    print("H = " + str(H))
    # W_LR_scale = 1.
    W_LR_scale = "Glorot"  # default: "Glorot"
    print("W_LR_scale = " + str(W_LR_scale))

    # Decaying LR
    LR_start = 0.01  # default: .003
    print("LR_start = " + str(LR_start))
    LR_fin = 0.00001  # default: 0.0000003
    print("LR_fin = " + str(LR_fin))
    LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
    print("LR_decay = " + str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    save_path = None  # default: "mnist_parameters.npz"
    print("save_path = " + str(save_path))

    # Load the dataset (https://github.com/mnielsen/neural-networks-and-deep-learning)
    print('Loading MNIST dataset...')
    mnist = MnistReader("./data/mnist.pkl.gz")

    shuffle_parts = 1  # default: 1
    print("shuffle_parts = " + str(shuffle_parts))

    print("noise = " + str(noise))
    print("nalpha = " + str(nalpha))

    train_set_size = 10000  # default: 50000
    train_X, train_y = mnist.get_train_data(n_samples=train_set_size, noise=noise, alpha=nalpha)
    validation_X, validation_y = mnist.get_validation_data()
    test_X, test_y = mnist.get_test_data()
    print("train_set_size = "+str(train_y.shape[0]))
    print("validation_set_size = "+str(validation_y.shape[0]))
    print("test_set_size = "+str(test_y.shape[0]))

    # Log output
    with open(result_path + "params.txt", "a+") as l:
        print("batch_size = " + str(batch_size), file=l)
        print("alpha = " + str(alpha), file=l)
        print("epsilon = " + str(epsilon), file=l)
        print("num_units = " + str(num_units), file=l)
        print("n_hidden_layers = " + str(n_hidden_layers), file=l)
        print("num_epochs = " + str(num_epochs), file=l)
        print("dropout_in = " + str(dropout_in), file=l)
        print("dropout_hidden = " + str(dropout_hidden), file=l)
        if binary:
            print("activation = binary_net.binary_tanh_unit", file=l)
        else:
            print("activation = lasagne.nonlinearities.tanh", file=l)
        print("binary = " + str(binary), file=l)
        print("stochastic = " + str(stochastic), file=l)
        print("H = " + str(H), file=l)
        print("W_LR_scale = " + str(W_LR_scale), file=l)
        print("LR_start = " + str(LR_start), file=l)
        print("LR_fin = " + str(LR_fin), file=l)
        print("LR_decay = " + str(LR_decay), file=l)
        print("save_path = " + str(save_path), file=l)
        print("shuffle_parts = " + str(shuffle_parts), file=l)
        print("noise = " + str(noise), file=l)
        print("nalpha = " + str(nalpha), file=l)
        print("train_set_size = "+str(train_y.shape[0]), file=l)
        print("validation_set_size = "+str(validation_y.shape[0]), file=l)
        print("test_set_size = "+str(test_y.shape[0]), file=l)

    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    train_X = 2 * train_X.reshape(-1, 1, 28, 28) - 1.
    validation_X = 2 * validation_X.reshape(-1, 1, 28, 28) - 1.
    test_X = 2 * test_X.reshape(-1, 1, 28, 28) - 1.

    # flatten targets
    train_y = np.hstack(train_y)
    validation_y = np.hstack(validation_y)
    test_y = np.hstack(test_y)

    # Onehot the targets
    train_y = np.float32(np.eye(10)[train_y])
    validation_y = np.float32(np.eye(10)[validation_y])
    test_y = np.float32(np.eye(10)[test_y])

    # for hinge loss
    train_y = 2 * train_y - 1.
    validation_y = 2 * validation_y - 1.
    test_y = 2 * test_y - 1.

    print('Building the MLP...')

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
        shape=(None, 1, 28, 28),
        input_var=input)

    mlp = lasagne.layers.DropoutLayer(
        mlp,
        p=dropout_in)

    for k in range(n_hidden_layers):
        mlp = binary_net.DenseLayer(
            mlp,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=num_units)

        mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon,
            alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
            mlp,
            nonlinearity=activation)

        mlp = lasagne.layers.DropoutLayer(
            mlp,
            p=dropout_hidden)

    mlp = binary_net.DenseLayer(
        mlp,
        binary=binary,
        stochastic=stochastic,
        H=H,
        W_LR_scale=W_LR_scale,
        nonlinearity=lasagne.nonlinearities.identity,
        num_units=10)

    mlp = lasagne.layers.BatchNormLayer(
        mlp,
        epsilon=epsilon,
        alpha=alpha)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))

    if binary:

        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_net.compute_grads(loss, mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates, mlp)

        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates.update(lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR))

    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')

    binary_net.train(
        train_fn, val_fn,
        mlp,
        batch_size,
        LR_start, LR_decay,
        num_epochs,
        train_X, train_y,
        validation_X, validation_y,
        test_X, test_y,
        save_path,
        shuffle_parts,
        result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.set_defaults(binary=False)
    parser.add_argument("-a", type=int, required=True)

    args = parser.parse_args()

    result_path = "./results/" + str(datetime.datetime.now().isoformat()) + "/"
    os.mkdir(result_path)

    run(binary=args.binary, noise="u", nalpha=args.a, result_path=result_path)

    print("\nDone")
