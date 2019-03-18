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

# http://deeplearning.net/software/theano/tutorial/using_gpu.html
def testTheano():
    from theano import function, config, shared, tensor
    import numpy
    import time

    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], tensor.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, tensor.Elemwise) and
                ('Gpu' not in type(x.op).__name__)
                for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')


def run(binary=False, noise=None, nalpha=0):
    # BN parameters
    batch_size = 128
    print("batch_size = " + str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = " + str(alpha))
    epsilon = 1e-4
    print("epsilon = " + str(epsilon))

    # MLP parameters
    num_units = 2048
    print("num_units = " + str(num_units))
    n_hidden_layers = 1
    print("n_hidden_layers = " + str(n_hidden_layers))

    # Training parameters
    num_epochs = 200
    print("num_epochs = " + str(num_epochs))

    # Dropout parameters
    dropout_in = .2  # 0. means no dropout
    print("dropout_in = " + str(dropout_in))
    dropout_hidden = .5
    print("dropout_hidden = " + str(dropout_hidden))

    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")

    # BinaryConnect
    # binary = True
    print("binary = " + str(binary))
    stochastic = False
    print("stochastic = " + str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = " + str(H))
    # W_LR_scale = 1.
    W_LR_scale = "Glorot"  # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = " + str(W_LR_scale))

    # Decaying LR
    LR_start = .01
    print("LR_start = " + str(LR_start))
    LR_fin = .0001
    print("LR_fin = " + str(LR_fin))
    LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
    print("LR_decay = " + str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    save_path = "mnist_parameters.npz"
    print("save_path = " + str(save_path))

    shuffle_parts = 1
    print("shuffle_parts = " + str(shuffle_parts))

    # Load the dataset (props to https://github.com/mnielsen/neural-networks-and-deep-learning)
    print('Loading MNIST dataset...')
    mnist = MnistReader("./data/mnist.pkl.gz")

    train_X, train_y = mnist.get_train_data(n_samples=5000, noise=noise, alpha=nalpha)
    validation_X, validation_y = mnist.get_validation_data()
    test_X, test_y = mnist.get_test_data()

    # # Just for testing
    # train_samples = {i: None for i in range(10)}
    # train_labels = {i: None for i in range(10)}
    # print(train_X.shape)
    # print(train_y.shape)
    # print(real_train_y.shape)
    #
    # for i in range(10):
    #     train_i = np.where(real_train_y == i)[0]
    #     train_samples[i] = train_X[train_i, :]
    #     train_labels[i] = train_y[train_i]
    #
    #     plt.hist(train_labels[i])
    #     plt.show()
    # exit(0)

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

    train_losses, val_losses, val_errors, test_err = binary_net.train(
        train_fn, val_fn,
        mlp,
        batch_size,
        LR_start, LR_decay,
        num_epochs,
        train_X, train_y,
        validation_X, validation_y,
        test_X, test_y,
        save_path,
        shuffle_parts)

    # Init csv file writer
    csvfile = open('./results/comparison.csv', 'a')
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow([binary, noise, nalpha, 0, test_err])

    test_errors = np.zeros(train_losses.shape)
    test_errors[0] = test_err
    data = np.column_stack((train_losses, val_losses, val_errors, test_errors))
    header = "Train Loss, Validation Loss, Validation Error, Test Error"
    np.savetxt('./results/bin_{}_noise_{}_nalpha_{}.dat'.format(binary, noise, nalpha), data, header=header)


if __name__ == "__main__":
    #for nalpha in range(10, 51, 10):
    #    for binary in [True, False]:
    run(binary=False, noise=None, nalpha=0)
