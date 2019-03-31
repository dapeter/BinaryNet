
import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import gzip

import binary_net

from cifar_reader import CifarReader

from collections import OrderedDict

import matplotlib.pyplot as plt

import csv


def run(binary=False, noise=None, nalpha=0, ndelta=0):
    # BN parameters
    batch_size = 128
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # BinaryOut
    if binary:
        activation = binary_net.binary_tanh_unit
        print("activation = binary_net.binary_tanh_unit")
    else:
        activation = lasagne.nonlinearities.rectify
        print("activation = lasagne.nonlinearities.rectify")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
    # BinaryConnect    
    # binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Training parameters
    num_epochs = 75
    print("num_epochs = "+str(num_epochs))

    # Decaying LR 
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.001  # 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    train_set_size = 10000
    print("train_set_size = "+str(train_set_size))
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))

    print("noise = " + str(noise))
    print("nalpha = " + str(nalpha))
    print("ndelta = " + str(ndelta))

    print('Loading CIFAR-10 dataset...')
    cifar = CifarReader("./data/cifar-10-batches-py/")

    train_X, train_y = cifar.get_train_data(n_samples=train_set_size, noise=noise, alpha=nalpha, delta=ndelta)
    valid_X, valid_y = cifar.get_validation_data()
    test_X, test_y = cifar.get_test_data()

    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    train_X = np.reshape(np.subtract(np.multiply(2./255.,train_X),1.),(-1,3,32,32))
    valid_X = np.reshape(np.subtract(np.multiply(2./255.,valid_X),1.),(-1,3,32,32))
    test_X = np.reshape(np.subtract(np.multiply(2./255.,test_X),1.),(-1,3,32,32))

    # flatten targets
    train_y = np.hstack(train_y)
    valid_y = np.hstack(valid_y)
    test_y = np.hstack(test_y)

    # Onehot the targets
    train_y = np.float32(np.eye(10)[train_y])
    valid_y = np.float32(np.eye(10)[valid_y])
    test_y = np.float32(np.eye(10)[test_y])

    # for hinge loss
    train_y = 2* train_y - 1.
    valid_y = 2* valid_y - 1.
    test_y = 2* test_y - 1.

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = lasagne.layers.InputLayer(
            shape=(None, 3, 32, 32),
            input_var=input)
    
    # 128C3-128C3-P2             
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 

    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 

    # 256C3-256C3-P2
    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # print(cnn.output_shape)
    
    # 512FP-512FP-10FP
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=512)
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 

    cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=512)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)      
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(cnn, binary=True)
        W_grads = binary_net.compute_grads(loss,cnn)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,cnn)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
        updates.update(lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR))

    else:
        params = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    train_losses, val_losses, val_errors, test_err = binary_net.train(
            train_fn,val_fn,
            cnn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_X,train_y,
            valid_X,valid_y,
            test_X,test_y,
            shuffle_parts=shuffle_parts)

    # Init csv file writer
    csvfile = open('./results/cnn_comparison.csv', 'a')
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow([binary, noise, nalpha, ndelta, test_err])

    test_errors = np.zeros(train_losses.shape)
    test_errors[0] = test_err
    data = np.column_stack((train_losses, val_losses, val_errors, test_errors))
    header = "Train Loss, Validation Loss, Validation Error, Test Error"
    np.savetxt('./results/bin_{}_noise_{}_nalpha_{}_ndelta_{}.dat'.format(binary, noise, nalpha, ndelta), data, header=header)


if __name__ == "__main__":
    for alpha in range(4, 6):
        for binary in [False, True]:
            run(binary=binary, noise='u', nalpha=alpha, ndelta=0)
