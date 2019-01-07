
from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_net

from collections import OrderedDict

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

if __name__ == "__main__":
    #testTheano()

    # BN parameters
    batch_size = 100
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    num_units = 4096
    print("num_units = "+str(num_units))
    n_hidden_layers = 3
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # Training parameters
    num_epochs = 1000
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    dropout_in = .2 # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = .5
    print("dropout_hidden = "+str(dropout_hidden))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
    # BinaryConnect
    binary = True
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
    
    # Decaying LR 
    LR_start = .003
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    save_path = "mnist_parameters.npz"
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    # Load the dataset (props to https://github.com/mnielsen/neural-networks-and-deep-learning)
    print('Loading MNIST dataset...')
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    train_X = [t_d[0] for t_d in training_data]
    train_X = np.stack(train_X, axis=0)
    train_y = [t_d[1] for t_d in training_data]
    train_y = np.stack(train_y, axis=0)

    validation_X = [v_d[0] for v_d in validation_data]
    validation_X = np.stack(validation_X, axis=0)
    validation_y = [v_d[1] for v_d in validation_data]
    validation_y = np.stack(validation_y, axis=0)

    test_X = [t_d[0] for t_d in test_data]
    test_X = np.stack(test_X, axis=0)
    test_y = [t_d[1] for t_d in test_data]
    test_y = np.stack(test_y, axis=0)

    # bc01 format    
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    train_X = 2* train_X.reshape(-1, 1, 28, 28) - 1.
    validation_X = 2* validation_X.reshape(-1, 1, 28, 28) - 1.
    test_X = 2* test_X.reshape(-1, 1, 28, 28) - 1.
    
    # flatten targets
    train_y = np.hstack(train_y)
    validation_y = np.hstack(validation_y)
    test_y = np.hstack(test_y)
    
    # Onehot the targets
    train_y = np.float32(np.eye(10)[train_y])    
    validation_y = np.float32(np.eye(10)[validation_y])
    test_y = np.float32(np.eye(10)[test_y])
    
    # for hinge loss
    train_y = 2* train_y - 1.
    validation_y = 2* validation_y - 1.
    test_y = 2* test_y - 1.

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
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_net.compute_grads(loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,mlp)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    binary_net.train(
            train_fn,val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_X,train_y,
            validation_X,validation_y,
            test_X,test_y,
            save_path,
            shuffle_parts)