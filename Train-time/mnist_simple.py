import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

class Layer(object):
    def __init__(self, W_init, b_init, activation):
        n_out, n_in = W_init.shape
        assert b_init.shape == (n_out,)

        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               name="W")
        self.b = theano.shared(value=b_init.reshape(n_out, 1).astype(theano.config.floatX),
                               name="b",
                               broadcastable=(False, True))
        self.activation = activation
        self.params = [self.W, self.b]

    def output(self, x):
        lin_output = T.dot(self.W, x) + self.b
        if self.activation:
            lin_output = self.activation(lin_output)
        return lin_output


class MLP(object):
    def __init__(self, W_init, b_init, activations):
        assert len(W_init) == len(b_init) == len(activations)

        self.layers = []
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        return T.sum((self.output(x) - y) ** 2)


def gradient_descent(cost, params, learning_rate):
    updates = []

    for param in params:
        step = - learning_rate * theano.grad(cost, param)
        updates.append((param, param + step))

    return updates


if __name__ == "__main__":
    # Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
    np.random.seed(0)
    # Number of points
    N = 1000
    # Labels for each cluster
    y = np.random.randint(0, 2, N)
    print(y.shape)
    # Mean of each cluster
    means = np.array([[-1, 1], [-1, 1]])
    # Covariance (in X and Y direction) of each cluster
    covariances = np.random.random_sample((2, 2)) + 1
    # Dimensions of each point
    X = np.vstack([np.random.randn(N) * covariances[0, y] + means[0, y],
                   np.random.randn(N) * covariances[1, y] + means[1, y]]).astype(theano.config.floatX)
    print(X.shape)
    # Convert to targets, as floatX
    y = y.astype(theano.config.floatX)
    # Plot the data
    plt.figure(figsize=(8, 8))
    plt.scatter(X[0, :], X[1, :], c=y, lw=.3, s=3, cmap=plt.cm.cool)
    plt.axis([-6, 6, -6, 6])
    plt.show()

    layer_sizes = [X.shape[0], X.shape[0] * 2, 1]
    W_init = []
    b_init = []
    activations = []

    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_init.append(np.random.randn(n_output, n_input))
        b_init.append(np.ones(n_output))
        activations.append(T.nnet.sigmoid)

    mlp = MLP(W_init, b_init, activations)
    mlp_in = T.matrix("mlp_in")
    mlp_target = T.vector("mlp_target")

    learning_rate = 0.001

    cost = mlp.squared_error(mlp_in, mlp_target)
    updates = gradient_descent(cost, mlp.params, learning_rate)

    train = theano.function([mlp_in, mlp_target], cost, updates=updates)
    mlp_out = theano.function([mlp_in], mlp.output(mlp_in))

    max_iteration = 20
    for iteration in range(max_iteration):
        current_cost = train(X, y)
        current_output = mlp_out(X)
        accuracy = np.mean((current_output > .5) == y)
        print('Cost: {:.3f}, Accuracy: {:.3f}'.format(float(current_cost), accuracy))