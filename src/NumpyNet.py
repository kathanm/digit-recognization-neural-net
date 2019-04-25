# Based heavily on code from Michael Neielsen's book "Neural Nets and Deep Learning"

import util
import numpy as np

class Net:
    # LayerSizes is the list of all layer sizes. For example, a net with input size 100, one hidden layer with 20 nodes,
    # and an output layer with 5 can be initialized with layers sizes =  [100, 20, 5]
    def __init__(self, layerSizes):
        self.layerSizes = layerSizes
        self.biases = [np.random.randn(size, 1) for size in layerSizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layerSizes[:-1], layerSizes[1:])]

    # Input to the function is input to the net
    # Outputs activations of output layer
    def feedforward(self, a):
        params = zip(self.weights, self.biases)
        for w, b in params:
            a = util.sigmoid(np.dot(w, a))
        return a

    # x is the input
    # y is the expected values
    # returns the gradient vector
    def backprop(self, x, y):
        # Changes in weights and biases
        deltaWeights = [np.zeros(w.shape) for w in self.weights]
        deltaBiases = [np.zeros(b.shape) for b in self.biases]

        # feed forward
        params = zip(self.weights, self.biases)
        activation = x
        activations = [x]
        zs = []
        for w, b in params:
            z = np.matmul(w, activation) + b
            zs.append(z)
            activation = util.sigmoid(z)
            activations.append(activation)

        # backprop

        # delta is dc/da * da/dz = dc/dz for last layer
        # because dz/db = 1 delta is also equal to dc/db
        delta = (y - activations[-1]) * util.sigmoid_derivative(zs[-1])
        deltaBiases[-1] = delta
        deltaWeights[-1] = np.matmul(delta, activations[-2].transpose())

        for layer in xrange(2, len(self.layerSizes)):
            z = zs[-layer]
            sp = util.sigmoid_derivative(z)
            delta = np.matmul(self.weights[-layer + 1].transpose(), delta) * sp
            deltaBiases[-layer] = delta
            deltaWeights[-layer] = np.matmul(delta, activations[-layer - 1].transpose())

        return (deltaWeights, deltaBiases)


n = Net([1, 2, 3])
a = np.array([3])
a.shape = (1,1)
y = np.array([1, 0, 0])
y.shape = (3, 1)
n.backprop(a, y)
