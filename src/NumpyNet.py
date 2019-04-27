# Based heavily on code from Michael Neielsen's book "Neural Nets and Deep Learning"

import util
import numpy as np
import csv
import pickle

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
        delta = (activations[-1] - y) * util.sigmoid_derivative(zs[-1])
        deltaBiases[-1] = delta
        deltaWeights[-1] = np.matmul(delta, activations[-2].transpose())

        for layer in xrange(2, len(self.layerSizes)):
            z = zs[-layer]
            sp = util.sigmoid_derivative(z)
            delta = np.matmul(self.weights[-layer + 1].transpose(), delta) * sp
            deltaBiases[-layer] = delta
            deltaWeights[-layer] = np.matmul(delta, activations[-layer - 1].transpose())

        return (deltaWeights, deltaBiases, activations[-1])

# x is actual output
# y is expected outoput
def get_loss(x, y):
    diff = x - y
    return np.sum(np.square(diff))


def train_net():
    # Settings
    layers = [784, 20, 10]
    learning_rate = 0.1
    mini_batch_size = 50
    epochs = 10

    # Initialize neural net with layer sizes
    nn = Net(layers)
    accuracies = []
    for i in xrange(epochs):
        with open('../resources/train.csv', 'rb') as trainingData:
            print "STARTING EPOCH " + str(i)
            reader = csv.reader(trainingData)
            deltaWeights = [np.zeros(w.shape) for w in nn.weights]
            deltaBiases = [np.zeros(b.shape) for b in nn.biases]
            count = 1
            numCorrect = 0
            batchLoss = 0
            for row in reader:
                # Setting up input and output
                input = list(map(int, row))
                expectedOutput = [0] * 10
                expectedOutput[input[0]] = 1
                expectedOutput = np.array(expectedOutput)
                expectedOutput.shape = (10, 1)
                input = [x * (1.0 / 255.0) for x in input]
                input = np.array(input[1:])
                input.shape = (784, 1)

                dw, db, output = nn.backprop(input, expectedOutput)
                if np.argmax(output) == np.argmax(expectedOutput):
                    numCorrect += 1
                batchLoss += get_loss(output, expectedOutput)
                deltaWeights = [tdw + dw for tdw, dw in zip(deltaWeights, dw)]
                deltaBiases = [tdb + db for tdb, db in zip(deltaBiases, db)]

                if count % mini_batch_size == 0:
                    deltaWeights = [dw / mini_batch_size for dw in deltaWeights]
                    deltaBiases = [db / mini_batch_size for db in deltaBiases]
                    nn.weights = [(w - (dw * learning_rate)) for w, dw in zip(nn.weights, deltaWeights)]
                    nn.biases = [(b - (db * learning_rate)) for b, db in zip(nn.biases, deltaBiases)]

                    deltaWeights = [np.zeros(w.shape) for w in nn.weights]
                    deltaBiases = [np.zeros(b.shape) for b in nn.biases]

                    print "Epoch " + str(i)
                    print "Loss for batch " + str(count / mini_batch_size) + ": " + str(batchLoss)
                    batchLoss = 0

                count += 1

            percentCorrect = 100 * numCorrect / (count - 1)
            accuracies.append(percentCorrect)
            print "Epoch accuracy: " + str(percentCorrect)
            print "Accuracies: " + str(accuracies)


    print "Complete Accuracies: " + str(accuracies)

    with open('nn.pkl', 'wb') as output:
        pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)

def test_net():
    with open('nn.pkl', 'rb') as input:
        nn = pickle.load(input)
        with open('../resources/submission.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['ImageId', 'Label'])
            with open('../resources/test.csv', 'rb') as testfile:
                reader = csv.reader(testfile, delimiter=',')
                count = 0
                for row in reader:
                    print("Test: " + str(count))
                    if count == 0:
                        count += 1
                        continue
                    input = list(map(int, row))
                    input = np.array(input)
                    input.shape = (784, 1)
                    output = nn.feedforward(input)
                    result = np.argmax(output)
                    writer.writerow([str(count), str(result)])
                    count += 1

train_net()