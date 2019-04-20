import util
import pickle
import numpy as np

class Neuron:
    def __init__(self, bias):
        self.value = None
        self.bias = bias


class SingleLayerNet:
    def __init__(self, inputSize, layerSize, outputSize, learningRate=0.0001):
        self.inputLayer = [Neuron(0) for i in xrange(inputSize)]
        self.hiddenLayer = [Neuron(0) for i in xrange(layerSize)]
        self.outputLayer = [Neuron(0) for i in xrange(outputSize)]
        self.learningRate = learningRate

        weights = {}
        for n1 in self.inputLayer:
            for n2 in self.hiddenLayer:
                weights[(n1, n2)] = 1

        for n1 in self.hiddenLayer:
            for n2 in self.outputLayer:
                weights[(n1, n2)] = 1
        self.weights = weights

    # Input is a list of numbers
    def readInput(self, input):
        if len(input) != len(self.inputLayer):
            raise Exception()
        for i in xrange(len(self.inputLayer)):
            self.inputLayer[i].value = input[i]

    def feedForward(self):
        for n in self.hiddenLayer:
            sum = n.bias
            for ni in self.inputLayer:
                sum += self.weights[(ni, n)] * ni.value
            n.value = util.sigmoid(sum)

        for n in self.outputLayer:
            sum = n.bias
            for nh in self.hiddenLayer:
                sum += self.weights[(nh, n)] * nh.value
            n.value = util.sigmoid(sum)

    # expectedOutput is a list of expected output values corresponding to each output neuron
    def backProp(self, expectedOutput):
        if len(expectedOutput) != len(self.outputLayer):
            raise Exception()
        result = zip(self.outputLayer, expectedOutput)

        hiddenLayerActivationDerviative = []
        for n1 in self.hiddenLayer:
            derivativeOfCostByActivation = 0
            for i2, n2 in enumerate(self.outputLayer):
                z = n1.value * self.weights[(n1, n2)] + n2.bias
                derivativeByZ = util.sigmoid_derivative(z) * (n2.value - expectedOutput[i2])
                gradient = n1.value * derivativeByZ
                derivativeOfCostByActivation += self.weights[(n1, n2)] * derivativeByZ
                n2.bias -= self.learningRate * derivativeByZ
                self.weights[(n1, n2)] -= self.learningRate * gradient
            hiddenLayerActivationDerviative.append(derivativeOfCostByActivation)

        for n in self.inputLayer:
            for i1, n1 in enumerate(self.hiddenLayer):
                z = n.value * self.weights[(n, n1)] + n1.bias
                derivativeByZ = util.sigmoid_derivative(z) * hiddenLayerActivationDerviative[i1]
                gradient = n.value * derivativeByZ
                n1.bias -= self.learningRate * derivativeByZ
                self.weights[(n, n1)] -= self.learningRate * gradient

def main():
    image_size = 28
    no_of_different_labels = 10
    image_pixels = image_size * image_size
    train_data = np.loadtxt(
        '..\\resources\\train.csv', None, '#', ',')
    test_data = np.loadtxt('..\\resources\\test.csv',
                           None, '#', ',')
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    lr = np.arange(no_of_different_labels)
    train_labels_one_hot = (lr == train_labels).astype(np.float)
    test_labels_one_hot = (lr == test_labels).astype(np.float)
    train_labels_one_hot[train_labels_one_hot == 0] = 0.01
    train_labels_one_hot[train_labels_one_hot == 1] = 0.99
    test_labels_one_hot[test_labels_one_hot == 0] = 0.01
    test_labels_one_hot[test_labels_one_hot == 1] = 0.99

    sln = SingleLayerNet(image_pixels, 20, no_of_different_labels)
    count = 0
    for i in range(len(train_imgs)):
        sln.readInput(train_imgs[i])
        sln.feedForward()
        sln.backProp(train_labels_one_hot[i])
        print count
        count = count + 1

    with open('sln.pkl', 'wb') as output:
        pickle.dump(sln, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
