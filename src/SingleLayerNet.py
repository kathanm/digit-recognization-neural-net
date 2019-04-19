import util

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

    def readInput(self, input):
        if len(input) != len(self.inputLayer):
            raise Exception()
        for i in xrange(len(self.inputLayer)):
            self.inputLayer[i].value = input[i]

    def feedForward(self):
        for n in self.hiddenLayer:
            sum = n.bias
            for ni in self.inputLayer:
                sum +=  self.weights[(ni, n)] * ni.value
            n.value = util.sigmoid(sum)

        for n in self.outputLayer:
            sum = n.bias
            for nh in self.hiddenLayer:
                sum += self.weights[(nh, n)] * nh.value
            n.value = util.sigmoid(sum)

    def backProp(self, expectedOutput):
        if len(expectedOutput) != len(self.outputLayer):
            raise Exception()
        result = zip(self.outputLayer, expectedOutput)

        for n1 in self.hiddenLayer:
            for i2, n2 in enumerate(self.outputLayer):
                z = n1.value * self.weights[(n1, n2)] + n2.bias
                gradient = n1.value * util.sigmoid_derivative(z) * 2 * (n2.value - expectedOutput[i2])
                self.weights[(n1, n2)] -= self.learningRate * gradient


sln = SingleLayerNet(3, 3, 3)
