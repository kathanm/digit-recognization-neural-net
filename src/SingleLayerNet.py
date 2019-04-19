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
                sum +=  self.weights[(ni, n)] * ni.value
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

        for n in self.inputLayer:
            for i1, n1 in enumerate(self.hiddenLayer):
                derivativeOfCostByActivation = 0
                for i2, n2 in enumerate(self.outputLayer):
                    z = n1.value * self.weights[(n1, n2)] + n2.bias
                    derivativeByZ = util.sigmoid_derivative(z) * (n2.value - expectedOutput[i2])
                    gradient = n1.value * derivativeByZ
                    derivativeOfCostByActivation += self.weights[(n1, n2)] * derivativeByZ
                    n2.bias -= self.learningRate * derivativeByZ
                    self.weights[(n1, n2)] -= self.learningRate * gradient
                z = n.value * self.weights[(n, n1)] + n1.bias
                derivativeByZ = util.sigmoid_derivative(z) * derivativeOfCostByActivation
                gradient = n.value * derivativeByZ
                n1.bias -= self.learningRate * derivativeByZ
                self.weights[(n, n1)] -= self.learningRate * gradient



sln = SingleLayerNet(3, 3, 3)
