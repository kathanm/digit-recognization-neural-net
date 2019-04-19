import util

class Neuron:
    def __init__(self, bias):
        self.bias = bias

class SingleLayerNet:
    def __init__(self, inputSize, layerSize, outputSize):
        self.inputLayer = [Neuron(0) for i in xrange(inputSize)]
        self.hiddenLayer = [Neuron(0) for i in xrange(layerSize)]
        self.outputLayer = [Neuron(0) for i in xrange(outputSize)]

        weights = {}
        for n1 in self.inputLayer:
            for n2 in self.hiddenLayer:
                weights[(n1, n2)] = 1

        for n1 in self.hiddenLayer:
            for n2 in self.outputLayer:
                weights[(n1, n2)] = 1
        self.weights = weights
        

sln = SingleLayerNet(3, 3, 3)
print sln.inputLayer