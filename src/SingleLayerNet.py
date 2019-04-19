import util

class Neuron:
    def __init__(self, bias):
        self.bias = bias

class SingleLayerNet:
    def __init__(self, inputSize, layerSize, outputSize):
        self.inputLayer = [Neuron(0) for i in xrange(inputSize)]

sln = SingleLayerNet(3, 3, 3)
print sln.inputLayer