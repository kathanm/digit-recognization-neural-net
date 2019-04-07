class Neuron:
    def __init__(self, bias):
        self.bias = bias


class NeuralNet:
    def __init__(self, num_layers, layer_size, input_size, output_size):
        layers = []
        for i in xrange(num_layers + 2):
            layers.append([])
        for i in xrange(input_size):
            layers[0].append(Neuron(0))
        for i in xrange(num_layers):
            for j in xrange(layer_size):
                layers[i+1].append(Neuron(0))
        for i in xrange(output_size):
            layers[num_layers+1].append(Neuron(0))
        self.layers = layers
        weights = {}
        for i in xrange(num_layers + 1):
            layer1_size = len(layers[i])
            layer2_size = len(layers[i+1])
            for j in xrange(layer1_size):
                for k in xrange(layer2_size):
                    weights[(layers[i][j], layers[i+1][k])] = 1
        self.weights = weights

n = NeuralNet(1,1,1,1)
