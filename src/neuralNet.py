import util

class Neuron:
    def __init__(self, bias):
        self.bias = bias


class NeuralNet:
    def __init__(self, num_layers, layer_size, input_size, output_size):
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.input_size = input_size
        self.output_size = output_size
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

    def feedforward(self):
        for i in xrange(self.num_layers + 1):
            prev_layer = self.layers[i]
            for neuron in self.layers[i+1]:
                sum = neuron.bias
                for prev_neuron in prev_layer:
                    sum += self.weights[(prev_neuron, neuron)] * prev_neuron.value
                neuron.value = util.sigmoid(sum)


n = NeuralNet(1,1,1,1)
