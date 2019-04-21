import util
import pickle
import numpy as np
import csv

class Neuron:
    def __init__(self, bias):
        self.value = None
        self.bias = bias


class SingleLayerNet:
    def __init__(self, inputSize, layerSize, outputSize, learningRate=10):
        self.inputLayer = [Neuron(0) for i in xrange(inputSize)]
        self.hiddenLayer = [Neuron(0) for i in xrange(layerSize)]
        self.outputLayer = [Neuron(0) for i in xrange(outputSize)]
        self.learningRate = learningRate

        weights = {}
        for n1 in self.inputLayer:
            for n2 in self.hiddenLayer:
                weights[(n1, n2)] = .0001

        for n1 in self.hiddenLayer:
            for n2 in self.outputLayer:
                weights[(n1, n2)] = .01
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

    def getChosenValue(self):
        maxIndex = 0
        maxValue = self.outputLayer[0].value
        for i, n in enumerate(self.outputLayer):
            if n.value > maxValue:
                maxIndex = i
                maxValue = n.value
        return maxIndex

    def getLoss(self, expected):
        sumOfSquares = 0
        for i, n in enumerate(self.outputLayer):
            sumOfSquares +=  (n.value - expected[i]) ** 2
        return sumOfSquares / 2



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

def train_net():
    inputSize = 28 ** 2
    sln = SingleLayerNet(inputSize, 20, 10)
    with open('../resources/train.csv', 'rb') as trainingData:
        reader = csv.reader(trainingData)
        count = 0
        numberCorrect = 0
        batchLoss = 0
        batchLosses = []
        for row in reader:
            input = list(map(int, row))
            expectedOutput = [0] * 10
            expectedOutput[input[0]] = 1
            input2 = input[1:]
            sln.readInput(input2)
            sln.feedForward()
            print("Expected value: " + str(input[0]) + " ----------------- Received Value: " + str(sln.getChosenValue()))
            if input[0] == sln.getChosenValue():
                numberCorrect += 1
            batchLoss += sln.getLoss(expectedOutput)
            if (count % 50 == 0):
                batchLosses.append(batchLoss)
                print ("Batch loss " + str(int(count / 50)) + ": " + str(batchLoss))
                print("Batch losses " + str(batchLosses))
                print("Learning rate: " + str(sln.learningRate))
                batchLoss = 0
                last_5_batches = batchLosses[-3:]
                last_5_avg = sum(last_5_batches) / 3 if len(last_5_batches) == 3 else 100
                if last_5_avg < 35:
                    sln.learningRate = 1 if sln.learningRate > 1 else sln.learningRate
                if last_5_avg < 30:
                    sln.learningRate = .1 if sln.learningRate > .1 else sln.learningRate
                if last_5_avg <  25:
                    sln.learningRate = 0.01 if sln.learningRate > .01 else sln.learningRate
                if last_5_avg < 23:
                    sln.learningRate = 0.001 if sln.learningRate > .001 else sln.learningRate
                if last_5_avg < 22:
                    sln.learningRate = 0.0001 if sln.learningRate > .0001 else sln.learningRate
            sln.backProp(expectedOutput)
            print(count)
            count += 1
            print ("Percent correct: " + str(100 * numberCorrect / count))

    with open('sln.pkl', 'wb') as output:
        pickle.dump(sln, output, pickle.HIGHEST_PROTOCOL)


def run_tests():
    with open('sln.pkl', 'rb') as input:
        sln = pickle.load(input)
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
                    sln.readInput(input)
                    sln.feedForward()
                    max_num = sln.getChosenValue()
                    max_val = sln.outputLayer[0].value
                    writer.writerow([str(count), str(max_num)])
                    count += 1

train_net()
run_tests()

if __name__ == '__main__':
    main()
    run_tests()