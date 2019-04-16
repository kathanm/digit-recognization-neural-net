import math


def sigmoid(n):
    return 1 / (1 + math.exp(-1 * n))

def sigmoid_derivative(n):
    return sigmoid(n) * (1 - sigmoid(n))
