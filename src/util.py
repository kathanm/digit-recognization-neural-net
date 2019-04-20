import math


def sigmoid(n):
    if n < 0:
        return 1 - 1 / (1 + math.exp(n))
    return 1 / (1 + math.exp(-n))


def sigmoid_derivative(n):
    return sigmoid(n) * (1 - sigmoid(n))
