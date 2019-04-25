import numpy as np


def sigmoid(n):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(n):
    return sigmoid(n) * (1 - sigmoid(n))
