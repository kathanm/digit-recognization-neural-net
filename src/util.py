import numpy as np


def sigmoid(n):
    return 1.0 / (1.0 + np.exp(-n))

def sigmoid_derivative(n):
    return sigmoid(n) * (1 - sigmoid(n))
