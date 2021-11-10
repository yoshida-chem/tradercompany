import numpy as np

def identity(x):
    return x

def tanh(x):
    return np.tanh(x)

def sign(x):
    return (x > 0.0) * 1.0

def ReLU(x):
    return sign(x) * x

