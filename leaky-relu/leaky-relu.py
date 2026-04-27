import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    input = np.asarray(x)
    output = np.where(input >= 0, input, input*alpha)
    return output