import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    input = np.asarray(x)
    return np.maximum(0, input)