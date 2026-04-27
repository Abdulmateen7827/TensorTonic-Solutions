import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    input = np.asarray(x)
    erf = np.vectorize(math.erf)
    return 0.5 * input * (1 + erf(input/(2**0.5)))
