import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos = np.arange(seq_length).reshape(-1, 1) #shape:(seq_len, 1)
    i = np.arange(d_model) #shape: (d_model)    
    denom = 10000 ** (2 * i / d_model)
    angle = pos / denom
    pe = np.zeros((seq_length, d_model))
    pe[:,0::2] = np.sin(angle[:,0::2])
    pe[:,1::2] = np.cos(angle[:,1::2])
    return pe        