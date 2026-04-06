import numpy as np
import torch.nn as nn

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch, seq, d_model = Q.shape
    dk = d_model // num_heads
    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v

    q = q.reshape(batch, seq, num_heads, dk).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq, num_heads, dk).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, num_heads, dk).transpose(0, 2, 1, 3)
    
    scores = q @ k.transpose(0, 1, 3, 2)
    scores = (scores / scores **0.5) @ v
    attn = softmax(scores)
    out = attn.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)

    return out @ W_o
    