import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm = (gamma * (x - mean) / np.sqrt(var + eps) )+ beta
    return norm
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch, seq, d_model = Q.shape
    dk = d_model // num_heads
    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v

    q = q.reshape(batch, seq, num_heads, dk).transpose(0, 2, 1, 3) #shape: (b, h, N, dk)
    k = k.reshape(batch, seq, num_heads, dk).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, num_heads, dk).transpose(0, 2, 1, 3)

    scores = q @ k.transpose(0, 1, 3, 2) #shape: (b, h, N, N)
    scores = scores / dk ** 0.5
    attn = softmax(scores) @ v #shape: (b, h, N, dk)
    attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
    out =  attn @ W_o
    return out
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    h1 = np.dot(x, W1) + b1
    h2 = np.maximum(0, h1)
    ffn = np.dot(h2, W2 ) + b2
    return ffn

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_out, gamma1, beta1)

    ffn_out = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ffn_out, gamma2, beta2)
    return x