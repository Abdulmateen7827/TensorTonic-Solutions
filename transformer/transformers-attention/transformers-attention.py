import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    qk = Q @ K.transpose(1,2)  #shape(batch, seq)
    dk = K.shape[-1] ** 0.5 
    att = qk / dk
    return F.softmax(att, dim=-1) @ V