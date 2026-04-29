import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p_t = np.array(p)
    q_t = np.array(q) 
    log = np.log(p_t/q_t)
    summ = np.dot(p_t, log)
    return summ