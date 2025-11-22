import numpy as np

def compute_f(T, H):
    """
    Compute F-score, Precision, and Recall
    """
    if len(T) != len(H):
        raise ValueError(f"Length mismatch: T={len(T)}, H={len(H)}")
    
    N = len(T)
    numT = 0
    numH = 0
    numI = 0
    
    for n in range(N):
        Tn = (T[n+1:] == T[n])
        Hn = (H[n+1:] == H[n])
        numT += np.sum(Tn)
        numH += np.sum(Hn)
        numI += np.sum(Tn * Hn)
    
    p = 1
    r = 1
    f = 1
    
    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p + r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    
    return f, p, r

