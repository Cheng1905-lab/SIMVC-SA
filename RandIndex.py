import numpy as np
from .Contingency import Contingency

def RandIndex(c1, c2):
    """
    RANDINDEX - calculates Rand Indices to compare two partitions
    Returns: [AR, RI, MI, HI]
    AR: adjusted Rand index
    RI: unadjusted Rand index
    MI: Mirkin's index
    HI: Hubert's index
    """
    if len(c1.shape) > 1 or len(c2.shape) > 1:
        raise ValueError('RandIndex: Requires two vector arguments')
    
    C = Contingency(c1, c2)  # form contingency matrix
    
    n = np.sum(C)
    nis = np.sum(np.sum(C, axis=1)**2)  # sum of squares of sums of rows
    njs = np.sum(np.sum(C, axis=0)**2)  # sum of squares of sums of columns
    
    t1 = n * (n - 1) / 2  # total number of pairs of entities
    t2 = np.sum(C**2)  # sum over rows & columns of nij^2
    t3 = 0.5 * (nis + njs)
    
    # Expected index (for adjustment)
    nc = (n * (n**2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    
    A = t1 + t2 - t3  # no. agreements
    D = -t2 + t3  # no. disagreements
    
    if t1 == nc:
        AR = 0  # avoid division by zero; if k=1, define Rand = 0
    else:
        AR = (A - nc) / (t1 - nc)  # adjusted Rand - Hubert & Arabie 1985
    
    RI = A / t1  # Rand 1971 - Probability of agreement
    MI = D / t1  # Mirkin 1970 - p(disagreement)
    HI = (A - D) / t1  # Hubert 1977 - p(agree)-p(disagree)
    
    return AR

