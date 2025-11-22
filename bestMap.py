import numpy as np
from scipy.optimize import linear_sum_assignment

def bestMap(L1, L2):
    """
    bestmap: permute labels of L2 to match L1 as good as possible
    """
    L1 = L1.flatten()
    L2 = L2.flatten()
    if len(L1) != len(L2):
        raise ValueError('size(L1) must == size(L2)')
    
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum((L1 == Label1[i]) & (L2 == Label2[j]))
    
    # Use scipy's Hungarian algorithm (linear_sum_assignment)
    # We want to maximize the sum, so we negate G
    # Pad G if necessary to make it square
    if nClass1 != nClass2:
        G_padded = np.zeros((nClass, nClass))
        G_padded[:nClass1, :nClass2] = G
        row_ind, col_ind = linear_sum_assignment(-G_padded)
    else:
        row_ind, col_ind = linear_sum_assignment(-G)
    
    # Create mapping from Label2 indices to Label1 indices
    c = np.zeros(nClass2, dtype=int)
    for i in range(len(row_ind)):
        if row_ind[i] < nClass2:
            c[row_ind[i]] = col_ind[i]
    
    newL2 = np.zeros(len(L2))
    for i in range(nClass2):
        if c[i] < nClass1:
            newL2[L2 == Label2[i]] = Label1[c[i]]
        else:
            # If no mapping found, keep original label
            newL2[L2 == Label2[i]] = Label2[i]
    
    return newL2

