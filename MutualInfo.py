import numpy as np

def MutualInfo(L1, L2):
    """
    mutual information
    """
    L1 = L1.flatten()
    L2 = L2.flatten()
    if len(L1) != len(L2):
        raise ValueError('size(L1) must == size(L2)')
    
    L1 = L1 - np.min(L1) + 1  # min (L1) <- 1
    L2 = L2 - np.min(L2) + 1  # min (L2) <- 1
    
    nClass = max(np.max(L1), np.max(L2))
    G = np.zeros((nClass, nClass))
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((L1 == i+1) & (L2 == j+1)) + np.finfo(float).eps
    
    sumG = np.sum(G)
    
    # calculate MIhat
    P1 = np.sum(G, axis=1)
    P1 = P1 / sumG
    P2 = np.sum(G, axis=0)
    P2 = P2 / sumG
    
    H1 = -np.sum(P1 * np.log2(P1 + np.finfo(float).eps))
    H2 = -np.sum(P2 * np.log2(P2 + np.finfo(float).eps))
    
    P12 = G / sumG
    PPP = P12 / np.tile(P2, (nClass, 1)) / np.tile(P1.reshape(-1, 1), (1, nClass))
    PPP[np.abs(PPP) < 1e-12] = 1
    MI = np.sum(P12 * np.log2(PPP + np.finfo(float).eps))
    MIhat = MI / max(H1, H2)
    
    MIhat = np.real(MIhat)
    return MIhat

