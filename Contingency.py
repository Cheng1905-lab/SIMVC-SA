import numpy as np

def Contingency(Mem1, Mem2):
    """
    Create contingency matrix from two membership vectors
    """
    if len(Mem1.shape) > 1 or len(Mem2.shape) > 1:
        raise ValueError('Contingency: Requires two vector arguments')
    
    # Ensure labels start from 1 (MATLAB convention)
    Mem1 = Mem1 - np.min(Mem1) + 1
    Mem2 = Mem2 - np.min(Mem2) + 1
    
    Cont = np.zeros((int(np.max(Mem1)), int(np.max(Mem2))))
    
    for i in range(len(Mem1)):
        Cont[int(Mem1[i]) - 1, int(Mem2[i]) - 1] = Cont[int(Mem1[i]) - 1, int(Mem2[i]) - 1] + 1
    
    return Cont

