import numpy as np

def NormalizeData(X):
    """
    Normalize each column of X by its L2 norm
    """
    nFea, nSmp = X.shape
    ProcessData = np.zeros((nFea, nSmp))
    
    for i in range(nSmp):
        norm_val = max(1e-12, np.linalg.norm(X[:, i]))
        ProcessData[:, i] = X[:, i] / norm_val
    
    return ProcessData

