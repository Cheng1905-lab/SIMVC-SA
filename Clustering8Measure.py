import numpy as np
from .bestMap import bestMap
from .MutualInfo import MutualInfo
from .compute_f import compute_f
from .compute_nmi import compute_nmi
from .RandIndex import RandIndex

def Clustering8Measure(Y, predY):
    """
    Compute 8 clustering evaluation measures
    Returns: [ACC, nmi, Purity, Fscore, Precision, Recall, AR, Entropy]
    """
    Y = Y.flatten()
    predY = predY.flatten()
    
    n = len(Y)
    
    uY = np.unique(Y)
    nclass = len(uY)
    Y0 = np.zeros(n)
    if nclass != np.max(Y):
        for i in range(nclass):
            Y0[Y == uY[i]] = i
        Y = Y0
    
    uY = np.unique(predY)
    nclass = len(uY)
    predY0 = np.zeros(n)
    if nclass != np.max(predY):
        for i in range(nclass):
            predY0[predY == uY[i]] = i
        predY = predY0
    
    Lidx = np.unique(Y)
    classnum = len(Lidx)
    predLidx = np.unique(predY)
    pred_classnum = len(predLidx)
    
    # purity
    correnum = 0
    for ci in range(pred_classnum):
        incluster = Y[predY == predLidx[ci]]
        if len(incluster) > 0:
            inclunub = np.bincount(incluster.astype(int), minlength=int(np.max(incluster)) + 1)
            if len(inclunub) > 0:
                correnum += np.max(inclunub)
    Purity = correnum / len(predY)
    
    res = bestMap(Y, predY)
    # accuracy
    ACC = np.sum(Y == res) / len(Y)
    # NMI
    MIhat = MutualInfo(Y, res)
    
    Fscore, Precision, Recall = compute_f(Y, predY)
    nmi, Entropy = compute_nmi(Y, predY)
    AR = RandIndex(Y, predY)
    
    result = np.array([ACC, nmi, Purity, Fscore, Precision, Recall, AR, Entropy])
    return result

