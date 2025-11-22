import numpy as np

def initialize(H, m, numsample, numview):
    """
    INITIALIZE 此处显示有关此函数的摘要
    """
    r = [None] * numview
    Z = [None] * numview
    np.random.seed(721)
    
    for v in range(numview):
        r[v] = H[:, v].T
        P = np.eye(m)
        Z[v] = np.zeros((m, numsample))
        Z[v][0, :] = 1
    
    # Return P as well (need to create it)
    P_list = [None] * numview
    for v in range(numview):
        P_list[v] = np.eye(m)
    
    return P_list, Z, r

