import numpy as np
from scipy.linalg import svd
from initialize import initialize
from EProjSimplex_new import EProjSimplex_new

def miss_algo(X, Y, m, lambda_val, mu, H):
    """
    m      : the number of anchor. the size of F is m*n.
    lambda_val : the hyper-parameter of regularization term.
    X      : list of n*di matrices
    """
    maxIter = 100  # the number of iterations
    numview = len(X)
    numsample = len(Y)
    k = len(np.unique(Y))
    
    gamma = np.ones(numview) / numview
    Z_temp = {}
    P, Z, r = initialize(H, m, numsample, numview)
    A = [None] * numview
    f = np.zeros(m)
    tao = np.zeros(numview)
    F = np.zeros((m, numsample))
    if m <= numsample:
        F[:, :m] = np.eye(m)
    else:
        # If m > numsample, just set diagonal elements
        for i in range(min(m, numsample)):
            F[i, i] = 1
    flag = 1
    iter_count = 0
    obj = []
    
    while flag:
        iter_count += 1
        # optimize A
        for v in range(numview):
            M = X[v] @ Z[v].T
            Unew, s, Vnew = svd(M, full_matrices=False)
            A[v] = Unew @ Vnew.T
        
        # optimize F
        M = np.zeros((numsample, m))
        for v in range(numview):
            M = M + Z[v].T @ P[v].T
        Unew, s, Vnew = svd(lambda_val * M, full_matrices=False)
        F = (Unew @ Vnew.T).T
        
        # optimize Z_v
        for v in range(numview):
            temp1 = (gamma[v]**2) * A[v].T @ X[v]
            temp2 = lambda_val * P[v].T @ F
            for j in range(numsample):
                for i in range(m):
                    f[i] = (temp1[i, j] + temp2[i, j]) / (gamma[v]**2 * r[v][j] + lambda_val + mu)
                Z[v][:, j] = EProjSimplex_new(f)
        
        # optimize P
        for v in range(numview):
            W = F @ Z[v].T
            Unew, s, Vnew = svd(W, full_matrices=False)
            P[v] = Unew @ Vnew.T
        
        # optimize gamma
        sum1 = 0
        for v in range(numview):
            tao[v] = np.linalg.norm(X[v] - A[v] @ (Z[v] * np.tile(r[v], (m, 1))), 'fro')**2
            sum1 += 1 / tao[v]
        
        for v in range(numview):
            gamma[v] = 1 / tao[v]
            gamma[v] = gamma[v] / sum1
        
        # objection
        term1 = 0
        for v in range(numview):
            term1 += gamma[v]**2 * tao[v] + lambda_val * np.linalg.norm(P[v] @ Z[v] - F, 'fro')**2 + mu * np.linalg.norm(Z[v], 'fro')**2
        obj.append(term1)
        Z_temp[iter_count] = F
        
        if (iter_count > 9) and (abs((obj[iter_count-2] - obj[iter_count-1]) / abs(obj[iter_count-2])) < 1e-6 or iter_count > maxIter or abs(obj[iter_count-1]) < 1e-6):
            UU, s, V = svd(F.T, full_matrices=False)
            UU = UU[:, :k]
            flag = 0
    
    return UU, V, A, F, iter_count, np.array(obj)

