import numpy as np
import scipy.io as sio
import time
import warnings 
from miss_algo import miss_algo
from findindex import findindex
from sklearn.cluster import KMeans
from measure.Clustering8Measure import Clustering8Measure

warnings.filterwarnings('ignore')

dsPath = 'E:/多视图代码/SIMVC-SA-main/SIMVC-SA-main/dataset/'
Incom_rate = ['_Per0.1', '_Per0.2', '_Per0.3', '_Per0.4', '_Per0.5', '_Per0.6', '_Per0.7', '_Per0.8', '_Per0.9']
dataname = ['BDGP_fea']
lambda_vals = [10**-4, 10**-2, 1, 10**2, 10**4]
mu_vals = [0, 10**-4, 10**-2, 1, 10**2, 10**4]

for dsi in range(len(dataname)):
    ResBest = np.zeros((10, 8))
    StdBest = np.zeros((10, 8))
    for ir in range(len(Incom_rate)):
        # load data & make folder
        dataName = dataname[dsi]
        rate = Incom_rate[ir]
        print(f"{dataName}{rate}")
        
        # Load .mat file
        mat_data = sio.loadmat(f"{dsPath}{dataName}{rate}.mat", squeeze_me=False)
        # Handle data - it might be a cell array in MATLAB
        X = mat_data['data']
        # Convert to list if it's a numpy array (MATLAB cell array)
        if isinstance(X, np.ndarray):
            if X.dtype == object:
                # MATLAB cell array stored as object array
                if X.ndim == 2:
                    X = [X[i, 0] for i in range(X.shape[0])]
                elif X.ndim == 1:
                    X = [X[i] for i in range(X.shape[0])]
                else:
                    X = [X]
            else:
                X = [X]
        elif not isinstance(X, list):
            X = [X]
        
        # Handle truelabel
        truelabel = mat_data['truelabel']
        if isinstance(truelabel, np.ndarray):
            if truelabel.dtype == object:
                # MATLAB cell array
                if truelabel.ndim >= 2:
                    Y = truelabel[0, 0].flatten()
                else:
                    Y = truelabel[0].flatten()
            else:
                Y = truelabel.flatten()
        else:
            Y = np.array(truelabel).flatten()
        
        k = len(np.unique(Y))
        numview = len(X)
        
        # para setting
        selectanchor = [1 * k, 2 * k, 5 * k]
        ACC = np.zeros((len(selectanchor), len(lambda_vals), len(mu_vals)))
        NMI = np.zeros((len(selectanchor), len(lambda_vals), len(mu_vals)))
        Purity = np.zeros((len(selectanchor), len(lambda_vals), len(mu_vals)))
        
        tic = time.time()
        X1, H = findindex(X, mat_data['index'])
        time1 = time.time() - tic
        
        term1 = 0
        id = 1
        runtime = []
        
        for ichor in range(len(selectanchor)):
            temp_anchor = selectanchor[ichor]
            for il in range(len(lambda_vals)):
                for im in range(len(mu_vals)):
                    temp_lambda = lambda_vals[il]
                    temp_mu = mu_vals[im]
                    
                    tic = time.time()
                    U, V, A, F, iter_count, obj = miss_algo(X1, Y, temp_anchor, temp_lambda, temp_mu, H)
                    # Normalize U (each row by its L2 norm)
                    row_norms = np.sqrt(np.sum(U**2, axis=1, keepdims=True))
                    row_norms[row_norms == 0] = 1  # Avoid division by zero
                    U = U / row_norms
                    time2 = time.time() - tic
                    
                    MAXiter = 1000  # Maximum number of iterations for KMeans
                    REPlic = 20  # Number of replications for KMeans
                    
                    tic = time.time()
                    res = np.zeros((10, 8))
                    for rep in range(10):
                        kmeans = KMeans(n_clusters=k, max_iter=MAXiter, n_init=REPlic, random_state=rep)
                        pY = kmeans.fit_predict(U)
                        res[rep, :] = Clustering8Measure(Y, pY)
                    time3 = time.time() - tic
                    
                    runtime.append(time1 + time2 + time3 / 10)
                    id += 1
                    
                    tempRes = np.mean(res, axis=0)
                    tempStd = np.std(res, axis=0)
                    ACC[ichor, il, im] = tempRes[0]
                    NMI[ichor, il, im] = tempRes[1]
                    Purity[ichor, il, im] = tempRes[2]
                    
                    for tempIndex in range(8):
                        if tempRes[tempIndex] > ResBest[ir, tempIndex]:
                            if tempIndex == 0:
                                newF = F
                                newU = U
                                objection = obj
                            ResBest[ir, tempIndex] = tempRes[tempIndex]
                            StdBest[ir, tempIndex] = tempStd[tempIndex]
                
                aRuntime = np.mean(runtime)
                PResBest = ResBest[ir, :]
                PStdBest = StdBest[ir, :]
        
        print(f'Res: {PResBest[0]:12.6f} {PResBest[1]:12.6f} {PResBest[2]:12.6f}')
        runtime = []

