import numpy as np
from NormalizeData import NormalizeData

def findindex(data, index):
    """
    data: list of views
    index: MATLAB cell array loaded by scipy.io (dtype=object)
    """

    numofview = len(data)
    numofsample = data[0].shape[1]

    # ---- 将 MATLAB 的 cell 转成 Python list ----
    # MATLAB cell 可能是：
    # (1) array([array([...])], dtype=object)
    # (2) array([[array([...]), array([...]), ...]], dtype=object)
    # (3) array([list], dtype=object)
    index_list = []

    if isinstance(index, np.ndarray) and index.dtype == object:
        # 情况 1 & 2: MATLAB cell 变成二维 / 一维 object array
        for item in index.flatten():
            index_list.append(np.array(item).astype(int))
    else:
        # 已经是 Python list 的情况
        index_list = index

    # 现在 index_list 应该是长度 numofview 的 list
    if len(index_list) != numofview:
        raise ValueError(f"index_list 长度 {len(index_list)} != 视图数 {numofview}")

    # ---- 输出变量 ----
    X1 = [None] * numofview
    ind = np.zeros((numofsample, numofview))

    # ---- 遍历每个视图 ----
    for i in range(numofview):

        # MATLAB -> Python: 1-based → 0-based
        idx = np.array(index_list[i]).flatten().astype(int) - 1

        # 保证 index 不越界
        idx = idx[(idx >= 0) & (idx < numofsample)]

        ind[idx, i] = 1

        origin = data[i].copy()
        origin[np.isnan(origin)] = 0
        X1[i] = NormalizeData(origin)

    return X1, ind
