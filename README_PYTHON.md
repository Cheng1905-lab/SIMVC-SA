# SIMVC-SA Python 版本

这是 SIMVC-SA 算法的 Python 实现，从原始 MATLAB 代码转换而来。

## 依赖要求

安装所需的 Python 包：

```bash
pip install -r requirements.txt
```

## 文件结构

- `run.py` - 主运行文件
- `miss_algo.py` - 核心算法实现
- `findindex.py` - 数据索引查找函数
- `initialize.py` - 初始化函数
- `NormalizeData.py` - 数据归一化函数
- `EProjSimplex_new.py` - 单纯形投影函数
- `measure/` - 评估指标模块
  - `Clustering8Measure.py` - 8种聚类评估指标
  - `bestMap.py` - 最佳标签映射
  - `MutualInfo.py` - 互信息计算
  - `compute_f.py` - F-score 计算
  - `compute_nmi.py` - NMI 计算
  - `RandIndex.py` - Rand Index 计算
  - `Contingency.py` - 列联表计算

## 使用方法

1. 确保数据集路径正确（在 `run.py` 中修改 `dsPath` 变量）
2. 运行主程序：

```bash
python run.py
```

## 主要改动说明

1. **数据加载**：使用 `scipy.io.loadmat` 加载 MATLAB .mat 文件
2. **数组索引**：MATLAB 使用 1-based 索引，Python 使用 0-based 索引，已做相应转换
3. **矩阵运算**：使用 NumPy 进行矩阵运算
4. **聚类算法**：使用 scikit-learn 的 KMeans 替代 MATLAB 的 kmeans
5. **匈牙利算法**：使用 scipy 的 `linear_sum_assignment` 替代原始实现

## 注意事项

- 确保数据集文件路径正确
- 数据集应包含 `data`、`truelabel` 和 `index` 字段
- 代码已处理 MATLAB cell 数组到 Python list 的转换

