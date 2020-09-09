from scipy.sparse import csr_matrix
import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import SubsampledNeighborsTransformer
from numpy.random import multivariate_normal
from sklearn.metrics import pairwise_distances

t0 = time.time()
x = csr_matrix(([1,0,2,7], [1,0,1,0],[0,2,4]), shape=(2, 2), dtype=np.float)
# dbscan_subsampled = DBSCAN(eps=0.1, min_samples=2, metric='precomputed')
# result_subsampled = dbscan_subsampled.fit_predict(x)
t1 = time.time()
print(t1-t0)

# t0 = time.time()
# x = csr_matrix(([1,3,7,2], [1,0,1,0],[0,2,4]), shape=(2, 2), dtype=np.float)
# print(x)
# dbscan_subsampled = DBSCAN(eps=0.1, min_samples=2, metric='precomputed')
# result_subsampled = dbscan_subsampled.fit_predict(x)
# t1 = time.time()
# print(t1-t0)

X = [multivariate_normal([0, 0], [[1, 0], [0, 1]]) for i in range(10000)]

t0 = time.time()
snt = SubsampledNeighborsTransformer(eps=100.0, s=0.1)
result = snt.fit_transform(X)
t1 = time.time()
print(t1-t0)