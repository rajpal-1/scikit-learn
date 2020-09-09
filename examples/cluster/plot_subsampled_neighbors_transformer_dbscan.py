"""
====================================================================
Comparison of the K-Means and MiniBatchKMeans clustering algorithms
====================================================================

We want to compare the performance of the MiniBatchKMeans and KMeans:
the MiniBatchKMeans is faster, but gives slightly different results (see
:ref:`mini_batch_kmeans`).

We will cluster a set of data, first with KMeans and then with
MiniBatchKMeans, and plot the results.
We will also plot the points that are labelled differently between the two
algorithms.
"""
print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.neighbors import SubsampledNeighborsTransformer
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn import datasets

from numpy.random import multivariate_normal

# #############################################################################
# Generate sample data

np.random.seed(0)

batch_size = 45
n_features = 2
centers = [[0]*n_features, [-1]*n_features, [0,-1]*int(n_features/2)]
X, labels = datasets.make_blobs(n_samples=3000, n_features=n_features, centers=centers, cluster_std=0.1)
# X = np.array([multivariate_normal([2, 2, 2, 2, 2, 2, 2, 2], [[0.2, 0, 0, 0, 0, 0, 0, 0], [0, 0.2, 0, 0, 0, 0, 0, 0], [0, 0, 0.2, 0, 0, 0, 0, 0], [0, 0, 0, 0.2, 0, 0, 0, 0], [0, 0, 0, 0, 0.2, 0, 0, 0], [0, 0, 0, 0, 0, 0.2, 0, 0], [0, 0, 0, 0, 0, 0, 0.2, 0], [0, 0, 0, 0, 0, 0, 0, 0.2]]) for i in range(3000)] + \
#     [multivariate_normal([-2, -2, -2, -2, -2, -2, -2, -2], [[0.2, 0, 0, 0, 0, 0, 0, 0], [0, 0.2, 0, 0, 0, 0, 0, 0], [0, 0, 0.2, 0, 0, 0, 0, 0], [0, 0, 0, 0.2, 0, 0, 0, 0], [0, 0, 0, 0, 0.2, 0, 0, 0], [0, 0, 0, 0, 0, 0.2, 0, 0], [0, 0, 0, 0, 0, 0, 0.2, 0], [0, 0, 0, 0, 0, 0, 0, 0.2]]) for i in range(3000)] + \
#     [multivariate_normal([2, -2, -2, 2, 2, -2, -2, 2], [[0.2, 0, 0, 0, 0, 0, 0, 0], [0, 0.2, 0, 0, 0, 0, 0, 0], [0, 0, 0.2, 0, 0, 0, 0, 0], [0, 0, 0, 0.2, 0, 0, 0, 0], [0, 0, 0, 0, 0.2, 0, 0, 0], [0, 0, 0, 0, 0, 0.2, 0, 0], [0, 0, 0, 0, 0, 0, 0.2, 0], [0, 0, 0, 0, 0, 0, 0, 0.2]]) for i in range(3000)])
# labels = [0]*3000 + [1]*3000 + [2]*3000

# #############################################################################
# Hyperparameters

min_samples = 10
eps = 0.3
s = 0.1

# #############################################################################
# Compute clustering with DBSCAN 

dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto')

t0 = time.time()
result_dbscan = dbscan.fit_predict(X)
t_dbscan = time.time() - t0
print('dbscan', t_dbscan)
print(adjusted_rand_score(result_dbscan, labels), adjusted_mutual_info_score(result_dbscan, labels))

# #############################################################################
# Compute clustering with DBSCAN and subsampled neighbors

dbscan_subsampled = DBSCAN(eps=eps, min_samples=min_samples * s, metric='precomputed')
snt = SubsampledNeighborsTransformer(s=s, eps=eps)

t0 = time.time()
X_subsampled = snt.fit_transform(X)
result_subsampled = dbscan_subsampled.fit_predict(X_subsampled)
t_subsampled = time.time() - t0

print('sng-dbscan', t_subsampled)
print(adjusted_rand_score(result_subsampled, labels), adjusted_mutual_info_score(result_subsampled, labels))

# # # # #############################################################################
# # # # Plot result

# fig = plt.figure(figsize=(8, 3))
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
# colors = ['cyan', 'red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'gray', 'magenta', 'black']

# # DBSCAN
# ax = fig.add_subplot(1, 3, 1)
# print(len(np.unique(result_dbscan)), len(colors))
# for cluster, c in zip(np.unique(result_dbscan), colors):
#     ax.plot(X[result_dbscan == cluster, 0], X[result_dbscan == cluster, 1], 'w', markerfacecolor=c, marker='.')
# ax.set_title('DBSCAN')
# ax.set_xticks(())
# ax.set_yticks(())
# plt.text(-1.2, .2, 'runtime: %.2fs' % t_dbscan)

# # DBSCAN with subsampling
# ax = fig.add_subplot(1, 3, 2)
# print(len(np.unique(result_subsampled)), len(colors))
# for cluster, c in zip(np.unique(result_subsampled), colors):
#     ax.plot(X[result_subsampled == cluster, 0], X[result_subsampled == cluster, 1], 'w', markerfacecolor=c, marker='.')
# ax.set_title('DBSCAN with subsampling')
# ax.set_xticks(())
# ax.set_yticks(())
# plt.text(-1.2, .2, 'runtime: %.2fs' % t_subsampled)

# plt.show()

# # #############################################################################
# # Load Iris dataset
# iris = datasets.load_iris()
# X = iris.data
# labels = iris.target

# # #############################################################################
# # Compute clustering with DBSCAN 

# # We use = 1.57, which was found using grid search.
# dbscan = DBSCAN(eps=1.57)

# t0 = time.time()
# result_dbscan = dbscan.fit_predict(X)
# t_dbscan = time.time() - t0
# print(adjusted_rand_score(result_dbscan, labels), adjusted_mutual_info_score(result_dbscan, labels), t_dbscan)

# # #############################################################################
# # Compute clustering with DBSCAN and subsampled neighbors

# dbscan_subsampled = DBSCAN(eps=1.57, metric='precomputed')
# snt = SubsampledNeighborsTransformer(s=0.3)

# t0 = time.time()
# X_subsampled = snt.fit_transform(X)
# print(time.time() - t0)
# result_subsampled = dbscan_subsampled.fit_predict(X_subsampled)
# t_subsampled = time.time() - t0
# print(adjusted_rand_score(result_subsampled, labels), adjusted_mutual_info_score(result_subsampled, labels), t_subsampled)

# # #############################################################################
# # Plot result

# fig = plt.figure(figsize=(8, 3))
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
# colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# # We want to have the same colors for the same cluster from
# # DBSCAN and subsampled neighbors DBSCAN. Let's pair the cluster centers per
# # closest one.

# plt.show()

# # #############################################################################
# # Load Olivetti Faces dataset
# faces = datasets.fetch_olivetti_faces()
# X = faces.data
# labels = faces.target

# # #############################################################################
# # Compute clustering with DBSCAN 

# dbscan = DBSCAN()
# t0 = time.time()
# result_dbscan = dbscan.fit_predict(X)
# t_dbscan = time.time() - t0

# # #############################################################################
# # Generate subsampled neighborhood graph

# snt = SubsampledNeighborsTransformer(s=0.3)
# neighborhood = snt.fit(X).transform(X)

# # #############################################################################
# # Compute clustering with DBSCAN and subsampled neighbors

# dbscan_subsampled = DBSCAN()
# t0 = time.time()
# dbscan.fit_predict(X)
# t_mini_batch = time.time() - t0

# # #############################################################################
# # Plot result

# fig = plt.figure(figsize=(8, 3))
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
# colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# # Again, let's use the same colors for the same cluster from
# # DBSCAN and subsampled neighbors DBSCAN. 

# plt.show()
