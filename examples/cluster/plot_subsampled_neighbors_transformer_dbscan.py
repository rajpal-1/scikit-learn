"""
====================================================================
Comparison of DBSCAN and DBSCAN with Subsampling
====================================================================

We want to compare the performance of DBSCAN and DBSCAN with subsampling.
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

random_state = 0

batch_size = 45
n_features = 2
centers = [[0]*n_features, [-1]*n_features, [0,-1]*int(n_features/2)]
X, labels = datasets.make_blobs(n_samples=3000, n_features=n_features, centers=centers, cluster_std=0.1)

# #############################################################################
# Hyperparameters

min_samples = 10
eps = 0.35
s = 0.01

# #############################################################################
# Compute clustering with DBSCAN 

dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto')

t0 = time.time()
labels_dbscan = dbscan.fit_predict(X)
t_dbscan = time.time() - t0
print('dbscan', t_dbscan)
rand_dbscan = adjusted_rand_score(labels_dbscan, labels)
mi_dbscan = adjusted_mutual_info_score(labels_dbscan, labels)

# #############################################################################
# Compute clustering with DBSCAN and subsampled neighbors

dbscan_subsampled = DBSCAN(eps=eps, min_samples=min_samples * s, metric='precomputed')
snt = SubsampledNeighborsTransformer(s=s, eps=eps, random_state=random_state)

t0 = time.time()
X_subsampled = snt.fit_transform(X)
labels_subsampled = dbscan_subsampled.fit_predict(X_subsampled)
t_subsampled = time.time() - t0
print('t_subsampled', t_subsampled)
rand_subsampled = adjusted_rand_score(labels_subsampled, labels)
mi_subsampled = adjusted_mutual_info_score(labels_subsampled, labels)

# # # #############################################################################
# # # Plot result

fig = plt.figure(figsize=(6, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['blue', 'green', 'red']

# DBSCAN
ax = fig.add_subplot(1, 2, 1)
for cluster, c in zip(np.unique(labels_dbscan), colors):
    ax.plot(X[labels_dbscan == cluster, 0], X[labels_dbscan == cluster, 1], 'w', markerfacecolor=c, marker='.')
ax.set_title('DBSCAN')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-1.35, .2, 'Runtime: %.3fs' % t_dbscan)
plt.text(-1.35, .1, 'ARI: %.1f' % rand_dbscan)
plt.text(-1.35, 0, 'AMI: %.1f' % mi_dbscan)

# DBSCAN with subsampling
ax = fig.add_subplot(1, 2, 2)
for cluster, c in zip(np.unique(labels_subsampled), colors):
    ax.plot(X[labels_subsampled == cluster, 0], X[labels_subsampled == cluster, 1], 'w', markerfacecolor=c, marker='.')
ax.set_title('DBSCAN with subsampling')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-1.35, .2, 'Runtime: %.3fs' % t_subsampled)
plt.text(-1.35, .1, 'ARI: %.1f' % rand_subsampled)
plt.text(-1.35, 0, 'AMI: %.1f' % mi_subsampled)

plt.show()
