"""
====================================================================
Comparison of DBSCAN and DBSCAN with Subsampling
====================================================================

We want to compare the performance of DBSCAN and DBSCAN with subsampling:
subsampled DBSCAN is faster but the speedup is more apparent with larger
datasets. To run DBSCAN with subsampling, we use
SubsampledNeighborsTransformer to precompute a neighborhood graph with
subsampled edges and pass the graph to DBSCAN with `metric=precomputed`.

We cluster a set of data and plot the results. The edge sampling rate `s`
is chosen to be just large enough to recover the clusters for the maximum
speedup.

In order to compare the results of DBSCAN and DBSCAN with subsampling,
we must set `min_samples` for the latter to `min_samples * s`
because each point in the subsampled neighborhood graph will have on
expectation `s` of its original neighbors.
"""
print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.neighbors import SubsampledNeighborsTransformer
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn import datasets

# #############################################################################
# Generate sample data

np.random.seed(0)

centers = [[-2, -1], [0, 0], [0, -2]]
X, labels = datasets.make_blobs(n_samples=30000, centers=centers,
                                cluster_std=0.2)
X = X.astype(np.float64)

# #############################################################################
# Hyperparameters

eps = 0.3
min_samples = 20
min_samples_sub = 2
s = 0.1

# #############################################################################
# Compute clustering with DBSCAN

dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto')
t0 = time.time()
dbscan.fit(X)
labels_dbscan = dbscan.labels_
t_dbscan = time.time() - t0
rand_dbscan = adjusted_rand_score(labels_dbscan, labels)
mi_dbscan = adjusted_mutual_info_score(labels_dbscan, labels)

# ############################################################################
# Compute clustering with DBSCAN and subsampled neighbors

dbscan_sub = DBSCAN(eps=eps, min_samples=min_samples_sub, metric='precomputed')
snt = SubsampledNeighborsTransformer(s=s, eps=eps)
t0 = time.time()
X_sub = snt.fit_transform(X)
dbscan_sub.fit(X_sub)
labels_sub = dbscan_sub.labels_
t_sub = time.time() - t0
rand_sub = adjusted_rand_score(labels_sub, labels)
mi_sub = adjusted_mutual_info_score(labels_sub, labels)

# # # ########################################################################
# # # Plot result

fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['black', 'lightblue', 'red', 'orange']

# DBSCAN
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

ax = fig.add_subplot(1, 2, 1)
ax.axis('equal')

for cluster, c in zip(np.unique(labels_dbscan), colors):
    class_member_mask = labels_dbscan == cluster

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c, marker='o',
             markeredgecolor='k', markersize=4, alpha=0.8)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c, marker='o',
             markeredgecolor='k', markersize=8, alpha=0.8)
ax.set_title('DBSCAN')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-2.6, .5, 'Runtime: %.2fs' % t_dbscan)
plt.text(-2.6, .3, 'ARI: %.4f' % rand_dbscan)
plt.text(-2.6, .1, 'AMI: %.4f' % mi_dbscan)

# DBSCAN with subsampling
core_samples_mask = np.zeros_like(dbscan_sub.labels_, dtype=bool)
core_samples_mask[dbscan_sub.core_sample_indices_] = True

ax = fig.add_subplot(1, 2, 2)
ax.axis('equal')

for cluster, c in zip(np.unique(labels_sub), colors):
    class_member_mask = labels_sub == cluster

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c, marker='o',
             markeredgecolor='k', markersize=4, alpha=0.8)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c, marker='o',
             markeredgecolor='k', markersize=8, alpha=0.8)
ax.set_title('DBSCAN with subsampling')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-2.6, .5, 'Runtime: %.2fs' % t_sub)
plt.text(-2.6, .3, 'ARI: %.4f' % rand_sub)
plt.text(-2.6, .1, 'AMI: %.4f' % mi_sub)

plt.show()
