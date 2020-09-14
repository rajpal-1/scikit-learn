"""Subsampled neighbors transformer"""

# Author: Jennifer Jang <j.jang42@gmail.com>
#         Heinrich Jiang <heinrichj@google.com>
#
# License: BSD 3 clause

cimport cython
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.pair cimport pair
from libc.stdlib cimport rand
from libcpp.algorithm cimport sort as stdsort

cimport numpy as np
import numpy as np

from ..metrics.pairwise import paired_distances, check_pairwise_arrays, \
    PAIRED_DISTANCES, paired_euclidean_distances
from ._base import UnsupervisedMixin
from ..base import TransformerMixin, BaseEstimator
from ..utils import check_random_state
from ..utils.validation import check_is_fitted
from ..utils.validation import check_array

np.import_array()

import time

# def subsample2(double eps,
#               double s,
#               np.int32_t n,
#               np.int32_t d,
#               np.ndarray[double, ndim=1, mode='c'] X):

#     cdef np.int32_t i, j, k, neighbor, cnt = 0
#     cdef np.npy_float distance

#     cdef vector[vector[pair[np.npy_float, np.int32_t]]] neighbors
#     cdef vector[np.npy_float] data
#     cdef vector[np.int32_t] indices, indptr

#     for i in range(n):
#         neighbors.push_back(vector[pair[np.npy_float, np.int32_t]]())

#         # Explicity set each point as its own neighbor
#         neighbors[i].push_back((0., i))

#     for i in range(n - 1):
#         # To ensure neighborhood graph is symmetric, we only sample points that come after
#         for j in range(int(s * (n - i)) - 1):

#             neighbor = rand() % (n - i - 1) + i + 1;
            
#             distance = 0
#             for k in range(d):
#                 distance = (X[i * d + k] - X[neighbor * d + k])**2
#             # distance = paired_euclidean_distances(X[i:i+1], X[neighbor:neighbor+1])[0]

#             distance **= 0.5
#             if distance <= eps:
#                 # Add edge between both vertices
#                 neighbors[i].push_back((distance, neighbor))
#                 neighbors[neighbor].push_back((distance, i))

#     for i in range(n):
#         stdsort(neighbors[i].begin(), neighbors[i].end())
#         indptr.push_back(cnt)

#         for dist_neighb in neighbors[i]:
#             data.push_back(dist_neighb.first)
#             indices.push_back(dist_neighb.second)
#             cnt += 1

#     indptr.push_back(cnt)

#     return data, indices, indptr

def subsample(double s,
              np.int32_t n,
              np.int32_t d,
              np.ndarray[np.int32_t, ndim=1, mode='c'] rows,
              np.ndarray[np.int32_t, ndim=1, mode='c'] cols
             ):

    cdef np.int32_t i, _, cnt = 0

    for i in range(n):
        
        # Explicity set each point as its own neighbor
        rows[cnt] = i
        cols[cnt] = i
        cnt += 1
        
        # Sample neighbors
        for _ in range(int(s * n)):
            
            rows[cnt] = i
            cols[cnt] = rand() % n
            cnt += 1

def sort_by_data(int n,
                 int m,
                 np.ndarray[double, ndim=1, mode='c'] distances,
                 np.ndarray[np.int32_t, ndim=1, mode='c'] rows,
                 np.ndarray[np.int32_t, ndim=1, mode='c'] cols,
                 np.ndarray[np.int32_t, ndim=1, mode='c'] indptr
                ):

    cdef np.int32_t i, j, start, end, row = 0
    cdef vector[pair[np.npy_float, np.int32_t]] dist_column

    for i in range(m):#4

        # Fill in indptr array
        for j in range(rows[i] - row):
            indptr[row + 1 + j] = i
        row = rows[i] #2

        # Create vector of pairs for sorting
        dist_column.push_back((distances[i], cols[i]))

    # Fill in indices for trailing rows that don't have distances
    for row in range(rows[m - 1], n):
        indptr[row + 1] = m

    for i in range(n):
        start = indptr[i]
        end = indptr[i + 1]

        # Sort each row by distance
        stdsort(dist_column.begin() + start, dist_column.begin() + end)

        # Recreate output arrays
        for j in range(start, end):
            distances[j] = dist_column[j].first
            cols[j] = dist_column[j].second


class SubsampledNeighborsTransformer(TransformerMixin, UnsupervisedMixin,
                                     BaseEstimator):
    """Compute subsampled sparse distance matrix of neighboring points in X.

    Parameters
    ----------

    s : float
        Sampling probability.

    eps : float, default=None
        Neighborhood radius. Pairs of points which are at most eps apart are
        considered neighbors. If None, radius is assumed to be infinity.

    metric : string or callable, default='euclidean'
        Input to paired_distances function. Can be string specified
        in PAIRED_DISTANCES, including "euclidean", "manhattan", or
        "cosine." Alternatively, can be a callable function, which should
        take two arrays from X as input and return a value indicating
        the distance between them.

    random_state : int, RandomState instance, default=None
        Seeds the random sampling of lists of vertices. Use an int to
        make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    fit_X_ : array-like of shape (n_train, n_features)
        Training set

    n_train_ : int
        Number of training samples

    random_state_ : numpy.RandomState
        Pseudo random number generator object used during initialization.

    References
    ----------
    - Faster DBSCAN via subsampled similarity queries, 2020
        Heinrich Jiang, Jennifer Jang, Jakub Łącki
        https://arxiv.org/abs/2006.06743

    Notes
    -----
    Each pair of points is sampled uniformly with probability s.
    """

    def __init__(self, s=0.1, eps=None, metric='euclidean',
                 random_state=None):
        self.s = s
        self.eps = eps
        self.metric = metric
        self.random_state = random_state
        self._check_parameters()

    def _check_parameters(self):
        if self.s < 0:
            raise ValueError("Sampling rate needs to be non-negative: %s" %
                             self.s)

        if self.eps is not None and self.eps <= 0:
            raise ValueError("Epsilon needs to be positive: %s" % self.eps)

        if self.metric not in PAIRED_DISTANCES and not callable(self.metric):
            raise ValueError('Unknown distance %s' % self.metric)

        return self

    def _fit(self, X):

        self.fit_X_ = check_array(X, accept_sparse='csr')
        self.n_train_ = self.fit_X_.shape[0]
        self.random_state_ = check_random_state(self.random_state)

        return self

    def transform(self, X):
        """Transform data into a subsampled graph of neighbors.

        Parameters
        ----------
        X : array-like of shape (n, n_features)
            Sample data.

        Returns
        -------
        neighborhood : sparse matrix of shape (n, n)
            Sparse matrix where the i-jth value is equal to the distance
            between X[i] and fit_X[j] for randomly sampled pairs of neighbors.
            The matrix is of CSR format.
        """

        check_is_fitted(self)

        return self.subsampled_neighbors(X, self.s, self.eps, self.metric,
                                         self.random_state_)

    def subsampled_neighbors(self, X, s, eps=None, metric='euclidean',
                             random_state=None):
        """Compute the subsampled sparse distance matrix of the neighboring
        points of X in fit_X.

        Parameters
        ----------
        X : array-like of shape (n, n_features)
            Sample data.

        s : float
            Sampling probability.

        eps : float, default=None
            Neighborhood radius. Pairs of points which are at most eps apart
            are considered neighbors. If not given, radius is assumed to be
            infinity.

        metric : string or callable, default='euclidean'
            Input to paired_distances function. Can be string specified
            in PAIRED_DISTANCES, including "euclidean", "manhattan", or
            "cosine." Alternatively, can be a callable function, which should
            take two arrays from X as input and return a value indicating
            the distance between them.

        random_state : int, RandomState instance, default=None
            Seeds the random sampling of lists of vertices. Use an int to
            make the randomness deterministic.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        neighborhood : sparse matrix of shape (n, n)
            Sparse matrix where the i-jth value is equal to the distance
            between X[i] and fit_X[j] for randomly sampled pairs of neighbors.
            The matrix is of CSR format.
        """

        from scipy.sparse import csr_matrix

        X, fit_X = check_pairwise_arrays(X, self.fit_X_, accept_sparse='csr')
        
        n, d = X.shape
        n_neighbors = 2 * int(s * n / 2) * n + n
        
        # No edges sampled
        if n_neighbors < 1:
            return csr_matrix((n, self.n_train_), dtype=np.float)
        
        ## FROM HERE
        # # Sample the edges with replacement
        # t0 = time.time()
        # x = np.repeat(np.arange(n), n_edges)
        # y = np.array(np.random.rand(n * n_edges) * n, dtype=np.int32)
        # t1 = time.time()
        # print('s1', t1-t0)
        
        # TO HERE

        t0 = time.time()
        rows = np.full(n_neighbors, -1, dtype=np.int32)
        cols = np.full(n_neighbors, -1, dtype=np.int32)
        subsample(s, n, d, rows, cols)
        t1 = time.time()
        print('s1', t1-t0)

        # THIS IS VERY SLOW
        distances = paired_distances(X[rows], X[cols], metric=metric)
        t2 = time.time()
        print('s2', t2-t1)
        
        if eps is not None:
            eps_neighb = np.where(distances <= eps)[0]
            rows = rows[eps_neighb]
            cols = cols[eps_neighb]
            distances = distances[eps_neighb]

        t3 = time.time()
        print('s3', t3-t2)
        
        t4 = time.time()
        indptr = np.zeros(n + 1, dtype=np.int32)
        sort_by_data(n, rows.shape[0], distances, rows, cols, indptr)
        t5 = time.time()
        print('s4', t5-t4)

        neighborhood = csr_matrix((distances, cols, indptr),
                                  shape=(n, self.n_train_),
                                  dtype=np.float)
        print('s5', time.time()-t5)
        
        return neighborhood

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_methods_subset_invariance':
                'Fails for the transform method'
            }
        }
