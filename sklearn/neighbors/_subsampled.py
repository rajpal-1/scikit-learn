"""Subsampled neighbors transformer"""

# Author: Jennifer Jang <j.jang42@gmail.com>
#         Heinrich Jiang <heinrichj@google.com>
#
# License: BSD 3 clause

import numpy as np

from ..metrics.pairwise import paired_distances, PAIRED_DISTANCES
from ..utils.random import check_random_state
from ..base import TransformerMixin, BaseEstimator
from ..utils.validation import check_is_fitted


class SubsampledNeighborsTransformer(TransformerMixin, BaseEstimator):
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

    random_seed : int, default=None
        Seeds the random sampling of lists of vertices.

    Attributes
    ----------
    fit_X_ : array-like of shape (n_train, n_features)
        Training set

    n_train_ : int
        Number of training samples

    random_state_ : numpy.RandomState
        Pseudo random number generator object used for sampling.

    References
    ----------
    - Faster DBSCAN via subsampled similarity queries, 2020
        Heinrich Jiang, Jennifer Jang, Jakub Łącki
        https://arxiv.org/abs/2006.06743

    Notes
    -----
    Each pair of points is sampled uniformly with probability s.
    """

    def __init__(self, s=0.1, eps=None, metric='euclidean', random_state=None):
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

    def fit(self, X, Y=None):

        self.fit_X_ = self._validate_data(X, accept_sparse='csr')
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

        return self.subsampled_neighbors(X)

    def subsampled_neighbors(self, X):
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

    random_state : int or numpy.RandomState, default=None
        A pseudo random number generator object or a seed for it if int.
        See :term: `Glossary <random_state>`.

        Returns
        -------
        neighborhood : sparse matrix of shape (n, n)
            Sparse matrix where the i-jth value is equal to the distance
            between X[i] and fit_X[j] for randomly sampled pairs of neighbors.
            The matrix is of CSR format.
        """

        from scipy.sparse import csr_matrix

        X = self._validate_data(X, accept_sparse='csr')

        n, d = X.shape
        n_neighbors = int(self.s * self.n_train_)

        # Sample edges
        rows = np.repeat(np.arange(n), n_neighbors)
        cols = self.random_state_.randint(self.n_train_, size=n * n_neighbors)

        # No edges sampled
        if n_neighbors < 1:
            return csr_matrix((n, self.n_train_))

        distances = paired_distances(X[rows], self.fit_X_[cols],
                                     metric=self.metric)

        # Keep only neighbors within epsilon-neighborhood
        if self.eps is not None:
            eps_neighb = np.where(distances <= self.eps)
            rows = rows[eps_neighb]
            cols = cols[eps_neighb]
            distances = distances[eps_neighb]

        line_changes = np.bincount(rows + 1).cumsum()
        is_dupe = np.zeros(rows.shape[0], dtype=bool)

        # Loop over each row in our neighborhood graph
        for start, stop in zip(line_changes, line_changes[1:]):
            # Sort each row by distance
            dist_order = np.argsort(distances[start:stop], kind='mergesort')
            distances[start:stop] = distances[start:stop][dist_order]
            cols[start:stop] = cols[start:stop][dist_order]

            # Sort column indices and label duplicates
            # When consecutive elements in sorted array are equal,
            # it means there is a duplicate
            col_order = np.argsort(cols[start:stop], kind='mergesort')
            cols_tmp = cols[start:stop][col_order]
            is_dupe[start:stop][col_order[1:]] = cols_tmp[:-1] == cols_tmp[1:]

        # Dedupe
        rows = rows[~is_dupe]
        cols = cols[~is_dupe]
        distances = distances[~is_dupe]

        indptr = np.bincount(rows + 1, minlength=n + 1).cumsum()

        neighborhood = csr_matrix((distances, cols, indptr),
                                  shape=(n, self.n_train_))
        return neighborhood

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_methods_subset_invariance':
                'Fails for the transform method'
            }
        }
