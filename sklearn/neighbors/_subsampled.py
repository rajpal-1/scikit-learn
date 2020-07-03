"""Subsampled neighbors transformer"""

# Author: Jennifer Jang <j.jang42@gmail.com>
#                 Heinrich Jiang <heinrichj@google.com>
#
# License: BSD 3 clause

import numpy as np

from ..metrics.pairwise import paired_distances
from ._base import UnsupervisedMixin
from ..base import TransformerMixin, BaseEstimator
from ..utils import check_random_state
from ..utils.validation import check_is_fitted, _deprecate_positional_args
from ..utils.validation import check_array


class SubsampledNeighborsTransformer(TransformerMixin, UnsupervisedMixin, 
    BaseEstimator):
    """Compute the subsampled graph of neighbors for points in X.

    Parameters
    ----------

    s : float
        Sampling probability.

    metric : string or callable, default='euclidean'
        Input to paired_distances function. Can be string specified 
        in PAIRED_DISTANCES, including "euclidean", "manhattan", or 
        "cosine." Alternatively, can be a callable function, which should 
        take two arrays from X as input and return a value indicating 
        the distance between them.

    symmetric : boolean, default=True
        Sample an undirected graph, where an edge from vertices i and j 
        implies an edge from j to i, or a directed graph where no such 
        symmetry is enforced.

    random_state : int, RandomState instance, default=None
        Seeds the random sampling of lists of vertices. Use an int to 
        make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    References
    ----------
    - Faster DBSCAN via subsampled similarity queries, 2020
        Heinrich Jiang, Jennifer Jang, Jakub Łącki
        https://arxiv.org/abs/2006.06743

    Notes
    -----
    Each edge in the fully connected graph of X is sampled with probability s
    with replacement. We sample two arrays of n_samples * n_samples * s vertices 
    from X with replacement. Since (i, j) is equivalent to (j, i), we discard any 
    pairs where j >= i. We ensure symmetry by adding the neighborhood matrix to its 
    transpose. 
    """

    @_deprecate_positional_args
    def __init__(self, s, *, metric='euclidean', symmetric=True, random_state=None):
        self.s = s
        self.metric = metric
        self.symmetric = symmetric
        self.random_state = random_state


    def _fit(self, X):
        if self.s <= 0 or self.s > 1:
            raise ValueError("Sampling rate needs to be in (0, 1]: %s" % self.s)

        self.s_ = self.s

        return self


    def transform(self, X):
        """Transform data into a subsampled graph of neighbors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
                Sample data.

        Returns
        -------
        neighborhood : sparse matrix of shape (n_samples, n_samples)
                Non-zero entries in neighborhood[i, j] indicate an edge 
                between X[i] and X[j] with value equal to weight of edge.
                The matrix is of CSR format.
        """

        check_is_fitted(self)

        return self.subsampled_neighbors(X, self.s_, self.metric, 
            self.symmetric, self.random_state)


    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
                Sample data.
        y : ignored

        Returns
        -------
        neighborhood : sparse matrix of shape (n_samples, n_samples)
                Non-zero entries in neighborhood[i, j] indicate an edge 
                between X[i] and X[j] with value equal to weight of edge.
                The matrix is of CSR format.
        """
        
        return self.fit(X).transform(X)


    def subsampled_neighbors(self, X, s, metric='euclidean', symmetric=True,
        random_state=None):
        """Compute the subsampled graph of neighbors for points in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
                Sample data.

        s : float
            Sampling probability.

        metric : string or callable, default='euclidean'
            Input to paired_distances function. Can be string specified 
            in PAIRED_DISTANCES, including "euclidean", "manhattan", or 
            "cosine." Alternatively, can be a callable function, which should 
            take two arrays from X as input and return a value indicating 
            the distance between them.

        symmetric : boolean, default=True
            Sample an undirected graph, where an edge from vertices i and j 
            implies an edge from j to i, or a directed graph where no such 
            symmetry is enforced.

        random_state : int, RandomState instance, default=None
            Seeds the random sampling of lists of vertices. Use an int to 
            make the randomness deterministic.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        neighborhood : sparse matrix of shape (n_samples, n_samples)
                Non-zero entries in neighborhood[i, j] indicate an edge 
                between X[i] and X[j] with value equal to weight of edge.
                The matrix is of CSR format.
        """

        from scipy.sparse import csr_matrix, triu

        X = check_array(X, accept_sparse='csr')
        random_state = check_random_state(random_state)

        n_samples = X.shape[0]

        # Sample the neighbors with replacement
        x = random_state.choice(np.arange(n_samples), size=int(n_samples * n_samples * s), 
            replace=True)
        y = random_state.choice(np.arange(n_samples), size=int(n_samples * n_samples * s), 
            replace=True)

        # Remove duplicates
        neighbors = np.unique(np.column_stack((x, y)), axis=0)

        i = neighbors[:, 0]
        j = neighbors[:, 1]

        # Compute the edge weights for the remaining edges
        if len(neighbors) > 0:
            distances = paired_distances(X[i], X[j], metric=metric)
        else:
            distances = []

        # Create the distance matrix in CSR format 
        neighborhood = csr_matrix((distances, (i, j)), shape=(n_samples, n_samples), 
            dtype=np.float)
        
        if symmetric:
            # Upper triangularize the matrix
            neighborhood = triu(neighborhood)
            # Make the matrix symmetric
            neighborhood += neighborhood.transpose()

        return neighborhood