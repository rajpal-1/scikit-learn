"""
Testing for the subsampled module.
"""

import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal, assert_almost_equal

from sklearn.neighbors import SubsampledNeighborsTransformer
from sklearn import datasets
from sklearn.utils._testing import assert_array_almost_equal, assert_raises

# Toy samples
X = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
X_csr = csr_matrix(X)
X2 = [[6.0], [5.0], [4.0], [3.0]]
X2_csr = csr_matrix(X2)
X_fit = [[5.6, 6.4, 3.0, 3.6], [7.8, 9.0, 4.7, 4.1], [1.5, 2.9, 0.4, 1.5]]
X_transform = [[3.0, 3.0, 1.5, 6.2], [7.7, 3.3, 0.4, 9.3]]

# Load the iris dataset and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
n_iris = iris.data.shape[0]


def test_sample_toy_fit_nonsparse_transform_nonsparse():
    # Test with non-sparse matrix
    n = SubsampledNeighborsTransformer(s=1., eps=5., random_state=0)
    expected_result = csr_matrix(([1.732051, 1.732051],
                                  ([0, 2], [1, 1])),
                                 shape=(4, 4))
    assert_array_almost_equal(n.fit_transform(X).toarray(),
                              expected_result.toarray())

    expected_result = csr_matrix(([1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 3.0],
                                  ([0, 0, 1, 2, 2, 3, 3],
                                   [1, 3, 3, 1, 3, 2, 0])),
                                 shape=(4, 4))
    assert_array_equal(n.fit_transform(X2).toarray(),
                       expected_result.toarray())


def test_sample_toy_fit_sparse_transform_sparse():
    # Fit and transform with sparse
    n = SubsampledNeighborsTransformer(s=0.3, random_state=1)
    expected_result = csr_matrix(([1.732051, 6.928203, 3.464102,
                                   8.660254],
                                  ([0, 1, 2, 3], [1, 3, 0, 0])),
                                 shape=(4, 4))
    assert_array_almost_equal(n.fit_transform(X_csr).toarray(),
                              expected_result.toarray())

    expected_result = csr_matrix(([1.0, 2.0, 2.0, 3.0],
                                  ([0, 1, 2, 3], [1, 3, 0, 0])),
                                 shape=(4, 4))
    assert_array_equal(n.fit_transform(X2_csr).toarray(),
                       expected_result.toarray())


def test_sample_toy_fit_sparse_transform_nonsparse():
    # Fit with sparse, test with non-sparse
    n = SubsampledNeighborsTransformer(0.9, random_state=2)
    expected_result = csr_matrix(([1.732051, 8.660254, 1.732051,
                                   1.732051, 6.928203, 3.464102,
                                   5.196152, 5.196152, 6.928203],
                                  ([0, 0, 1, 1, 1, 2, 2, 3, 3],
                                   [1, 3, 0, 2, 3, 0, 3, 2, 1])),
                                 shape=(4, 4))
    assert_array_almost_equal(n.fit(X_csr).transform(X).toarray(),
                              expected_result.toarray())

    expected_result = csr_matrix(([1.0, 3.0, 1.0, 1.0, 2.0, 1.0,
                                   2.0, 1.0, 2.0],
                                  ([0, 0, 1, 1, 1, 2, 2, 3, 3],
                                   [1, 3, 0, 2, 3, 3, 0, 2, 1])),
                                 shape=(4, 4))
    assert_array_equal(n.fit(X2_csr).transform(X2).toarray(),
                       expected_result.toarray())


def test_sample_toy_fit_nonsparse_transform_sparse():
    # Fit with non-sparse, test with sparse
    n = SubsampledNeighborsTransformer(0.3, random_state=3)
    expected_result = csr_matrix(([3.464102, 1.732051, 1.732051],
                                  ([0, 1, 2], [2, 0, 1])), shape=(4, 4))
    assert_array_almost_equal(n.fit(X).transform(X_csr).toarray(),
                              expected_result.toarray())

    expected_result = csr_matrix(([2.0, 1.0, 1.0], ([0, 1, 2], [2, 0, 1])),
                                 shape=(4, 4))
    assert_array_equal(n.fit(X2).transform(X2_csr).toarray(),
                       expected_result.toarray())


def test_sample_toy_noncsr():
    # Fit and transform with non-CSR sparse matrices
    n = SubsampledNeighborsTransformer(0.8, random_state=4)
    expected_result = csr_matrix(([3.464102, 8.660254, 1.732051, 3.464102,
                                   5.196152, 5.196152, 6.928203, 8.660254],
                                  ([0, 0, 1, 2, 2, 3, 3, 3],
                                   [2, 3, 0, 0, 3, 2, 1, 0])),
                                 shape=(4, 4))
    assert_array_almost_equal(n.fit(X_csr.tocoo()).transform(
        X_csr.tolil()).toarray(), expected_result.toarray())

    expected_result = csr_matrix(([2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0],
                                  ([0, 0, 1, 2, 2, 3, 3, 3],
                                   [2, 3, 0, 3, 0, 2, 1, 0])),
                                 shape=(4, 4))
    assert_array_equal(n.fit(X2_csr.todok()).transform(
                       X2_csr.tocsc()).toarray(), expected_result.toarray())


def test_sample_toy_different():
    # Fit and transform with different matrices
    n = SubsampledNeighborsTransformer(0.5, random_state=5)
    expected_result = csr_matrix(([5.055689, 8.833459],
                                  ([0, 1], [2, 1])),
                                 shape=(2, 3))
    assert_array_almost_equal(n.fit(X_fit).transform(X_transform).toarray(),
                              expected_result.toarray())


def test_sample_toy_no_edges():
    # Sampling rate too low
    n = SubsampledNeighborsTransformer(0.01, random_state=6)
    expected_result = csr_matrix(([], ([], [])), shape=(4, 4))
    assert_array_almost_equal(n.fit_transform(X).toarray(),
                              expected_result.toarray())
    expected_result = csr_matrix(([], ([], [])), shape=(4, 4))
    assert_array_equal(n.fit_transform(X2_csr).toarray(),
                       expected_result.toarray())

    # Epsilon too small
    n = SubsampledNeighborsTransformer(0.9, eps=0.01, random_state=6)
    expected_result = csr_matrix(([], ([], [])), shape=(4, 4))
    assert_array_almost_equal(n.fit_transform(X).toarray(),
                              expected_result.toarray())
    expected_result = csr_matrix(([], ([], [])), shape=(4, 4))
    assert_array_equal(n.fit_transform(X2_csr).toarray(),
                       expected_result.toarray())


def test_invalid_s():
    assert_raises(ValueError, SubsampledNeighborsTransformer, -1)
    assert_raises(ValueError, SubsampledNeighborsTransformer, -1, 0.5)


def test_invalid_metric():
    assert_raises(ValueError, SubsampledNeighborsTransformer, 0.7, 0.5,
                  'invalid')

    # Invalid metric and sampling rate too low
    assert_raises(ValueError, SubsampledNeighborsTransformer, 0.01, 0.5,
                  'invalid')


def test_invalid_eps():
    assert_raises(ValueError, SubsampledNeighborsTransformer, 0.5, -1)

    # Invalid eps and sampling rate too low
    assert_raises(ValueError, SubsampledNeighborsTransformer, 0.01, -1)


def test_invalid():
    # Invalid eps and invalid metric
    assert_raises(ValueError, SubsampledNeighborsTransformer, 0.5, -1,
                  'invalid')

    # Invalid eps and invalid metric and invalid s
    assert_raises(ValueError, SubsampledNeighborsTransformer, -0.5, -1,
                  'invalid')


def test_iris_euclidean():
    n = SubsampledNeighborsTransformer(0.4, metric='euclidean',
                                       random_state=42)
    assert_almost_equal(np.mean(n.fit(iris.data).transform(iris.data)),
                        0.836, decimal=2)

    n = SubsampledNeighborsTransformer(0.4, eps=2.0, metric='euclidean',
                                       random_state=42)
    assert_almost_equal(np.mean(n.fit(iris.data).transform(iris.data)),
                        0.147, decimal=2)


def test_iris_cosine():
    n = SubsampledNeighborsTransformer(0.6, eps=0.1, metric='cosine',
                                       random_state=42)
    assert_almost_equal(np.mean(n.fit(iris.data).transform(iris.data)),
                        0.009, decimal=2)


def test_iris_manhattan():
    # Manhattan distance
    n = SubsampledNeighborsTransformer(0.5, eps=10.0, metric='manhattan',
                                       random_state=42)
    assert_almost_equal(np.mean(n.fit_transform(iris.data)), 1.613, decimal=3)


def test_iris_callable():
    # Callable lambda function
    def fn(a, b):
        return np.mean(np.maximum(a, b))
    n = SubsampledNeighborsTransformer(0.5, eps=5.0, metric=fn,
                                       random_state=42)
    assert_almost_equal(np.mean(n.fit_transform(iris.data)), 1.504, decimal=3)


def test_iris_small_s():
    # Small s
    n = SubsampledNeighborsTransformer(0.001, random_state=42)
    expected_result = csr_matrix((n_iris, n_iris))
    assert_array_almost_equal(n.fit_transform(iris.data).toarray(),
                              expected_result.toarray(), decimal=2)

    n = SubsampledNeighborsTransformer(0.01, eps=0.5, random_state=42)
    expected_result = csr_matrix(([0.3, 0.3, 0.412311, 0.424264, 0.424264],
                                 ([18, 23, 27, 50, 128],
                                  [37, 88, 50, 80, 133])),
                                 shape=(n_iris, n_iris))
    assert_array_almost_equal(n.fit_transform(iris.data).toarray(),
                              expected_result.toarray(), decimal=2)


def test_iris_large_s():
    # Large s
    n = SubsampledNeighborsTransformer(2.0, random_state=42)
    assert_almost_equal(np.mean(n.fit_transform(iris.data)), 2.18, decimal=3)

    # Large s
    n = SubsampledNeighborsTransformer(2.0, eps=2.5, random_state=42)
    assert_almost_equal(np.mean(n.fit_transform(iris.data)), 0.52, decimal=3)


def test_iris_small_eps():
    # Small eps
    n = SubsampledNeighborsTransformer(0.5, eps=0.1, random_state=42)
    expected_result = csr_matrix(([0.1, 0.1, 0.1],
                                 ([50, 127, 133], [133, 41, 50])),
                                 shape=(n_iris, n_iris))
    assert_array_almost_equal(n.fit_transform(iris.data).toarray(),
                              expected_result.toarray())


def test_iris_large_eps():
    # Large eps
    n = SubsampledNeighborsTransformer(0.8, eps=100., random_state=42)
    assert_almost_equal(np.mean(n.fit_transform(iris.data)), 1.399, decimal=3)


def test_iris_no_edges():
    # Sampling rate too low
    n = SubsampledNeighborsTransformer(0.00001)
    expected_result = csr_matrix(([], ([], [])), shape=(n_iris, n_iris))
    assert_array_almost_equal(n.fit_transform(iris.data).toarray(),
                              expected_result.toarray())

    # Epsilon too small
    n = SubsampledNeighborsTransformer(0.5, eps=0.00001)
    expected_result = csr_matrix(([], ([], [])), shape=(n_iris, n_iris))
    assert_array_almost_equal(n.fit_transform(iris.data).toarray(),
                              expected_result.toarray())
