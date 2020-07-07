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
X = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [6, 7, 8]]
X_csr = csr_matrix(X)
X2 = [[6], [5], [4], [3]]
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
    n = SubsampledNeighborsTransformer(1., eps=5., random_state=0)
    expected_result = csr_matrix(([1.732051, 1.732051, 1.732051, 1.732051],
                                  ([0, 1, 1, 2], [1, 0, 2, 1])), shape=(4, 4))
    assert_array_almost_equal(n.fit_transform(X).toarray(),
                              expected_result.toarray())

    n = SubsampledNeighborsTransformer(1., eps=5., random_state=0)
    expected_result = csr_matrix(([1., 1., 1., 1., 3., 2., 1.],
                                  ([0, 1, 1, 2, 3, 3, 3],
                                   [1, 0, 2, 1, 0, 1, 2])), shape=(4, 4))
    assert_array_equal(n.fit_transform(X2).toarray(),
                       expected_result.toarray())


def test_sample_toy_fit_sparse_transform_sparse():
    # Fit and transform with sparse
    n = SubsampledNeighborsTransformer(s=0.2, random_state=1)
    expected_result = csr_matrix(([1.732051, 1.732051], ([0, 1], [1, 0])),
                                 shape=(4, 4))
    assert_array_almost_equal(n.fit(X_csr).transform(X_csr).toarray(),
                              expected_result.toarray())

    n = SubsampledNeighborsTransformer(s=0.2, random_state=1)
    expected_result = csr_matrix(([1., 1.], ([0, 1], [1, 0])), shape=(4, 4))
    assert_array_equal(n.fit(X2_csr).transform(X2_csr).toarray(),
                       expected_result.toarray())


def test_sample_toy_fit_sparse_transform_nonsparse():
    # Fit with sparse, test with non-sparse
    n = SubsampledNeighborsTransformer(0.9, random_state=2)
    expected_result = csr_matrix(([1.732051, 8.660254, 1.732051, 6.928203,
                                   3.464102, 5.196152, 8.660254, 5.196152],
                                  ([0, 0, 1, 1, 2, 2, 3, 3],
                                   [1, 3, 2, 3, 0, 3, 0, 2])),
                                 shape=(4, 4))
    assert_array_almost_equal(n.fit(X_csr).transform(X).toarray(),
                              expected_result.toarray())
    expected_result = csr_matrix(([1.0, 3.0, 1.0, 2., 2., 1., 3., 1.],
                                  ([0, 0, 1, 1, 2, 2, 3, 3],
                                   [1, 3, 2, 3, 0, 3, 0, 2])),
                                 shape=(4, 4))
    assert_array_equal(n.fit(X2_csr).transform(X2).toarray(),
                       expected_result.toarray())


def test_sample_toy_fit_nonsparse_transform_sparse():
    # Fit with non-sparse, test with sparse
    n = SubsampledNeighborsTransformer(0.3, random_state=3)
    expected_result = csr_matrix(([1.732051, 3.464102, 6.928203],
                                  ([1, 2, 3], [0, 0, 1])), shape=(4, 4))
    assert_array_almost_equal(n.fit(X).transform(X_csr).toarray(),
                              expected_result.toarray())
    expected_result = csr_matrix(([1., 2., 2.], ([1, 2, 3], [0, 0, 1])),
                                 shape=(4, 4))
    assert_array_equal(n.fit(X2).transform(X2_csr).toarray(),
                       expected_result.toarray())


def test_sample_toy_noncsr():
    # Fit and transform with non-CSR sparse matrices
    n = SubsampledNeighborsTransformer(0.8, random_state=4)
    expected_result = csr_matrix(([1.732051, 3.464102, 1.732051, 6.928203,
                                   1.732051, 5.196152, 8.660254, 5.196152],
                                  ([0, 0, 1, 1, 2, 2, 3, 3],
                                   [1, 2, 0, 3, 1, 3, 0, 2])),
                                 shape=(4, 4))
    assert_array_almost_equal(n.fit(X_csr.tocoo()).transform(
        X_csr.tolil()).toarray(), expected_result.toarray())
    expected_result = csr_matrix(([1., 2., 1., 2., 1., 1., 3., 1.],
                                  ([0, 0, 1, 1, 2, 2, 3, 3],
                                   [1, 2, 0, 3, 1, 3, 0, 2])),
                                 shape=(4, 4))
    assert_array_equal(n.fit(X2_csr.todok()).transform(
                       X2_csr.tocsc()).toarray(), expected_result.toarray())


def test_sample_toy_different():
    # Fit and transform with different matrices
    n = SubsampledNeighborsTransformer(0.8, random_state=4)
    expected_result = csr_matrix(([5.22781, 8.584288, 7.29863, 9.971961],
                                  ([0, 0, 1, 1], [0, 1, 0, 2])),
                                 shape=(2, 3))
    assert_array_almost_equal(n.fit(X_fit).transform(X_transform).toarray(),
                              expected_result.toarray())


def test_sample_toy_no_edges():
    # Sampling rate too low
    n = SubsampledNeighborsTransformer(0.01, random_state=5)
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
                        0.8225096)

    n = SubsampledNeighborsTransformer(0.4, eps=2.0, metric='euclidean',
                                       random_state=42)
    assert_almost_equal(np.mean(n.fit(iris.data).transform(iris.data)),
                        0.1517863)


def test_iris_cosine():
    n = SubsampledNeighborsTransformer(0.6, eps=0.1, metric='cosine',
                                       random_state=42)
    assert_almost_equal(np.mean(n.fit(iris.data).transform(iris.data)),
                        0.0097946)


def test_iris_callable():
    # Callable lambda function
    fn = lambda a, b : np.mean(np.maximum(a, b))
    n = SubsampledNeighborsTransformer(0.5, eps=5.0, metric=fn,
                                       random_state=42)
    assert_almost_equal(np.mean(n.fit_transform(iris.data)), 1.4916244)


def test_iris_small_s():
    # Small s
    n = SubsampledNeighborsTransformer(0.0001, random_state=42)
    expected_result = csr_matrix(([1.516575, 3.780212],
                                 ([92, 102], [106, 14])),
                                 shape=(n_iris, n_iris))
    assert_array_almost_equal(n.fit_transform(iris.data).toarray(),
                              expected_result.toarray())

    n = SubsampledNeighborsTransformer(0.0002, eps=6.0, random_state=42)
    expected_result = csr_matrix(([3.780212, 2.651415, 5.480876, 1.174734],
                                 ([14, 92, 102, 106], [102, 20, 71, 121])),
                                 shape=(n_iris, n_iris))
    assert_array_almost_equal(n.fit_transform(iris.data).toarray(),
                              expected_result.toarray())


def test_iris_large_s():
    # Large s
    n = SubsampledNeighborsTransformer(2.0, random_state=42)
    assert_almost_equal(np.mean(n.fit_transform(iris.data)), 2.18850706)

    # Large s
    n = SubsampledNeighborsTransformer(2.0, eps=2.5, random_state=42)
    assert_almost_equal(np.mean(n.fit_transform(iris.data)), 0.51904614)


def test_iris_small_eps():
    # Small eps
    n = SubsampledNeighborsTransformer(0.5, eps=0.1, random_state=42)
    expected_result = csr_matrix(([0.1, 0.1, 0.1, 0.1, 0.1],
                                 ([41, 50, 81, 91, 127],
                                  [127, 133, 91, 81, 41])),
                                 shape=(n_iris, n_iris))
    assert_array_almost_equal(n.fit_transform(iris.data).toarray(),
                              expected_result.toarray())


def test_iris_large_eps():
    # Large eps
    n = SubsampledNeighborsTransformer(0.8, eps=100., random_state=42)
    assert_almost_equal(np.mean(n.fit_transform(iris.data)), 1.40401357)


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
