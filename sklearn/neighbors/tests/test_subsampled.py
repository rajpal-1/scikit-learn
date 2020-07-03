"""
Testing for the subsampled module.
"""

import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal, assert_almost_equal

from sklearn.neighbors import SubsampledNeighborsTransformer
from sklearn import datasets
from sklearn.utils._testing import assert_array_almost_equal

# Toy samples
X = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [6, 7, 8]]
X_csr = csr_matrix(X) 
X2 = [[6], [5], [4], [3]]
X2_csr = csr_matrix(X2)

# Load the iris dataset and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

def test_sample_toy():
    # Check sampling on a toy dataset, including sparse versions.

    # Fit and transform with non-sparse
    n = SubsampledNeighborsTransformer(1.0, random_state=0)
    expected_result = csr_matrix(([1.732051, 1.732051, 1.732051, 1.732051], ([0, 1, 1, 2], 
        [1, 0, 2, 1])), shape=(4, 4))
    assert_array_almost_equal(n.fit_transform(X).toarray(), expected_result.toarray())
    expected_result = csr_matrix(([1., 1., 1., 1.], ([0, 1, 1, 2], [1, 0, 2, 1])), shape=(4, 4))
    assert_array_equal(n.fit_transform(X2).toarray(), expected_result.toarray())

    # Fit and transform with sparse
    n = SubsampledNeighborsTransformer(0.1, symmetric=False, random_state=1)
    expected_result = csr_matrix(([6.92820323], ([1], [3])), shape=(4, 4))
    assert_array_almost_equal(n.fit(X_csr).transform(X).toarray(), expected_result.toarray())
    expected_result = csr_matrix(([2.], ([1], [3])), shape=(4, 4))
    assert_array_equal(n.fit_transform(X2_csr).toarray(), expected_result.toarray())

    # Fit with sparse, test with non-sparse
    n = SubsampledNeighborsTransformer(0.9, random_state=2)
    expected_result = csr_matrix(([1.732051, 8.6602540, 1.732051, 1.732051, 6.928203, 1.732051, 
      5.196152, 8.660254, 6.928203, 5.196152], ([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], [1, 3, 0, 2, 3, 1, 3, 0, 1, 2])), 
      shape=(4, 4))
    assert_array_almost_equal(n.fit(X_csr).transform(X).toarray(), expected_result.toarray())
    expected_result = csr_matrix(([1., 3., 1., 1., 2., 1., 1., 3., 2., 1.], ([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], 
        [1, 3, 0, 2, 3, 1, 3, 0, 1, 2])), shape=(4, 4))
    assert_array_equal(n.fit(X2_csr).transform(X2).toarray(), expected_result.toarray())

    # Fit with non-sparse, test with sparse
    n = SubsampledNeighborsTransformer(0.2, symmetric=False, random_state=3)
    expected_result = csr_matrix(([1.732051, 5.196152], ([1, 2], [0, 3])), shape=(4, 4))
    assert_array_almost_equal(n.fit(X).transform(X_csr).toarray(), expected_result.toarray())
    expected_result = csr_matrix(([1., 1.], ([1, 2], [0, 3])), shape=(4, 4))
    assert_array_equal(n.fit(X2).transform(X2_csr).toarray(), expected_result.toarray())

    # Fit and transform with non-CSR sparse matrices
    n = SubsampledNeighborsTransformer(0.8, random_state=4)
    expected_result = csr_matrix(([1.732051, 3.464102, 1.732051, 6.928203, 3.464102, 5.196152, 6.928203, 
        5.196152], ([0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2])), shape=(4, 4))
    assert_array_almost_equal(n.fit(X_csr.tocoo()).transform(X_csr.tolil()).toarray(), expected_result.toarray())
    expected_result = csr_matrix(([1., 2., 1., 2., 2., 1., 2., 1.], ([0, 0, 1, 1, 2, 2, 3, 3], 
        [1, 2, 0, 3, 0, 3, 1, 2])), shape=(4, 4))
    assert_array_equal(n.fit(X2_csr.todok()).transform(X2_csr.tocsc()).toarray(), expected_result.toarray())


def test_iris():
    # Check consistency on dataset iris.

    # Euclidean distance
    n = SubsampledNeighborsTransformer(0.3, metric='euclidean', random_state=42)
    assert_almost_equal(np.mean(n.fit(iris.data).transform(iris.data)), 0.64403748)

    # Cosine distance
    n = SubsampledNeighborsTransformer(0.7, metric='cosine', random_state=42)
    assert_almost_equal(np.mean(n.fit(iris.data).transform(iris.data)), 0.02264031)

    # Manhattan distance
    n = SubsampledNeighborsTransformer(0.4, metric='manhattan', random_state=42)
    assert_almost_equal(np.mean(n.fit(iris.data).transform(iris.data)), 1.35717333)