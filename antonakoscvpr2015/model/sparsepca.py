from __future__ import division
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

from menpo.visualize import print_dynamic, progress_bar_str
from menpo.model.pca import PCAModel


class SparsePCAModel(PCAModel):
    def __init__(self, adjacency_array, samples, patch_len, centre=True,
                 bias=False, level_str='', verbose=False):
        # build data matrix
        n_samples = len(samples)
        n_features = samples[0].n_parameters
        data = np.zeros((n_samples, n_features))
        for i, sample in enumerate(samples):
            data[i] = sample.as_vector()

        # compute pca
        eigenvectors, eigenvalues, mean_vector = \
            principal_component_decomposition_sparse(data, adjacency_array,
                                                     patch_len, whiten=False,
                                                     centre=centre, bias=bias,
                                                     level_str=level_str,
                                                     verbose=verbose)

        super(PCAModel, self).__init__(eigenvectors, mean_vector, samples[0])
        self.centred = centre
        self.biased = bias
        self._eigenvalues = eigenvalues
        self._n_components = self.n_components
        self._trimmed_eigenvalues = None

    def __str__(self):
        str_out = 'Sparse PCA Model \n'
        str_out = str_out + \
            ' - centred:             {}\n' \
            ' - biased:               {}\n' \
            ' - # features:           {}\n' \
            ' - # active components:  {}\n'.format(
            self.centred, self.biased, self.n_features,
            self.n_active_components)
        str_out = str_out + \
            ' - kept variance:        {:.2}  {:.1%}\n' \
            ' - noise variance:       {:.2}  {:.1%}\n'.format(
            self.variance(), self.variance_ratio(),
            self.noise_variance(), self.noise_variance_ratio())
        str_out = str_out + \
            ' - total # components:   {}\n' \
            ' - components shape:     {}\n'.format(
            self.n_components, self.components.shape)
        return str_out


def eigenvalue_decomposition(S, eps=10**-10):
    r"""

    Parameters
    ----------
    S : (N, N)  ndarray
        Covariance/Scatter matrix

    Returns
    -------
    pos_eigenvectors: (N, p) ndarray
    pos_eigenvalues: (p,) ndarray
    """
    # compute eigenvalue decomposition
    eigenvalues, eigenvectors = eigsh(S)
    # sort eigenvalues from largest to smallest
    index = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    # set tolerance limit
    limit = np.max(np.abs(eigenvalues)) * eps

    # select positive eigenvalues
    pos_index = eigenvalues > 0.0
    pos_eigenvalues = eigenvalues[pos_index]
    pos_eigenvectors = eigenvectors[:, pos_index]
    # check they are within the expected tolerance
    index = pos_eigenvalues > limit
    pos_eigenvalues = pos_eigenvalues[index]
    pos_eigenvectors = pos_eigenvectors[:, index]

    return pos_eigenvectors, pos_eigenvalues


def principal_component_decomposition_sparse(X, adjacency_array, patch_len,
                                             whiten=False, centre=True,
                                             bias=False, inplace=False,
                                             level_str='', verbose=False):
    r"""
    Apply PCA on the data matrix X. In the case where the data matrix is very
    large, it is advisable to set `inplace=True`. However, note this this
    destructively edits the data matrix by subtracting the mean inplace.

    Parameters
    ----------
    x : (n_samples, n_features) ndarray
        Training data
    whiten : bool, optional
        Normalise the eigenvectors to have unit magnitude

        Default: `False`
    centre : bool, optional
        Whether to centre the data matrix. If `False`, zero will be subtracted.

        Default: `True`
    bias : bool, optional
        Whether to use a biased estimate of the number of samples. If `False`,
        subtracts `1` from the number of samples.

        Default: `False`
    inplace : bool, optional
        Whether to do the mean subtracting inplace or not. This is crucial if
        the data matrix is greater than half the available memory size.

        Default: `False`

    Returns
    -------
    eigenvectors : (n_components, n_features) ndarray
        The eigenvectors of the data matrix
    eigenvalues : (n_components,) ndarray
        The positive eigenvalues from the data matrix
    mean_vector : (n_components,) ndarray
        The mean that was subtracted from the dataset
    """
    n_samples, n_features = X.shape

    if bias:
        N = n_samples
    else:
        N = n_samples - 1.0

    if centre:
        # centre data
        mean_vector = np.mean(X, axis=0)
    else:
        mean_vector = np.zeros(n_features)

    # This is required if the data matrix is very large!
    if inplace:
        X -= mean_vector
    else:
        X = X - mean_vector

    # compute covariance matrix
    # S:  n_features  x  n_features
    S = compute_sparse_covariance(X.T, adjacency_array, patch_len, level_str,
                                  verbose)
    # S should be perfectly symmetrical, but numerical error can creep
    # in. Enforce symmetry here to avoid creating complex
    # eigenvectors from eigendecomposition
    S = (S + S.T) / 2.0

    # perform eigenvalue decomposition
    # eigenvectors:  n_features x  n_features
    # eigenvalues:   n_features
    eigenvectors, eigenvalues = eigenvalue_decomposition(S)

    if whiten:
        # whiten eigenvectors
        eigenvectors *= np.sqrt(1.0 / eigenvalues)

    # transpose eigenvectors
    # eigenvectors:  n_samples  x  n_features
    eigenvectors = eigenvectors.T

    return eigenvectors, eigenvalues, mean_vector


def compute_sparse_covariance(X, adjacency_array, patch_len, level_str,
                              verbose):
    n_features, n_samples = X.shape
    n_edges = adjacency_array.shape[0]

    # initialize block sparse covariance matrix
    all_cov = lil_matrix((n_features, n_features))

    # compute covariance matrix for each edge
    for e in range(n_edges):
        # print progress
        if verbose:
            print_dynamic('{}Distribution per edge - {}'.format(
                          level_str,
                          progress_bar_str(float(e + 1) / n_edges,
                                           show_bar=False)))

        # edge vertices
        v1 = np.min(adjacency_array[e, :])
        v2 = np.max(adjacency_array[e, :])

        # find indices in target covariance matrix
        v1_from = v1 * patch_len
        v1_to = (v1 + 1) * patch_len
        v2_from = v2 * patch_len
        v2_to = (v2 + 1) * patch_len

        # extract data
        edge_data = np.concatenate((X[v1_from:v1_to, :], X[v2_from:v2_to, :]))

        # compute covariance inverse
        icov = np.cov(edge_data)

        # v1, v2
        all_cov[v1_from:v1_to, v2_from:v2_to] += icov[:patch_len, patch_len::]

        # v2, v1
        all_cov[v2_from:v2_to, v1_from:v1_to] += icov[patch_len::, :patch_len]

        # v1, v1
        all_cov[v1_from:v1_to, v1_from:v1_to] += icov[:patch_len, :patch_len]

        # v2, v2
        all_cov[v2_from:v2_to, v2_from:v2_to] += icov[patch_len::, patch_len::]

    return all_cov.tocsr()
