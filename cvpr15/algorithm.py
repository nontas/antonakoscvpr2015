from __future__ import division
import abc
import numpy as np

from menpofit.fittingresult import SemiParametricFittingResult


class APSInterface(object):
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def ds_dp(self):
        r"""
        Shape jacobian
        dS_dp = U --> (size: 2 x n x n_s)
        """
        return np.rollaxis(self.algorithm.transform.d_dp(None), -1)

    def ds_dp_vectorized(self):
        r"""
        Shape jacobian
        dS_dp = U --> (size: 2n x n_s)
        """
        n_params = self.ds_dp().shape[-1]
        return self.ds_dp().reshape([-1, n_params], order='F')

    def Sigma_s(self):
        r"""
        Deformation covariance
        Sigma_s --> (size: 2n x 2n)
        """
        return self.algorithm.deformation_model

    def H_s(self):
        r"""
        Deformation hessian
        H_s = U.T * Sigma_s * U --> (size: n_s x n_s)
        """
        tmp = self.ds_dp_vectorized().T.dot(self.Sigma_s())
        return tmp.dot(self.ds_dp_vectorized())

    def warp(self, image):
        r"""
        Warp function F: It extracts the patches around each shape point
        returns a list with the patches
        """
        return extract_patches(image, self.algorithm.transform.target,
                               self.algorithm.patch_shape,
                               normalize_patches=False)

    def vectorize(self, patches_list):
        r"""
        Vectorization function A: It vectorizes a given list of patches.
        Returns an ndarray of size (m*n) x 1
        """
        return vectorize_patches(patches_list)

    def gradient(self, patches):
        r"""
        Returns the gradients of the patches
        n_dims x n_channels x n_points x (w x h)
        """
        n_dims = patches[0].n_dims
        n_channels = patches[0].n_channels
        wh = np.prod(patches[0].shape)
        n_points = len(patches)
        gradients = np.empty((n_dims, n_channels, wh, n_points))
        for k, i in enumerate(patches):
            g = i.gradient().as_vector(keep_channels=True)
            gradients[..., k] = np.reshape(g, (-1, n_channels, n_dims)).T
        gradients = np.rollaxis(gradients, 3, 2)
        return gradients

    def steepest_descent_images(self, gradient, ds_dp):
        # reshape gradient
        # gradient: n_dims x n_channels x n_parts x (w x h)

        # compute steepest descent images
        # gradient: n_dims x n_channels x n_parts x (w x h)
        # ds_dp:    n_dims x            x n_parts x         x n_params
        # sdi:               n_channels x n_parts x (w x h) x n_params
        sdi = 0
        a = gradient[..., None] * ds_dp[..., None, :, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (n_channels x n_parts x w x h) x n_params
        return sdi.reshape((-1, sdi.shape[3]))

    def H_a(self, J, Sigma):
        r"""
        Appearance hessian
        H_a = J.T * Sigma_a * J --> (size: n_s x n_s)
        """
        tmp = J.T.dot(Sigma)
        return tmp.dot(J)

    def solve(self, h, j, e, sigma_a, h_s, p):
        r"""
        Parameters increment
        dp --> (size: n_s x 1)
        h: hessian (H = H_a + H_s)
        j: appearance jacobian (J_a)
        e: error image
        sigma_a: appearance covariance (S_a)
        h_s: shape hessian (H_s)
        p: current shape parameters
        """
        inv_h = np.linalg.inv(h)
        j_T_sigma_a = j.T.dot(sigma_a)
        b = j_T_sigma_a.dot(e) + h_s.dot(p)
        dp = - np.linalg.solve(inv_h, b)
        return dp

    def fitting_result(self, image, shape_parameters, gt_shape):
        return SemiParametricFittingResult(image, self.algorithm,
                                           parameters=shape_parameters,
                                           gt_shape=gt_shape)


class APSAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aps_interface, appearance_model, deformation_model,
                 patch_shape, transform, eps=10**-5):
        self.appearance_model = appearance_model
        self.deformation_model = deformation_model
        self.patch_shape = patch_shape
        self.template = appearance_model[0]
        self.transform = transform
        self.eps = eps

        # check if provided svd or covariance matrix
        self.use_svd = len(appearance_model) > 2

        # set interface
        self.interface = aps_interface(self)

        # precompute
        self._H_s = self.interface.H_s()
        self._ds_dp = self.interface.ds_dp()

    @abc.abstractmethod
    def _precompute(self):
        pass


class Forward(APSAlgorithm):

    def __init__(self, aps_interface, appearance_model, deformation_model,
                 patch_shape, transform, eps=10**-5):
        # call super constructor
        super(Forward, self).__init__(aps_interface, appearance_model,
                                      deformation_model, patch_shape, transform,
                                      eps)

        # pre-compute
        self._precompute()

    def _precompute(self):
        pass

    def run(self, image, initial_shape, gt_shape=None, max_iters=20):
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # vectorize appearance
            vec_i = self.interface.vectorize(i)

            # compute error image
            e = vec_i - self.appearance_model[0]

            # compute image gradient
            nabla_i = self.interface.gradient(i)

           # compute appearance jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._ds_dp)

            # compute hessian
            h = self.interface.H_a(j, self.appearance_model[1]) + self._H_s

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, self.appearance_model[1],
                                      self._H_s, shape_parameters[-1])

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return fitting result
        return self.interface.fitting_result(image, shape_parameters,
                                             gt_shape=gt_shape)


def extract_patches(image, centres, patch_shape, normalize_patches=False):
    # extract patches
    patches = image.extract_patches(centres, patch_size=patch_shape,
                                    as_single_array=False)

    # normalize if asked to
    if normalize_patches:
        for p in range(len(patches)):
            patches[p].normalize_norm_inplace()
    return patches


def vectorize_patches(patches):
    # find lengths
    patch_len = np.prod(patches[0].shape) * patches[0].n_channels
    n_points = len(patches)

    # initialize output matrix
    patches_vectors = np.empty(patch_len * n_points)

    # extract each vector
    for p in range(n_points):
        # find indices in target vector
        i_from = p * patch_len
        i_to = (p + 1) * patch_len

        # store vector
        patches_vectors[i_from:i_to] = patches[p].as_vector()

    return patches_vectors


def steepest_descent_image(image, dW_dp):
    # compute gradient
    # gradient:  height  x  width  x  (n_channels x n_dims)
    gradient = image.gradient(image)

    # reshape gradient
    # gradient:  n_pixels  x  (n_channels x n_dims)
    gradient = gradient.as_vector(keep_channels=True)

    # reshape gradient
    # gradient:  n_pixels  x  n_channels  x  n_dims
    gradient = np.reshape(gradient, (-1, image.n_channels,
                                     image.n_dims))

    # compute steepest descent images
    # gradient:  n_pixels  x  n_channels  x            x  n_dims
    # dW_dp:     n_pixels  x              x  n_params  x  n_dims
    # sdi:       n_pixels  x  n_channels  x  n_params
    sdi = np.sum(dW_dp[:, None, :, :] * gradient[:, :, None, :], axis=3)

    # reshape steepest descent images
    # sdi:  (n_pixels x n_channels)  x  n_params
    return np.reshape(sdi, (-1, dW_dp.shape[1]))