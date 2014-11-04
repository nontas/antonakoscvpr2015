from __future__ import division
import abc
import numpy as np

from menpofit.fittingresult import SemiParametricFittingResult

from menpofast.image import Image
from menpofast.feature import gradient as fast_gradient
from menpofast.utils import build_parts_image, convert_to_menpo


class APSInterface(object):
    def __init__(self, algorithm, sampling_mask=None):
        self.algorithm = algorithm

        if sampling_mask is None:
            patch_shape = self.algorithm.patch_shape
            sampling_mask = np.require(np.ones((patch_shape)), dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling_mask[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.image_vec_mask = np.nonzero(image_mask.flatten())[0]
        self.gradient_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))

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
        returns an image object of size:
        n_points x patch_shape[0] x patch_shape[1] x n_channels
        """
        return build_parts_image(image, self.algorithm.transform.target,
                                 self.algorithm.patch_shape)

    def gradient(self, image):
        r"""
        Returns the gradients of the patches
        n_dims x n_parts x n_channels x  (w x h)
        """
        g = fast_gradient(image.pixels.reshape(
            (-1,) + self.algorithm.patch_shape))
        return g.reshape((2,) + image.pixels.shape)

    def steepest_descent_images(self, gradient, ds_dp):
        # reshape gradient
        # gradient: n_dims x n_parts x offsets x n_ch x (h x w)
        gradient = gradient[self.gradient_mask].reshape(
            gradient.shape[:-2] + (-1,))
        # compute steepest descent images
        # gradient: n_dims x n_parts x offsets x n_ch x (h x w)
        # ds_dp:    n_dims x n_parts x                          x n_params
        # sdi:               n_parts x offsets x n_ch x (h x w) x n_params
        sdi = 0
        a = gradient[..., None] * ds_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (n_parts x n_offsets x n_ch x w x h) x n_params
        return sdi.reshape((-1, sdi.shape[-1]))

    def ja(self, j, S):
        # compute the dot product between the apperance jacobian and the
        # covariance matrix
        # j: (n_parts x n_offsets x n_ch x w x h) x n_params
        # S: (n_parts x n_offsets x n_ch x w x h) x (n_parts x n_offsets x n_ch x w x h)
        return S.T.dot(j).T

    # def solve(self, h, ja, e, h_s, p, neg_sign):
    #     r"""
    #     Parameters increment
    #     dp --> (size: n_s x 1)
    #     h: hessian (H = H_a + H_s)
    #     ja: (J_a^T * S_a)
    #     e: error image
    #     sigma_a: appearance covariance (S_a)
    #     h_s: shape hessian (H_s)
    #     p: current shape parameters
    #     """
    #     b = ja.dot(e) + h_s.dot(p)
    #     if neg_sign:
    #         dp = -np.linalg.solve(h, b)
    #     else:
    #         dp = np.linalg.solve(h, b)
    #     return dp

    def fitting_result(self, image, shape_parameters, gt_shape):
        return SemiParametricFittingResult(image, self.algorithm,
                                           parameters=shape_parameters,
                                           gt_shape=gt_shape)


class APSAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aps_interface, appearance_model, deformation_model,
                 patch_shape, transform, use_procrustes, eps=10**-5, **kwargs):
        self.appearance_model = appearance_model
        self.deformation_model = deformation_model
        self.patch_shape = patch_shape
        self.use_procrustes = use_procrustes
        self.template = Image(appearance_model[0])
        self.transform = transform
        self.eps = eps

        # check if provided svd or covariance matrix
        self.use_svd = len(appearance_model) > 2

        # set interface
        self.interface = aps_interface(self, **kwargs)

        # mask appearance model
        self.Sigma_a = self.appearance_model[1]
        if len(self.interface.image_vec_mask) < self.Sigma_a.shape[0]:
            x, y = np.meshgrid(self.interface.image_vec_mask,
                               self.interface.image_vec_mask)
            self.Sigma_a = self.Sigma_a[x, y]

    @abc.abstractmethod
    def _precompute(self):
        pass

    @abc.abstractmethod
    def _algorithm_str(self):
        pass

    @abc.abstractmethod
    def run(self, **kwarg):
        pass


class Forward(APSAlgorithm):

    def __init__(self, aps_interface, appearance_model, deformation_model,
                 patch_shape, transform, eps=10**-5, **kwargs):
        # call super constructor
        super(Forward, self).__init__(aps_interface, appearance_model,
                                      deformation_model, patch_shape,
                                      transform, eps,  **kwargs)

        # pre-compute
        self._precompute()

    def _precompute(self):
        # compute warp jacobian
        self._ds_dp = self.interface.ds_dp()

        # compute shape hessian
        self._H_s = self.interface.H_s()

    def _algorithm_str(self):
        return 'Forward'

    def run(self, image, initial_shape, gt_shape=None, max_iters=20):
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        # masked model mean
        masked_m = self.template.as_vector()[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # masked image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_i - masked_m

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute appearance jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._ds_dp)

            # transposed jacobian and covariance dot product
            ja = self.interface.ja(j, self.Sigma_a)

            # compute hessian
            h = ja.dot(j) + self._H_s

            # compute gauss-newton parameter updates
            p = shape_parameters[-1].copy()
            if self.use_procrustes:
                p[0:4] = 0
            dp = np.linalg.solve(h, ja.dot(e) + self._H_s.dot(p))

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return fitting result
        return self.interface.fitting_result(image, shape_parameters,
                                             gt_shape=gt_shape)


class Inverse(APSAlgorithm):

    def __init__(self, aps_interface, appearance_model, deformation_model,
                 patch_shape, transform, eps=10**-5, **kwargs):
        # call super constructor
        super(Inverse, self).__init__(aps_interface, appearance_model,
                                      deformation_model, patch_shape,
                                      transform, eps, **kwargs)

        # pre-compute
        self._precompute()

    def _precompute(self):

        # compute warp jacobian
        ds_dp = self.interface.ds_dp()

        # compute shape hessian
        self._H_s = self.interface.H_s()

        # compute model's gradient
        nabla_a = self.interface.gradient(self.template)

        # compute appearance jacobian
        j = self.interface.steepest_descent_images(nabla_a, ds_dp)

        # transposed jacobian and covariance dot product
        self._ja = self.interface.ja(j, self.Sigma_a)

        # compute hessian inverse
        h = self._ja.dot(j) + self._H_s
        self._inv_h = np.linalg.inv(h)

    def _algorithm_str(self):
        return 'Inverse'

    def run(self, image, initial_shape, gt_shape=None, max_iters=20):
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        # masked model mean
        masked_m = self.template.as_vector()[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image
            i = self.interface.warp(image)

            # masked image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_i - masked_m

            # compute gauss-newton parameter updates
            p = shape_parameters[-1].copy()
            if self.use_procrustes:
                p[0:4] = 0
            dp = -self._inv_h.dot(self._ja.dot(e) + self._H_s.dot(p))

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return fitting result
        return self.interface.fitting_result(convert_to_menpo(image),
                                             shape_parameters,
                                             gt_shape=gt_shape)
