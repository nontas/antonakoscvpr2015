from __future__ import division
import abc
import numpy as np

from menpofit.fittingresult import SemiParametricFittingResult
from menpo.image import Image
from menpofit.transform.modeldriven import OrthoPDM
from .utils import build_patches_image, vectorize_patches_image


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
        returns an image object of size:
        n_points x patch_shape[0] x patch_shape[1] x n_channels
        """
        return build_patches_image(image, self.algorithm.transform.target,
                                   self.algorithm.patch_shape)

    def vectorize(self, patches_image):
        r"""
        Vectorization function A: It vectorizes a given patches image.
        Returns an ndarray of size:
        (n_points * patch_shape[0] * patch_shape[1] * n_channels) x 1
        """
        return vectorize_patches_image(patches_image)

    def gradient(self, patches):
        r"""
        Returns the gradients of the patches
        n_dims x n_channels x n_points x (w x h)
        """
        n_points = patches.pixels.shape[0]
        h = patches.pixels.shape[1]
        w = patches.pixels.shape[2]
        n_channels = patches.pixels.shape[3]

        # initial patches: n_parts x height x width x n_channels
        pixels = patches.pixels
        # move parts axis to end: height x width x n_channels x n_parts
        pixels = np.rollaxis(pixels, 0, pixels.ndim)
        # merge channels and parts axes: height x width x (n_channels * n_parts)
        pixels = np.reshape(pixels, (h, w, -1), order='F')
        # compute and vectorize gradient: (height * width) x (n_channels * n_parts * n_dims)
        g = Image(pixels).gradient().as_vector(keep_channels=True)
        # reshape gradient: (height * width) x n_parts x n_channels x n_dims
        # then transpose it: n_dims x n_channels x n_parts x (height * width)
        return np.reshape(g, (-1, n_points, n_channels, 2)).T

    def steepest_descent_images(self, gradient, ds_dp):
        # compute steepest descent images
        # gradient: n_dims x n_channels x n_parts x (w x h)
        # ds_dp:    n_dims x            x n_parts x         x n_params
        # sdi:               n_channels x n_parts x (w x h) x n_params
        sdi = 0
        a = gradient[..., None] * ds_dp[..., None, :, None, :]
        for d in a:
            sdi += d
        # sdi: n_parts x n_channels x (w x h) x n_params
        sdi = np.rollaxis(sdi, 1, 0)
        # sdi: n_parts x (w x h) x n_channels x n_params
        sdi = np.rollaxis(sdi, 2, 1)
        # sdi: (n_parts * w * h * n_channels) x n_params
        return sdi.reshape((-1, sdi.shape[3]))

    def ja(self, j, S):
        # compute the dot product between the appearance jacobian and the
        # covariance matrix
        # j: (n_parts * w * h * n_channels) x n_params
        # S: (n_parts * w * h * n_channels) x (n_parts * w * h * n_channels)
        return S.T.dot(j).T

    def solve(self, h, ja, e, h_s, p, neg_sign):
        r"""
        Parameters increment
        dp --> (size: n_s x 1)
        h: hessian (H = H_a + H_s)
        ja: (J_a^T * S_a)
        e: error image
        sigma_a: appearance covariance (S_a)
        h_s: shape hessian (H_s)
        p: current shape parameters
        """
        b = ja.dot(e)
        if h_s is not None:
            if isinstance(self.algorithm.transform, OrthoPDM):
                tmp_p = p.copy()
                tmp_p[0:4] = 0.
                b = b + h_s.dot(tmp_p)
            else:
                b = b + h_s.dot(p)
        if neg_sign:
            dp = -np.linalg.solve(h, b)
        else:
            dp = np.linalg.solve(h, b)
        return dp

    def fitting_result(self, image, shape_parameters, gt_shape):
        return SemiParametricFittingResult(image, self.algorithm,
                                           parameters=shape_parameters,
                                           gt_shape=gt_shape)


class APSAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aps_interface, appearance_model, deformation_model,
                 patch_shape, transform, use_deformation, eps=10**-5):
        self.appearance_model = appearance_model
        self.deformation_model = deformation_model
        self.patch_shape = patch_shape
        n_points = deformation_model.shape[0] / 2
        self.template = Image(np.reshape(appearance_model[0],
                                         (n_points, patch_shape[0],
                                          patch_shape[1], -1)))
        self.transform = transform
        self.use_deformation = use_deformation
        self.eps = eps

        # check if provided svd or covariance matrix
        self.use_svd = len(appearance_model) > 2

        # set interface
        self.interface = aps_interface(self)

    @abc.abstractmethod
    def _precompute(self):
        pass

    @abc.abstractmethod
    def _algorithm_str(self):
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
        # compute warp jacobian
        self._ds_dp = self.interface.ds_dp()

        # compute shape hessian
        self._H_s = None
        if self.use_deformation:
            self._H_s = self.interface.H_s()

    def _algorithm_str(self):
        return 'Forward'

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

            # transposed jacobian and covariance dot product
            ja = self.interface.ja(j, self.appearance_model[1])

            # compute hessian
            h = ja.dot(j)
            if self.use_deformation:
                h = h + self._H_s

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, ja, e, self._H_s,
                                      shape_parameters[-1], True)

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
                 patch_shape, transform, eps=10**-5):
        # call super constructor
        super(Inverse, self).__init__(aps_interface, appearance_model,
                                      deformation_model, patch_shape, transform,
                                      eps)

        # pre-compute
        self._precompute()

    def _precompute(self):
        # compute model's gradient
        nabla_a = self.interface.gradient(self.template)

        # compute warp jacobian
        ds_dp = self.interface.ds_dp()

        # compute appearance jacobian
        j = self.interface.steepest_descent_images(nabla_a, ds_dp)

        # transposed jacobian and covariance dot product
        self._ja = self.interface.ja(j, self.appearance_model[1])

        # compute hessian
        self._H_s = None
        self._h = self._ja.dot(j)
        if self.use_deformation:
            self._H_s = self.interface.H_s()
            self._h = self._h + self._H_s

    def _algorithm_str(self):
        return 'Inverse'

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

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self._h, self._ja, e, self._H_s,
                                      shape_parameters[-1], False)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() - dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return fitting result
        return self.interface.fitting_result(image, shape_parameters,
                                             gt_shape=gt_shape)
