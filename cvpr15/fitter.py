from __future__ import division
import numpy as np

from menpofit.fitter import MultilevelFitter
from menpofit.fittingresult import MultilevelFittingResult
from menpofit.transform.modeldriven import OrthoPDM
from menpofit.transform.homogeneous import AlignmentSimilarity
from menpo.transform.homogeneous import Scale

from .algorithm import APSInterface, Forward


class APSFitter(MultilevelFitter):
    def __init__(self, aps):
        self.aps = aps

    @property
    def reference_shape(self):
        return self.aps.reference_shape

    @property
    def features(self):
        return self.aps.features

    @property
    def n_levels(self):
        return self.aps.n_levels

    @property
    def downscale(self):
        r"""
        """
        return self.aps.downscale

    def _create_fitting_result(self, image, fitting_results, affine_correction,
                               gt_shape=None):
        r"""
        """
        return MultilevelFittingResult(
            image, self, fitting_results, affine_correction, gt_shape=gt_shape)

    def _fit(self, images, initial_shape, gt_shapes=None, max_iters=50,
             **kwargs):
        r"""
        """
        shape = initial_shape
        gt_shape = None
        n_levels = self.n_levels

        # check max_iters parameter
        if type(max_iters) is int:
            max_iters = [np.round(max_iters/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) == 1 and n_levels > 1:
            max_iters = [np.round(max_iters[0]/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) != n_levels:
            raise ValueError('max_iters can be integer, integer list '
                             'containing 1 or {} elements or '
                             'None'.format(self.n_levels))

        # fit images
        fitting_results = []
        for j, (i, f, it) in enumerate(zip(images, self._fitters, max_iters)):
            if gt_shapes is not None:
                gt_shape = gt_shapes[j]

            fitting_result = f.run(i, shape, gt_shape=gt_shape,
                                   max_iters=it, **kwargs)
            fitting_results.append(fitting_result)

            shape = fitting_result.final_shape
            Scale(self.downscale, n_dims=shape.n_dims).apply_inplace(shape)

        return fitting_results


class LucasKanadeAPSFitter(APSFitter):

    def __init__(self, aps, algorithm=Forward, md_transform=OrthoPDM,
                 n_shape=None, n_appearance=None, **kwargs):
        super(LucasKanadeAPSFitter, self).__init__(aps)
        self._set_up(algorithm=algorithm, md_transform=md_transform,
                     n_shape=n_shape, n_appearance=n_appearance, **kwargs)

    @property
    def algorithm(self):
        r"""
        """
        return 'LK-APS-' + self._fitters[0].algorithm

    def _set_up(self, algorithm=Forward, md_transform=OrthoPDM,
                global_transform=AlignmentSimilarity,
                n_shape=None, n_appearance=None, **kwargs):
        r"""
        """
        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.aps.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.aps.n_levels > 1:
                for sm in self.aps.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.aps.n_levels:
                for sm, n in zip(self.aps.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float or None '
                                 'or a list containing 1 or {} of '
                                 'those'.format(self.aps.n_levels))

        self._fitters = []
        for j, (am, sm, dm) in enumerate(zip(self.aps.appearance_models,
                                         self.aps.shape_models,
                                         self.aps.deformation_models)):
            pdm = OrthoPDM(sm, global_transform)
            self._fitters.append(algorithm(APSInterface, am, dm,
                                           self.aps.patch_shape, pdm, **kwargs))
