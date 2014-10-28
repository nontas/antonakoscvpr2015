from __future__ import division
import numpy as np

from hdf5able import HDF5able, SerializableCallable
from menpofit.base import DeformableModel


class APS(DeformableModel, HDF5able):
    """
    """
    def __init__(self, shape_models, deformation_models, appearance_models,
                 n_training_images, tree, reference_shape, patch_shape,
                 features, sigma, scales, scale_shapes, scale_features):
        DeformableModel.__init__(self, features)
        self.shape_models = shape_models
        self.deformation_models = deformation_models
        self.appearance_models = appearance_models
        self.n_training_images = n_training_images
        self.tree = tree
        self.patch_shape = patch_shape
        self.features = features
        self.sigma = sigma
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    def h5_dict_to_serializable_dict(self):
        """
        """
        import menpo.feature
        d = self.__dict__.copy()

        features = d.pop('features')
        if self.scale_features:
            # features is a single callable
            d['features'] = SerializableCallable(features, [menpo.feature])
        else:
            # features is a list of callables
            d['features'] = [SerializableCallable(f, [menpo.feature])
                             for f in features]
        return d

    @property
    def n_levels(self):
        """
        """
        return len(self.scales)

    def instance(self, shape_weights=None, level=-1):
        r"""
        """
        sm = self.shape_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)

        return shape_instance

    def random_instance(self, level=-1):
        r"""
        """
        sm = self.shape_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)

        return shape_instance
