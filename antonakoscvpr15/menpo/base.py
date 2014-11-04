from __future__ import division
import numpy as np

from serializablecallable import SerializableCallable
from menpofit.base import DeformableModel
from menpo.shape import PointTree


class APS(DeformableModel):
    """
    """
    def __init__(self, shape_models, deformation_models, appearance_models,
                 n_training_images, tree, patch_shape, features,
                 reference_shape, downscale, scaled_shape_models):
        DeformableModel.__init__(self, features)
        self.n_training_images = n_training_images
        self.shape_models = shape_models
        self.deformation_models = deformation_models
        self.appearance_models = appearance_models
        self.tree = tree
        self.patch_shape = patch_shape
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models

    def __getstate__(self):
        import menpo.feature as menpo_feature
        d = self.__dict__.copy()

        features = d.pop('features')
        if self.pyramid_on_features:
            # features is a single callable
            d['features'] = SerializableCallable(features, [menpo_feature])
        else:
            # features is a list of callables
            d['features'] = [SerializableCallable(f, [menpo_feature])
                             for f in features]
        return d

    def __setstate__(self, state):
        try:
            state['features'] = state['features'].callable
        except AttributeError:
            state['features'] = [f.callable for f in state['features']]
        self.__dict__.update(state)

    @property
    def n_levels(self):
        """
        """
        return len(self.shape_models)

    def instance(self, shape_weights=None, level=-1, as_tree=False):
        r"""
        """
        sm = self.shape_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)
        if as_tree:
            shape_instance = PointTree(shape_instance.points,
                                       self.tree.adjacency_array,
                                       self.tree.root_vertex)

        return shape_instance

    def random_instance(self, level=-1, as_tree=False):
        r"""
        """
        sm = self.shape_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        if as_tree:
            shape_instance = PointTree(shape_instance.points,
                                       self.tree.adjacency_array,
                                       self.tree.root_vertex)

        return shape_instance

    def view_widget(self, n_parameters=5, parameters_bounds=(-3.0, 3.0),
                    mode='multiple', popup=False):
        r"""
        """
        from menpofit.visualize import visualize_shape_model
        visualize_shape_model(self.shape_models, n_parameters=n_parameters,
                              parameters_bounds=parameters_bounds,
                              figure_size=(7, 7), mode=mode, popup=popup)

    @property
    def _str_title(self):
        r"""
        """
        return 'Active Pictorial Structure'

    def __str__(self):
        r"""
        """
        return self._str_title
