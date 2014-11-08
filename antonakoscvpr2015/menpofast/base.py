from __future__ import division
import numpy as np

from serializablecallable import SerializableCallable
from menpofit.base import DeformableModel
from menpo.shape import PointTree, PointDirectedGraph, Tree


class APS(DeformableModel):
    """
    """
    def __init__(self, shape_models, deformation_models, appearance_models,
                 n_training_images, graph_shape, graph_appearance,
                 graph_deformation, patch_shape, features, reference_shape,
                 downscale, scaled_shape_models, use_procrustes):
        DeformableModel.__init__(self, features)
        self.n_training_images = n_training_images
        self.shape_models = shape_models
        self.deformation_models = deformation_models
        self.appearance_models = appearance_models
        self.graph_shape = graph_shape
        self.graph_appearance = graph_appearance
        self.graph_deformation = graph_deformation
        self.patch_shape = patch_shape
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.use_procrustes = use_procrustes

    def __getstate__(self):
        import menpofast.feature as menpo_feature
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

    def instance(self, shape_weights=None, level=-1, as_graph=False):
        r"""
        """
        sm = self.shape_models[level]

        if shape_weights is None:
            shape_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)
        if as_graph:
            if isinstance(self.graph_deformation, Tree):
                shape_instance = PointTree(
                    shape_instance.points,
                    self.graph_deformation.adjacency_array,
                    self.graph_deformation.root_vertex)
            else:
                shape_instance = PointDirectedGraph(
                    shape_instance.points,
                    self.graph_deformation.adjacency_array)

        return shape_instance

    def random_instance(self, level=-1, as_graph=False):
        r"""
        """
        sm = self.shape_models[level]

        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        if as_graph:
            if isinstance(self.graph_deformation, Tree):
                shape_instance = PointTree(
                    shape_instance.points,
                    self.graph_deformation.adjacency_array,
                    self.graph_deformation.root_vertex)
            else:
                shape_instance = PointDirectedGraph(
                    shape_instance.points,
                    self.graph_deformation.adjacency_array)

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
