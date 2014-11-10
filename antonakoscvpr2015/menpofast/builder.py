from __future__ import division

import numpy as np
from scipy.sparse import block_diag, lil_matrix

from menpo.visualize import print_dynamic, progress_bar_str
from menpo.feature import no_op
from menpo.shape import (Tree, UndirectedGraph, DirectedGraph)
from menpo.model import PCAModel

from menpofit.builder import (DeformableModelBuilder,
                              normalization_wrt_reference_shape)
from menpofit.base import create_pyramid
from menpofit import checks

from menpofast.utils import build_parts_image

from antonakoscvpr2015.utils.base import (_build_deformation_model,
                                          _build_shape_model,
                                          _check_n_parameters,
                                          _compute_minimum_spanning_tree,
                                          _covariance_matrix_inverse,
                                          _get_relative_locations,
                                          _procrustes_analysis)


class APSBuilder(DeformableModelBuilder):
    r"""
    """
    def __init__(self, adjacency_array_appearance=None, gaussian_per_patch=True,
                 adjacency_array_deformation=None, root_vertex_deformation=None,
                 adjacency_array_shape=None, features=no_op,
                 patch_shape=(17, 17), normalization_diagonal=None, n_levels=2,
                 downscale=2, scaled_shape_models=False, use_procrustes=True,
                 max_shape_components=None, n_appearance_parameters=None):
        # check parameters
        checks.check_n_levels(n_levels)
        checks.check_downscale(downscale)
        checks.check_normalization_diagonal(normalization_diagonal)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        features = checks.check_features(features, n_levels)
        n_appearance_parameters = _check_n_parameters(
            n_appearance_parameters, n_levels, 'n_appearance_parameters')

        # appearance graph
        if adjacency_array_appearance is None:
            self.graph_appearance = None
        elif adjacency_array_appearance == 'yorgos':
            self.graph_appearance = 'yorgos'
        else:
            self.graph_appearance = UndirectedGraph(adjacency_array_appearance)

        # shape graph
        if adjacency_array_shape is None:
            self.graph_shape = None
        else:
            self.graph_shape = UndirectedGraph(adjacency_array_shape)

        # check adjacency_array_deformation, root_vertex_deformation
        if adjacency_array_deformation is None:
            self.graph_deformation = None
            if root_vertex_deformation is None:
                self.root_vertex = 0
        else:
            if root_vertex_deformation is None:
                self.graph_deformation = DirectedGraph(adjacency_array_deformation)
            else:
                self.graph_deformation = Tree(adjacency_array_deformation,
                                              root_vertex_deformation)

        # store parameters
        self.features = features
        self.patch_shape = patch_shape
        self.normalization_diagonal = normalization_diagonal
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.max_shape_components = max_shape_components
        self.n_appearance_parameters = n_appearance_parameters
        self.use_procrustes = use_procrustes
        self.gaussian_per_patch = gaussian_per_patch

    def build(self, images, group=None, label=None, verbose=False):
        # compute reference_shape and normalize images size
        self.reference_shape, normalized_images = \
            normalization_wrt_reference_shape(images, group, label,
                                              self.normalization_diagonal,
                                              verbose=verbose)

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)

        # if graph_deformation not provided, compute the MST
        if self.graph_deformation is None:
            shapes = [i.landmarks[group][label] for i in normalized_images]
            self.graph_deformation = _compute_minimum_spanning_tree(
                shapes, self.root_vertex, '- ', verbose)

        # build the model at each pyramid level
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building model for each of the {} pyramid '
                              'levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building model\n')

        shape_models = []
        appearance_models = []
        deformation_models = []
        # for each pyramid level (high --> low)
        for j in range(self.n_levels):
            # since models are built from highest to lowest level, the
            # parameters in form of list need to use a reversed index
            rj = self.n_levels - j - 1

            level_str = '  - '
            if verbose:
                if self.n_levels > 1:
                    level_str = '  - Level {}: '.format(j + 1)

            # get feature images of current level
            feature_images = []
            for c, g in enumerate(generators):
                if verbose:
                    print_dynamic(
                        '{}Computing feature space/rescaling - {}'.format(
                        level_str,
                        progress_bar_str((c + 1.) / len(generators),
                                         show_bar=False)))
                feature_images.append(next(g))

            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label] for i in feature_images]

            # define shapes that will be used for training
            if j == 0:
                original_shapes = shapes
                train_shapes = shapes
            else:
                if self.scaled_shape_models:
                    train_shapes = shapes
                else:
                    train_shapes = original_shapes

            # apply procrustes if asked
            if self.use_procrustes:
                if verbose:
                    print_dynamic('{}Procrustes analysis'.format(level_str))
                train_shapes = _procrustes_analysis(train_shapes)

            # train shape model
            if verbose:
                print_dynamic('{}Building shape model'.format(level_str))
            shape_models.append(_build_shape_model(
                train_shapes, self.graph_shape,
                self.max_shape_components[rj]))

            # compute relative locations from all shapes
            relative_locations = _get_relative_locations(
                train_shapes, self.graph_deformation, level_str, verbose)

            # build and add deformation model to the list
            deformation_models.append(_build_deformation_model(
                self.graph_deformation, relative_locations, level_str, verbose))

            # extract patches from all images
            all_patches = _warp_images(feature_images, group, label,
                                       self.patch_shape,
                                       self.gaussian_per_patch, level_str,
                                       verbose)

            # build and add appearance model to the list
            if self.gaussian_per_patch:
                n_channels = feature_images[0].n_channels
                if self.graph_appearance is None:
                    # diagonal block covariance
                    n_points = images[0].landmarks[group][label].n_points
                    appearance_models.append(
                        _build_appearance_model_block_diagonal(
                            all_patches, n_points, self.patch_shape, n_channels,
                            self.n_appearance_parameters[rj], level_str,
                            verbose))
                else:
                    # sparse block covariance
                    appearance_models.append(_build_appearance_model_sparse(
                        all_patches, self.graph_appearance, self.patch_shape,
                        n_channels, self.n_appearance_parameters[rj], level_str,
                        verbose))
            else:
                if self.graph_appearance is None:
                    # full covariance
                    n_points = images[0].landmarks[group][label].n_points
                    patches_image_shape = (n_points, 1, images[0].n_channels) + self.patch_shape
                    appearance_models.append(_build_appearance_model_full(
                        all_patches, self.n_appearance_parameters[rj],
                        patches_image_shape, level_str, verbose))
                elif self.graph_appearance == 'yorgos':
                    # full covariance
                    n_points = images[0].landmarks[group][label].n_points
                    patches_image_shape = (n_points, 1, images[0].n_channels) + self.patch_shape
                    appearance_models.append(_build_appearance_model_full_yorgos(
                        all_patches, self.n_appearance_parameters[rj],
                        patches_image_shape, level_str, verbose))

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of models so that they are ordered from lower to
        # higher resolution
        shape_models.reverse()
        appearance_models.reverse()
        deformation_models.reverse()
        n_training_images = len(images)

        return self._build_aps(shape_models, deformation_models,
                               appearance_models, n_training_images)

    def _build_aps(self, shape_models, deformation_models, appearance_models,
                   n_training_images):
        r"""
        """
        from .base import APS
        return APS(shape_models, deformation_models, appearance_models,
                   n_training_images, self.graph_shape, self.graph_appearance,
                   self.graph_deformation, self.patch_shape, self.features,
                   self.reference_shape, self.downscale,
                   self.scaled_shape_models, self.use_procrustes)


def _warp_images(images, group, label, patch_shape, as_vectors, level_str,
                 verbose):
    r"""
    returns numpy.array of size (68*16*16*36) x n_images
    """
    # find length of each patch and number of points
    n_points = images[0].landmarks[group][label].n_points
    # TODO: introduce support for offsets
    patches_image_shape = (n_points, 1, images[0].n_channels) + patch_shape
    n_images = len(images)

    # initialize the output
    if as_vectors:
        all_patches = np.empty(patches_image_shape + (n_images,))
    else:
        all_patches = []

    # extract parts
    for c, i in enumerate(images):
        # print progress
        if verbose:
            print_dynamic('{}Extracting patches from images - {}'.format(
                level_str,
                progress_bar_str(float(c + 1) / len(images),
                                 show_bar=False)))

        # extract patches from this image
        patches_image = build_parts_image(
            i, i.landmarks[group][label], patch_shape)

        # store
        if as_vectors:
            all_patches[..., c] = patches_image.pixels
        else:
            all_patches.append(patches_image)

    return all_patches


def _build_appearance_model_block_diagonal(all_patches_array, n_points,
                                           patch_shape, n_channels,
                                           n_appearance_parameters, level_str,
                                           verbose):
    # build appearance model
    if verbose:
        print_dynamic('{}Training appearance distribution per '
                      'patch'.format(level_str))

    # compute mean appearance vector
    app_mean = np.mean(all_patches_array, axis=-1)

    # number of images
    n_images = all_patches_array.shape[-1]

    # compute covariance matrix for each patch
    all_cov = []
    for e in range(n_points):
        # print progress
        if verbose:
            print_dynamic('{}Training appearance distribution '
                          'per patch - {}'.format(
                          level_str,
                          progress_bar_str(float(e + 1) / n_points,
                                           show_bar=False)))
        # select patches and vectorize
        patches_vector = all_patches_array[e, ...].reshape(-1, n_images)

        # compute covariance
        cov_mat = np.cov(patches_vector)

        # compute covariance inverse
        inv_cov_mat = _covariance_matrix_inverse(cov_mat,
                                                 n_appearance_parameters)

        # store covariance
        all_cov.append(inv_cov_mat)

    # create final sparse covariance matrix
    return app_mean, block_diag(all_cov).tocsr()


def _build_appearance_model_sparse(all_patches_array, graph, patch_shape,
                                   n_channels, n_appearance_parameters,
                                   level_str, verbose):
    # build appearance model
    if verbose:
        print_dynamic('{}Training appearance distribution per '
                      'edge'.format(level_str))

    # compute mean appearance vector
    app_mean = np.mean(all_patches_array, axis=-1)

    # number of images
    n_images = all_patches_array.shape[-1]

    # appearance vector and patch vector lengths
    patch_len = np.prod(patch_shape) * n_channels

    # initialize block sparse covariance matrix
    all_cov = lil_matrix((graph.n_vertices * patch_len,
                          graph.n_vertices * patch_len))

    # compute covariance matrix for each edge
    for e in range(graph.n_edges):
        # print progress
        if verbose:
            print_dynamic('{}Training appearance distribution '
                          'per edge - {}'.format(
                          level_str,
                          progress_bar_str(float(e + 1) / graph.n_edges,
                                           show_bar=False)))

        # edge vertices
        v1 = np.min(graph.adjacency_array[e, :])
        v2 = np.max(graph.adjacency_array[e, :])

        # find indices in target covariance matrix
        v1_from = v1 * patch_len
        v1_to = (v1 + 1) * patch_len
        v2_from = v2 * patch_len
        v2_to = (v2 + 1) * patch_len

        # extract data
        edge_data = np.concatenate((all_patches_array[v1, ...].reshape(-1, n_images),
                                    all_patches_array[v2, ...].reshape(-1, n_images)))

        # compute covariance inverse
        icov = _covariance_matrix_inverse(np.cov(edge_data),
                                          n_appearance_parameters)

        # v1, v2
        all_cov[v1_from:v1_to, v2_from:v2_to] += icov[:patch_len, patch_len::]

        # v2, v1
        all_cov[v2_from:v2_to, v1_from:v1_to] += icov[patch_len::, :patch_len]

        # v1, v1
        all_cov[v1_from:v1_to, v1_from:v1_to] += icov[:patch_len, :patch_len]

        # v2, v2
        all_cov[v2_from:v2_to, v2_from:v2_to] += icov[patch_len::, patch_len::]

    return app_mean, all_cov.tocsr()


def _build_appearance_model_full(all_patches, n_appearance_parameters,
                                 patches_image_shape, level_str, verbose):
    # build appearance model
    if verbose:
        print_dynamic('{}Training appearance distribution'.format(level_str))

    # get mean appearance vector
    n_images = len(all_patches)
    tmp = np.empty(patches_image_shape + (n_images,))
    for c, i in enumerate(all_patches):
        tmp[..., c] = i.pixels
    app_mean = np.mean(tmp, axis=-1)

    # apply pca
    appearance_model = PCAModel(all_patches)

    # trim components
    if n_appearance_parameters is not None:
        appearance_model.trim_components(n_appearance_parameters)

    # compute covariance matrix
    app_cov = appearance_model.components.T.dot(np.diag(1/appearance_model.eigenvalues)).dot(appearance_model.components)

    return app_mean, app_cov


def _build_appearance_model_full_yorgos(all_patches, n_appearance_parameters,
                                        patches_image_shape, level_str, verbose):
    # build appearance model
    if verbose:
        print_dynamic('{}Training appearance distribution'.format(level_str))

    # get mean appearance vector
    n_images = len(all_patches)
    tmp = np.empty(patches_image_shape + (n_images,))
    for c, i in enumerate(all_patches):
        tmp[..., c] = i.pixels
    app_mean = np.mean(tmp, axis=-1)

    # apply pca
    appearance_model = PCAModel(all_patches)

    # trim components
    if n_appearance_parameters is not None:
        appearance_model.trim_components(n_appearance_parameters)

    # compute covariance matrix
    app_cov = np.eye(appearance_model.n_features, appearance_model.n_features) - appearance_model.components.T.dot(appearance_model.components)

    return app_mean, app_cov
