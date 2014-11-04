from __future__ import division

import numpy as np
from scipy.misc import comb as nchoosek
from scipy.stats import multivariate_normal
from scipy.sparse import block_diag

from menpo.visualize import print_dynamic, progress_bar_str
from menpo.feature import no_op
from menpo.shape import PointTree, Tree, UndirectedGraph
from menpo.transform import Translation, GeneralizedProcrustesAnalysis
from menpo.model import PCAModel

from menpofit.builder import (DeformableModelBuilder,
                              normalization_wrt_reference_shape)
from menpofit.base import create_pyramid
from menpofit import checks

from menpofast.utils import build_parts_image


class APSBuilder(DeformableModelBuilder):
    def __init__(self, adjacency_array=None, root_vertex=0, features=no_op,
                 patch_shape=(17, 17), normalization_diagonal=None, n_levels=2,
                 downscale=2, scaled_shape_models=False, use_procrustes=True,
                 max_shape_components=None, n_appearance_parameters=None,
                 gaussian_per_patch=True):
        # check parameters
        checks.check_n_levels(n_levels)
        checks.check_downscale(downscale)
        checks.check_normalization_diagonal(normalization_diagonal)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        features = checks.check_features(features, n_levels)
        n_appearance_parameters = _check_n_parameters(
            n_appearance_parameters, n_levels, 'n_appearance_parameters')

        # flag whether to create MST
        self.tree_is_mst = adjacency_array is None
        if adjacency_array is not None:
            self.tree = Tree(adjacency_array, root_vertex)

        # store parameters
        self.root_vertex = root_vertex
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

        # if tree not provided, compute the MST
        shapes = [i.landmarks[group][label] for i in normalized_images]
        if self.tree_is_mst:
            self.tree = _compute_minimum_spanning_tree(shapes, self.root_vertex,
                                                       '- ', verbose)

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
                train_shapes, self.max_shape_components[rj]))

            # compute relative locations from all shapes
            relative_locations = _get_relative_locations(
                train_shapes, self.tree, level_str, verbose)

            # build and add deformation model to the list
            deformation_models.append(_build_deformation_model(
                self.tree, relative_locations, level_str, verbose))

            # extract patches from all images
            all_patches = _warp_images(feature_images, group, label,
                                       self.patch_shape, self.gaussian_per_patch, 
									   level_str, verbose)

            # build and add appearance model to the list
            if self.gaussian_per_patch:
                n_points = images[0].landmarks[group][label].n_points
                n_channels = feature_images[0].n_channels
                appearance_models.append(_build_appearance_model_per_patch(
                    all_patches, n_points, self.patch_shape, n_channels,
                    self.n_appearance_parameters[rj], level_str, verbose))
            else:
                appearance_models.append(_build_appearance_model_full(
                    all_patches, self.n_appearance_parameters[rj],
                    level_str, verbose))

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
                   n_training_images, self.tree, self.patch_shape,
                   self.features, self.reference_shape, self.downscale,
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


def _get_relative_locations(shapes, tree, level_str, verbose):
    r"""
    returns numpy.array of size 2 x n_images x n_edges
    """
    # convert given shapes to point trees
    point_trees = [PointTree(shape.points, tree.adjacency_array,
                             tree.root_vertex)
                   for shape in shapes]

    # initialize an output numpy array
    rel_loc_array = np.empty((2, tree.n_edges, len(point_trees)))

    # get relative locations
    for c, pt in enumerate(point_trees):
        # print progress
        if verbose:
            print_dynamic('{}Computing relative locations from '
                          'shapes - {}'.format(
                          level_str,
                          progress_bar_str(float(c + 1) / len(point_trees),
                                           show_bar=False)))

        # get relative locations from this shape
        rl = pt.relative_locations()

        # store
        rel_loc_array[..., c] = rl.T

    # rollaxis and return
    return np.rollaxis(rel_loc_array, 2, 1)


def _compute_minimum_spanning_tree(shapes, root_vertex, level_str, verbose):
    # initialize edges and weights matrix
    n_vertices = shapes[0].n_points
    n_edges = nchoosek(n_vertices, 2)
    weights = np.zeros((n_vertices, n_vertices))
    edges = np.empty((n_edges, 2), dtype=np.int32)

    # fill edges and weights
    e = -1
    for i in range(n_vertices-1):
        for j in range(i+1, n_vertices, 1):
            # edge counter
            e += 1

            # print progress
            if verbose:
                print_dynamic('{}Computing complete graph`s weights - {}'.format(
                    level_str,
                    progress_bar_str(float(e + 1) / n_edges,
                                     show_bar=False)))

            # fill in edges
            edges[e, 0] = i
            edges[e, 1] = j

            # create data matrix of edge
            diffs_x = [s.points[i, 0] - s.points[j, 0] for s in shapes]
            diffs_y = [s.points[i, 1] - s.points[j, 1] for s in shapes]
            coords = np.array([diffs_x, diffs_y])

            # compute mean
            m = np.mean(coords, axis=1)

            # compute covariance
            c = np.cov(coords)

            # get weight
            for im in range(len(shapes)):
                weights[i, j] += -np.log(multivariate_normal.pdf(coords[:, im],
                                                                 mean=m, cov=c))
            weights[j, i] = weights[i, j]

    # create undirected graph
    complete_graph = UndirectedGraph(edges)

    if verbose:
        print_dynamic('{}Minimum spanning tree computed.\n'.format(level_str))

    # compute minimum spanning tree
    return complete_graph.minimum_spanning_tree(weights, root_vertex)


def _build_deformation_model(tree, relative_locations, level_str, verbose):
    # build deformation model
    if verbose:
        print_dynamic('{}Training deformation distribution per '
                      'tree edge'.format(level_str))
    def_len = 2 * tree.n_vertices
    def_cov = np.zeros((def_len, def_len))
    for e in range(tree.n_edges):
        # print progress
        if verbose:
            print_dynamic('{}Training deformation distribution '
                          'per edge - {}'.format(
                          level_str,
                          progress_bar_str(float(e + 1) / tree.n_edges,
                                           show_bar=False)))

        # get vertices adjacent to edge
        parent = tree.adjacency_array[e, 0]
        child = tree.adjacency_array[e, 1]

        # compute covariance matrix
        edge_cov = np.linalg.inv(np.cov(relative_locations[..., e]))

        # store its values
        s1 = edge_cov[0, 0]
        s2 = edge_cov[1, 1]
        s3 = 2 * edge_cov[0, 1]

        # Fill the covariance matrix matrix
        # get indices
        p1 = 2 * parent
        p2 = 2 * parent + 1
        c1 = 2 * child
        c2 = 2 * child + 1

        # up-left block
        def_cov[p1, p1] += s1
        def_cov[p2, p2] += s2
        def_cov[p2, p1] += s3

        # up-right block
        def_cov[p1, c1] = - s1
        def_cov[p2, c2] = - s2
        def_cov[p1, c2] = - s3 / 2
        def_cov[p2, c1] = - s3 / 2

        # down-left block
        def_cov[c1, p1] = - s1
        def_cov[c2, p2] = - s2
        def_cov[c1, p2] = - s3 / 2
        def_cov[c2, p1] = - s3 / 2

        # down-right block
        def_cov[c1, c1] += s1
        def_cov[c2, c2] += s2
        def_cov[c1, c2] += s3

    return def_cov


def _build_appearance_model_per_patch(all_patches_array, n_points, patch_shape,
                                      n_channels, n_appearance_parameters,
                                      level_str, verbose):
    # build appearance model
    if verbose:
        print_dynamic('{}Training appearance distribution per '
                      'patch'.format(level_str))
    # compute mean appearance vector
    app_mean = np.mean(all_patches_array, axis=-1)

    # number of images
    n_images = all_patches_array.shape[-1]
    
	# appearance vector and patch vector lengths
    patch_len = np.prod(patch_shape) * n_channels

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

        # compute and store covariance
        cov_mat = np.cov(patches_vector)
        if n_appearance_parameters is None:
            all_cov.append(np.linalg.inv(cov_mat))
        else:
            s, v, d = np.linalg.svd(cov_mat)
            s = s[:, :n_appearance_parameters]
            v = v[:n_appearance_parameters]
            d = d[:n_appearance_parameters, :]
            all_cov.append(s.dot(np.diag(1/v)).dot(d))

    # create final sparse covariance matrix
    return app_mean, block_diag(all_cov).tocsr()


def _build_appearance_model_full(all_patches, n_appearance_parameters,
                                 level_str, verbose):
    # build appearance model
    if verbose:
        print_dynamic('{}Training appearance distribution'.format(level_str))

    # apply pca
    appearance_model = PCAModel(all_patches)

    # trim components
    if n_appearance_parameters is not None:
        appearance_model.trim_components(n_appearance_parameters)

    # get mean appearance vector
    app_mean = appearance_model.mean().as_vector()

    # compute covariance matrix
    app_cov = appearance_model.components.T.dot(np.diag(1/appearance_model.eigenvalues)).dot(appearance_model.components)

    return app_mean, app_cov


def _procrustes_analysis(shapes):
    # centralize shapes
    centered_shapes = [Translation(-s.centre()).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    return [s.aligned_source() for s in gpa.transforms]


def _build_shape_model(shapes, max_components):
    r"""
    Builds a shape model given a set of shapes.

    Parameters
    ----------
    shapes: list of :map:`PointCloud`
        The set of shapes from which to build the model.
    max_components: None or int or float
        Specifies the number of components of the trained shape model.
        If int, it specifies the exact number of components to be retained.
        If float, it specifies the percentage of variance to be retained.
        If None, all the available components are kept (100% of variance).

    Returns
    -------
    shape_model: :class:`menpo.model.pca`
        The PCA shape model.
    """
    # build shape model
    shape_model = PCAModel(shapes)
    if max_components is not None:
        # trim shape model if required
        shape_model.trim_components(max_components)

    return shape_model


def _check_n_parameters(n_params, n_levels, n_params_str):
    if n_params is not None:
        if type(n_params) is int or type(n_params) is float:
            n_params = [n_params] * n_levels
        elif len(n_params) == 1 and n_levels > 1:
            n_params = n_params * n_levels
        elif len(n_params) == n_levels:
            pass
        else:
            raise ValueError('{} can be an integer or a float or None '
                             'or a list containing 1 or {} of '
                             'those'.format(n_params_str, n_levels))
    return n_params
