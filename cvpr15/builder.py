from __future__ import division
from copy import deepcopy
import numpy as np

from menpo.transform import Scale
from menpo.fitmultilevel.builder import (DeformableModelBuilder,
                                         build_shape_model,
                                         compute_reference_shape)
from menpo.fitmultilevel import checks
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.feature import igo, gaussian_filter

from menpo.shape import PointTree, Tree


class APSBuilder(DeformableModelBuilder):
    def __init__(self, adjacency_array=None, root_vertex=0, features=igo,
                 patch_size=(16, 16), normalize_patches=False,
                 normalization_diagonal=None, sigma=None, scales=(1., 0.5),
                 scale_shapes=True, scale_features=True,
                 max_shape_components=None):
        # check parameters
        checks.check_normalization_diagonal(normalization_diagonal)
        n_levels = len(scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        features = checks.check_features(features, n_levels)

        # create and store tree
        self.tree = Tree(adjacency_array, root_vertex)

        # store parameters
        self.adjacency_array = adjacency_array
        self.features = features
        self.patch_size = patch_size
        self.normalize_patches = normalize_patches
        self.normalization_diagonal = normalization_diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components

    def build(self, images, group=None, label=None, verbose=False):
        r"""
        """
        # extract the shapes from the images and compute the reference shape
        shapes = [i.landmarks[group][label] for i in images]
        self.reference_shape = compute_reference_shape(
            shapes, self.normalization_diagonal, verbose)

        # get normalized images and shapes
        images = _normalize_images(images, group, label, self.reference_shape,
                                   self.sigma, verbose)
        shapes = [i.landmarks[group][label] for i in images]

        # build the model at each pyramid level
        if verbose:
            if len(self.scales) > 1:
                print_dynamic('- Building model for each of the {} pyramid '
                              'levels\n'.format(len(self.scales)))
            else:
                print_dynamic('- Building model\n')

        shape_models = []
        appearance_models = []
        deformation_models = []
        # for each pyramid level (high --> low)
        for j, s in enumerate(self.scales):
            rj = len(self.scales) - j - 1

            if verbose:
                level_str = '  - '
                if len(self.scales) > 1:
                    level_str = '  - Level {}: '.format(j + 1)

            # obtain images of current level
            if j == 0:
                # compute features at highest level
                scaled_images = _scale_images(images, s, level_str, verbose)
                feature_images = _compute_features(scaled_images, self.features,
                                                   level_str, verbose)
                level_images = feature_images
            elif self.scale_features:
                # scale features at other levels
                level_images = _scale_images(feature_images, s, level_str,
                                             verbose)
            else:
                # scale images and compute features at other levels
                scaled_images = _scale_images(images, s, level_str, verbose)
                level_images = _compute_features(scaled_images, self.features,
                                                 level_str, verbose)

            # obtain shapes of current level
            if self.scale_shapes:
                scale_transform = Scale(scale_factor=s, n_dims=2)
                level_shapes = [scale_transform.apply(shape)
                                for shape in shapes]

            # train shape model
            if j == 0 or self.scale_shapes:
                # obtain shape model
                if verbose:
                    print_dynamic('{}Building shape model'.format(level_str))
                shape_model = build_shape_model(level_shapes,
                                                self.max_shape_components[rj])
                # add shape model to the list
                shape_models.append(shape_model)
            else:
                # copy precious shape model and add it to the list
                shape_models.append(deepcopy(shape_model))

            # compute relative locations from all shapes
            rel_locs = _get_relative_locations(level_shapes, self.tree,
                                               level_str, verbose)

            # build deformation model
            if verbose:
                print_dynamic('{}Training deformation distribution per '
                              'tree edge'.format(level_str))
            def_len = 2 * self.tree.n_vertices
            def_cov = np.zeros((def_len, def_len))
            for e in range(self.tree.n_edges):
                # print progress
                if verbose:
                    print_dynamic('{}Training deformation distribution '
                                  'per edge - {}'.format(
                                  level_str,
                                  progress_bar_str(float(e + 1) /
                                                   self.tree.n_edges,
                                                   show_bar=False)))

                # get vertices adjacent to edge
                parent = self.tree.adjacency_array[e, 0]
                child = self.tree.adjacency_array[e, 1]

                # compute covariance matrix
                edge_cov = np.linalg.inv(np.cov(rel_locs[..., e]))

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

            # add deformation model to the list
            deformation_models.append(def_cov)

            # extract patches from all images
            warped_images = _warp_images(level_images, group, label,
                                         self.patch_size,
                                         self.normalize_patches, level_str,
                                         verbose)

            # build appearance model
            if verbose:
                print_dynamic('{}Training appearance distribution per '
                              'patch'.format(level_str))
            n_points = warped_images.shape[-1]
            patch_len = warped_images.shape[0]
            app_len = patch_len * n_points
            app_mean = np.empty(app_len)
            app_cov = np.zeros((app_len, app_len))
            for e in range(n_points):
                # print progress
                if verbose:
                    print_dynamic('{}Training appearance distribution '
                                  'per patch - {}'.format(
                                  level_str,
                                  progress_bar_str(float(e + 1) / n_points,
                                                   show_bar=False)))
                # find indices in target mean and covariance matrices
                i_from = e * patch_len
                i_to = (e + 1) * patch_len
                # compute and store mean
                app_mean[i_from:i_to] = np.mean(warped_images[..., e], axis=1)
                # compute and store covariance
                cov_mat = np.cov(warped_images[..., e])
                app_cov[i_from:i_to, i_from:i_to] = np.linalg.inv(cov_mat)

            # add appearance model to the list
            appearance_models.append((app_mean, app_cov))

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
                   n_training_images, self.reference_shape, self.patch_size,
                   self.features, self.sigma, self.scales, self.scale_shapes,
                   self.scale_features)


def _normalize_images(images, group, label, reference_shape, sigma, verbose):
    r"""
    """
    # normalize the scaling of all images wrt the reference_shape size
    norm_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic('- Normalizing images size: {}'.format(
                progress_bar_str((c + 1.) / len(images), show_bar=False)))
        i = i.rescale_to_reference_shape(reference_shape, group=group,
                                         label=label)
        if sigma:
            i.pixels = fsmooth(i.pixels, sigma)
        norm_images.append(i)
    return norm_images


fsmooth = lambda x, sigma: gaussian_filter(x, sigma, mode='constant')


def _compute_features(images, features, level_str, verbose):
    r"""
    """
    feature_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic(
                '- Computing feature space: {}'.format(
                    level_str, progress_bar_str((c + 1.) / len(images),
                                                show_bar=False)))
        i = features(i)
        feature_images.append(i)

    return feature_images


def _scale_images(images, s, level_str, verbose):
    r"""
    """
    scaled_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic(
                '- Scaling images: {}'.format(
                    level_str, progress_bar_str((c + 1.) / len(images),
                                                show_bar=False)))
        scaled_images.append(i.rescale(s))
    return scaled_images


def extract_patch_vectors(image, group, label, patch_size,
                          normalize_patches=False):
    r"""
    returns a numpy.array of size (16*16*36) x 68
    """
    # extract patches
    patches = image.extract_patches_around_landmarks(
        group=group, label=label, patch_size=patch_size,
        as_single_array=not normalize_patches)

    # vectorize patches
    if normalize_patches:
        # initialize output matrix
        patches_vectors = np.empty(
            (np.prod(patches[0].shape) * patches[0].n_channels, len(patches)))

        # extract each vector
        for p in range(len(patches)):
            # normalize part
            patches[p].normalize_norm_inplace()

            # extract vector
            patches_vectors[:, p] = patches[p].as_vector()
    else:
        # initialize output matrix
        patches_vectors = np.empty((np.prod(patches.shape[1:]),
                                    patches.shape[0]))

        # extract each vector
        for p in range(patches.shape[0]):
            patches_vectors[:, p] = patches[p, ...].ravel()

    # return vectorized parts
    return patches_vectors


def _warp_images(images, group, label, patch_size, normalize_patches, level_str,
                 verbose):
    r"""
    returns numpy.array of size (16*16*36) x n_images x 68
    """
    # find length of each patch and number of points
    patches_len = np.prod(patch_size) * images[0].n_channels
    n_points = images[0].landmarks[group][label].n_points

    # initialize an output numpy array
    patches_array = np.empty((patches_len, n_points, len(images)))

    # extract parts
    for c, i in enumerate(images):
        # print progress
        if verbose:
            print_dynamic('{}Extracting patches from images - {}'.format(
                level_str,
                progress_bar_str(float(c + 1) / len(images),
                                 show_bar=False)))

        # extract patches from this image
        patches_vectors = extract_patch_vectors(
            i, group=group, label=label, patch_size=patch_size,
            normalize_patches=normalize_patches)

        # store
        patches_array[..., c] = patches_vectors

    # rollaxis and return
    return np.rollaxis(patches_array, 2, 1)


def _get_relative_locations(shapes, tree, level_str, verbose):
    r"""
    returns numpy.array of size 2 x n_images x n_edges
    """
    # convert point clouds to point trees
    point_trees = [PointTree(shape.points, adjacency_array=tree.adjacency_array,
                             root_vertex=tree.root_vertex)
                   for shape in shapes]

    # initialize an output numpy array
    rel_loc_array = np.empty((2, tree.n_edges, len(point_trees)))

    # get relative locations
    for c, pt in enumerate(point_trees):
        # print progress
        if verbose:
            print_dynamic('{}Computing relative locations from shapes - {}'.format(
                level_str,
                progress_bar_str(float(c + 1) / len(point_trees),
                                 show_bar=False)))

        # get relative locations from this shape
        rl = pt.relative_locations()

        # store
        rel_loc_array[..., c] = rl.T

    # rollaxis and return
    return np.rollaxis(rel_loc_array, 2, 1)


def flatten_out(list_of_lists):
    return [i for l in list_of_lists for i in l]
