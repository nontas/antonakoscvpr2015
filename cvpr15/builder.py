from __future__ import division
from copy import deepcopy
import numpy as np

from menpo.transform import Scale
from menpo.shape import mean_pointcloud
from menpo.fitmultilevel.builder import (DeformableModelBuilder,
                                         build_shape_model)
from menpo.fitmultilevel import checks
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.feature import igo, gaussian_filter, hog

from menpo.fitmultilevel.functions import extract_local_patches_fast
from menpo.image import Image


class APSBuilder(DeformableModelBuilder):
    def __init__(self, features=hog, patch_shape=(16, 16),
                 normalize_patches=False, normalization_diagonal=None,
                 sigma=None, scales=(1., 0.5), scale_shapes=True,
                 scale_features=True, max_shape_components=None):
        # check parameters
        checks.check_normalization_diagonal(normalization_diagonal)
        n_levels = len(scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        features = checks.check_features(features, n_levels)

        # store parameters
        self.features = features
        self.patch_shape = patch_shape
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
        # compute reference_shape
        self.reference_shape = _compute_reference_shape(
            images, group, label, self.normalization_diagonal, verbose)

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

            # extract patches from all images
            level_images_shapes = [i.landmarks[group][label]
                                   for i in level_images]
            warped_images = _warp_images(level_images, level_images_shapes,
                                         self.patch_shape,
                                         self.normalize_patches, level_str,
                                         verbose)

            # build appearance model
            if verbose:
                print_dynamic('{}Training appearance distribution per '
                              'patch'.format(level_str))
            n_points = warped_images.shape[-1]
            patch_len = warped_images.shape[0]
            app_len = warped_images.shape[0]*warped_images.shape[2]
            app_mean = np.empty(app_len)
            app_cov = np.zeros((app_len, app_len))
            for p in range(n_points):
                # print progress
                if verbose:
                    print_dynamic('{}Training appearance distribution '
                                  'per patch - {}'.format(
                                  level_str,
                                  progress_bar_str(float(p + 1) / n_points,
                                                   show_bar=False)))
                # find indices in target mean and covariance matrices
                i_from = p * patch_len
                i_to = (p + 1) * patch_len
                # compute and store mean
                app_mean[i_from:i_to] = np.mean(warped_images[..., p], axis=1)
                # compute and store covariance
                app_cov[i_from:i_to, i_from:i_to] = np.cov(warped_images[..., p])

            # add appearance model to the list
            appearance_models.append((app_mean, app_cov))

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()
        n_training_images = len(images)

        return self._build_aps(shape_models, appearance_models,
                               n_training_images)

    def _build_aps(self, shape_models, appearance_models, n_training_images):
        r"""
        """
        from .base import APS
        return APS(shape_models, appearance_models, n_training_images,
                   self.reference_shape, self.patch_shape, self.features,
                   self.sigma, self.scales, self.scale_shapes,
                   self.scale_features)


def _compute_reference_shape(images, group, label, normalization_diagonal,
                             verbose):
    r"""
    """
    # the reference_shape is the mean shape of the images' landmarks
    if verbose:
        print_dynamic('- Computing reference shape')
    shapes = [i.landmarks[group][label] for i in images]
    ref_shape = mean_pointcloud(shapes)
    # fix the reference_shape's diagonal length if specified
    if normalization_diagonal:
        x, y = ref_shape.range()
        scale = normalization_diagonal / np.sqrt(x**2 + y**2)
        Scale(scale, ref_shape.n_dims).apply_inplace(ref_shape)
    return ref_shape


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


def extract_patch_vectors(image, centres, patch_shape, normalize_patches=False):
    r"""
    returns a numpy.array of size (16*16*36)x68
    """
    # Extract patches
    patches = extract_local_patches_fast(image, centres, patch_shape=patch_shape)

    # initialize output matrix
    patches_vectors = np.empty((np.prod(patches.shape[1:]), patches.shape[0]))

    # extract each vector
    for p in range(patches.shape[0]):
        # normalize parts
        if normalize_patches:
            # build parts image
            img = Image(patches[p, ...])

            # normalize
            img.normalize_norm_inplace()

            # extract vector
            patches_vectors[:, p] = img.as_vector()
        else:
            patches_vectors[:, p] = patches[p, ...].ravel()

    # vectorize parts
    return patches_vectors


def _warp_images(images, shapes, patch_shape, normalize_patches, level_str,
                 verbose):
    r"""
    returns numpy.array of size (16*16*36) x n_images x 68
    """
    # find length of each patch
    patches_len = np.prod(patch_shape) * images[0].n_channels

    # initialize an output numpy array
    patches_array = np.empty((patches_len, shapes[0].n_points, len(images)))

    # extract parts
    for c, (i, s) in enumerate(zip(images, shapes)):
        # print progress
        if verbose:
            print_dynamic('{}Extracting patches from images - {}'.format(
                level_str,
                progress_bar_str(float(c + 1) / len(images),
                                 show_bar=False)))
        # extract patches from this image
        patches_vectors = extract_patch_vectors(
            i, s, patch_shape=patch_shape, normalize_patches=normalize_patches)

        # store
        patches_array[..., c] = patches_vectors

    # rollaxis and return
    return np.rollaxis(patches_array, 2, 1)


def flatten_out(list_of_lists):
    return [i for l in list_of_lists for i in l]
