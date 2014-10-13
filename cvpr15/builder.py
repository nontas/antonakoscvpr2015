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

#from cvpr15.cython.extract_patches import extract_patches_cython
#from .image import Image, MaskedImage


class APSBuilder(DeformableModelBuilder):
    def __init__(self, features=hog, patch_shape=(16, 16),
                 normalize_parts=False, normalization_diagonal=None, sigma=None,
                 scales=(1., 0.5), scale_shapes=True, scale_features=True,
                 max_shape_components=None):
        # check parameters
        checks.check_normalization_diagonal(normalization_diagonal)
        n_levels = len(scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        features = checks.check_features(features, n_levels)

        # store parameters
        self.features = features
        self.patch_shape = patch_shape
        self.normalize_parts = normalize_parts
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
                feature_images = _compute_features(scaled_images, level_str,
                                                   self.features, verbose)
                level_images = feature_images
            elif self.scale_features:
                # scale features at other levels
                level_images = _scale_images(feature_images, s, level_str,
                                             verbose)
            else:
                # scale images and compute features at other levels
                scaled_images = _scale_images(images, s, level_str, verbose)
                level_images = _compute_features(scaled_images, level_str,
                                                 self.features, verbose)

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

            # obtain warped images
            level_images_shapes = [i.landmarks[group][label]
                                   for i in level_images]
            warped_images = _warp_images(level_images, level_images_shapes,
                                         self.patch_shape, self.normalize_parts,
                                         level_str, verbose)

            # build appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(level_str))

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        n_training_images = len(images)

        return self._build_aps(shape_models, n_training_images)

    def _build_aps(self, shape_models, n_training_images):
        r"""
        """
        from .base import APS
        return APS(shape_models, n_training_images, self.reference_shape,
                   self.patch_shape, self.features, self.sigma, self.scales,
                   self.scale_shapes, self.scale_features)


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
        if features:
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


def _warp_images(images, shapes, parts_shape, normalize_parts, level_str,
                 verbose):
    r"""
    """
    # extract parts
    parts_images = []
    for c, (i, s) in enumerate(zip(images, shapes)):
        if verbose:
            print_dynamic('{}Warping images - {}'.format(
                level_str,
                progress_bar_str(float(c + 1) / len(images),
                                 show_bar=False)))
        parts_image = build_parts_image(
            i, s, parts_shape=parts_shape,
            normalize_parts=normalize_parts)
        parts_images.append(parts_image)

    return parts_images


def build_parts_image(image, centres, parts_shape, normalize_parts=False):
    r"""
    """
    parts = extract_local_patches_fast(image, centres, patch_shape=parts_shape)

    # build parts image
    img = Image(parts)

    if normalize_parts:
        # normalize parts
        #img.normalize_norm_inplace()
        print 'I was supposed to normalize'

    return img


def flatten_out(list_of_lists):
    return [i for l in list_of_lists for i in l]
