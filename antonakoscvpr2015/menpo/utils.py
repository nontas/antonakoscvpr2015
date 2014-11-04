import cPickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from menpo.image import Image


def build_patches_image(image, centres, patch_shape):
    r"""
    Return the image patches as a menpo.image object
    size: n_points x patch_shape[0] x patch_shape[1] x n_channels
    """
    # extract patches
    if centres is None:
        patches = image.extract_patches_around_landmarks(patch_size=patch_shape,
                                                         as_single_array=True)
    else:
        patches = image.extract_patches(centres, patch_size=patch_shape,
                                        as_single_array=True)

    # build patches image
    return Image(patches)


def vectorize_patches_image(patches_image):
    r"""
    Return the image patches vector by calling the menpo.image.Image.as_vector()
    size: (n_points * patch_shape[0] * patch_shape[1] * n_channels) x 1
    """
    return patches_image.as_vector()


def pickle_load(path):
    with open(str(path), 'rb') as f:
        return cPickle.load(f)


def pickle_dump(obj, path):
    with open(str(path), 'wb') as f:
        cPickle.dump(obj, f, protocol=2)


def plot_gaussian_ellipse(cov, mean, n_std=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov
            The 2x2 covariance matrix to base the ellipse on
        mean
            The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        n_std
            The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax
            The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        **kwargs
            Keyword arguments are passed on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * n_std * np.sqrt(vals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip
