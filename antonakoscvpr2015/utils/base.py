import cPickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from menpo.shape import PointTree, PointDirectedGraph, Tree


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


def plot_deformation_model(aps, level):
    mean_shape = aps.shape_models[level].mean().points
    for e in range(aps.graph.n_edges):
        # find vertices
        parent = aps.graph.adjacency_array[e, 0]
        child = aps.graph.adjacency_array[e, 1]

        # relative location mean
        rel_loc_mean = mean_shape[child, :] - mean_shape[parent, :]

        # relative location cov
        n_points = aps.deformation_models[0].shape[0] / 2
        s1 = -aps.deformation_models[level][2*child, 2*parent]
        s2 = -aps.deformation_models[level][2*child+1, 2*parent+1]
        s3 = -aps.deformation_models[level][2*child, 2*parent+1]
        cov_mat = np.linalg.inv(np.array([[s1, s3], [s3, s2]]))

        # plot ellipse
        plot_gaussian_ellipse(cov_mat, mean_shape[parent, :] + rel_loc_mean,
                              n_std=2, facecolor='none', edgecolor='r')

    # plot mean shape points
    aps.shape_models[level].mean().view_on(plt.gcf().number)

    # create and plot edge connections
    if isinstance(aps.graph, Tree):
        PointTree(mean_shape, aps.graph.adjacency_array,
                  aps.graph.root_vertex).view_on(plt.gcf().number)
    else:
        PointDirectedGraph(mean_shape,
                           aps.graph.adjacency_array).view_on(plt.gcf().number)
