import cPickle
import numpy as np
from scipy.misc import comb as nchoosek
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from menpo.shape import (UndirectedGraph, Tree, PointUndirectedGraph, PointTree, PointDirectedGraph)
from menpo.visualize import progress_bar_str, print_dynamic
from menpo.transform import Translation, GeneralizedProcrustesAnalysis
from menpo.model import PCAModel

from antonakoscvpr2015.model.sparsepca import SparsePCAModel


def pickle_load(path):
    with open(str(path), 'rb') as f:
        return cPickle.load(f)


def pickle_dump(obj, path):
    with open(str(path), 'wb') as f:
        try:
            cPickle.dump(obj, f, protocol=2)
        except:
            cPickle.dump(obj, f)


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


def plot_deformation_model(aps, level, n_std):
    mean_shape = aps.shape_models[level].mean().points
    for e in range(aps.graph_deformation.n_edges):
        # find vertices
        parent = aps.graph_deformation.adjacency_array[e, 0]
        child = aps.graph_deformation.adjacency_array[e, 1]

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
                              n_std=n_std, facecolor='none', edgecolor='r')

    # plot mean shape points
    plt.scatter(aps.shape_models[level].mean().points[:, 0],
                aps.shape_models[level].mean().points[:, 1])
    #aps.shape_models[level].mean().pointsview_on(plt.gcf().number)

    # create and plot edge connections
    if isinstance(aps.graph_deformation, Tree):
        PointTree(mean_shape, aps.graph_deformation.adjacency_array,
                  aps.graph_deformation.root_vertex).view_on(plt.gcf().number)
    else:
        PointDirectedGraph(mean_shape,
                           aps.graph_deformation.adjacency_array).view_on(plt.gcf().number)


def plot_appearance_graph(aps, level):
    mean_shape = aps.shape_models[level].mean().points

    # plot mean shape points
    plt.scatter(aps.shape_models[level].mean().points[:, 0],
                aps.shape_models[level].mean().points[:, 1])

    # create and plot edge connections
    PointUndirectedGraph(mean_shape, aps.graph_appearance.adjacency_array).view_on(plt.gcf().number)


def plot_shape_graph(aps, level):
    mean_shape = aps.shape_models[level].mean().points

    # plot mean shape points
    plt.scatter(aps.shape_models[level].mean().points[:, 0],
                aps.shape_models[level].mean().points[:, 1])

    # create and plot edge connections
    PointUndirectedGraph(mean_shape, aps.graph_shape.adjacency_array).view_on(plt.gcf().number)


def _get_relative_locations(shapes, graph, level_str, verbose):
    r"""
    returns numpy.array of size 2 x n_images x n_edges
    """
    # convert given shapes to point graphs
    if isinstance(graph, Tree):
        point_graphs = [PointTree(shape.points, graph.adjacency_array,
                                  graph.root_vertex) for shape in shapes]
    else:
        point_graphs = [PointDirectedGraph(shape.points, graph.adjacency_array)
                        for shape in shapes]

    # initialize an output numpy array
    rel_loc_array = np.empty((2, graph.n_edges, len(point_graphs)))

    # get relative locations
    for c, pt in enumerate(point_graphs):
        # print progress
        if verbose:
            print_dynamic('{}Computing relative locations from '
                          'shapes - {}'.format(
                          level_str,
                          progress_bar_str(float(c + 1) / len(point_graphs),
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
        print_dynamic('{}Minimum spanning graph computed.\n'.format(level_str))

    # compute minimum spanning graph
    return complete_graph.minimum_spanning_tree(weights, root_vertex)


def _build_deformation_model(graph, relative_locations, level_str, verbose):
    # build deformation model
    if verbose:
        print_dynamic('{}Training deformation distribution per '
                      'graph edge'.format(level_str))
    def_len = 2 * graph.n_vertices
    def_cov = np.zeros((def_len, def_len))
    for e in range(graph.n_edges):
        # print progress
        if verbose:
            print_dynamic('{}Training deformation distribution '
                          'per edge - {}'.format(
                          level_str,
                          progress_bar_str(float(e + 1) / graph.n_edges,
                                           show_bar=False)))

        # get vertices adjacent to edge
        parent = graph.adjacency_array[e, 0]
        child = graph.adjacency_array[e, 1]

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


def _covariance_matrix_inverse(cov_mat, n_appearance_parameters):
    if n_appearance_parameters is None:
        return np.linalg.inv(cov_mat)
    else:
        s, v, d = np.linalg.svd(cov_mat)
        s = s[:, :n_appearance_parameters]
        v = v[:n_appearance_parameters]
        d = d[:n_appearance_parameters, :]
        return s.dot(np.diag(1/v)).dot(d)


def _procrustes_analysis(shapes):
    # centralize shapes
    centered_shapes = [Translation(-s.centre()).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    return [s.aligned_source() for s in gpa.transforms]


def _build_shape_model(shapes, graph_shape, max_components, verbose=False):
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
    if graph_shape is not None:
        shape_model = SparsePCAModel(graph_shape.adjacency_array, shapes, 2,
                                     verbose=verbose)
    else:
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
    else:
        n_params = [n_params] * n_levels
    return n_params
