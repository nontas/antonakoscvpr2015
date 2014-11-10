import os.path
import numpy as np

from menpofast.utils import convert_from_menpo, convert_to_menpo

import menpo.io as mio
from menpo.visualize import progress_bar_str, print_dynamic
from menpo.landmark import labeller

from antonakoscvpr2015.utils.base import pickle_dump, pickle_load
from .graphs import parse_deformation_graph, parse_appearance_graph


def load_database(path_to_images, save_path, db_name, crop_percentage,
                  fast, group, verbose=False):
    # create filename
    if group is not None:
        filename = (db_name + '_' + group.__name__ + '_crop' +
                    str(int(crop_percentage * 100)))
    else:
        filename = db_name + 'PTS' + '_crop' + str(int(crop_percentage * 100))
    if fast:
        filename += '_menpofast.pickle'
    else:
        filename += '_menpo.pickle'
    save_path = os.path.join(save_path, filename)

    # check if file exists
    if file_exists(save_path):
        if verbose:
            print_dynamic('Loading images...')
        images = pickle_load(save_path)
        if verbose:
            print_dynamic('Images Loaded.')
    else:
        # load images
        images = []
        for i in mio.import_images(path_to_images, verbose=verbose):
            if fast:
                i = convert_from_menpo(i)
            i.crop_to_landmarks_proportion_inplace(crop_percentage, group='PTS')
            if group is not None:
                labeller(i, 'PTS', group)
            if i.n_channels == 3:
                i = i.as_greyscale(mode='average')
            images.append(i)

        # save images
        pickle_dump(images, save_path)

    # return images
    return images


def train_aps(experiments_path, fast, group, training_images_options,
              training_options, save_model, verbose):
    # update training_images_options
    training_images_options['save_path'] = os.path.join(experiments_path,
                                                        'Databases')
    training_images_options['fast'] = fast
    training_images_options['group'] = group
    training_images_options['verbose'] = verbose

    # parse training options
    adj, rv = parse_deformation_graph(training_options['graph_deformation'])
    training_options['adjacency_array_deformation'] = adj
    training_options['root_vertex_deformation'] = rv
    adj, fl = parse_appearance_graph(training_options['graph_appearance'])
    training_options['adjacency_array_appearance'] = adj
    training_options['gaussian_per_patch'] = fl
    adj, _ = parse_appearance_graph(training_options['graph_shape'])
    training_options['adjacency_array_shape'] = adj
    training_options['features'] = parse_features(training_options['features'],
                                                  fast)
    graph_deformation_str = training_options['graph_deformation']
    graph_appearance_str = training_options['graph_appearance']
    graph_shape_str = training_options['graph_shape']
    del training_options['graph_deformation']
    del training_options['graph_appearance']
    del training_options['graph_shape']

    # Load training images
    training_images = load_database(**training_images_options)

    # make model filename
    filename = model_filename(training_images_options, training_options, group,
                              fast, graph_deformation_str, graph_appearance_str,
                              graph_shape_str)
    save_path = os.path.join(experiments_path, 'Models', filename)

    # train model
    if file_exists(save_path):
        if verbose:
            print_dynamic('Loading model...')
        aps = pickle_load(save_path)
        if verbose:
            print_dynamic('Model loaded.')
    else:
        training_options['max_shape_components'] = None

        # Train model
        if fast:
            from antonakoscvpr2015.menpofast.builder import APSBuilder
        else:
            from antonakoscvpr2015.menpo.builder import APSBuilder
        if group is not None:
            aps = APSBuilder(**training_options).build(training_images,
                                                       group=group.__name__,
                                                       verbose=verbose)
        else:
            aps = APSBuilder(**training_options).build(training_images,
                                                       verbose=verbose)

        # save model
        if save_model:
            pickle_dump(aps, save_path)

    return aps, filename, training_images


def fit_aps(aps, modelfilename, experiments_path, fast, group,
            fitting_images_options, fitting_options, verbose):
    # make results filename
    filename2 = results_filename(fitting_images_options, fitting_options, group,
                                 fast)

    # final save path
    filename = modelfilename[:modelfilename.rfind("_")+1] + '_' + filename2
    save_path = os.path.join(experiments_path, 'Results', filename)

    # fit model
    if file_exists(save_path):
        if verbose:
            print_dynamic('Loading fitting results...')
        fitting_results = pickle_load(save_path)
        if verbose:
            print_dynamic('Fitting results loaded.')
    else:
        fitter_cls, algorithm_cls = parse_algorithm(
            fast, fitting_options['algorithm'])
        fitter = fitter_cls(aps, algorithm=algorithm_cls, n_shape=[3, 6],
                            use_deformation=fitting_options['use_deformation'])

        # get fitting images
        fitting_images_options['save_path'] = os.path.join(experiments_path,
                                                           'Databases')
        fitting_images_options['fast'] = fast
        fitting_images_options['group'] = group
        fitting_images_options['verbose'] = verbose
        fitting_images = load_database(**fitting_images_options)

        # fit
        np.random.seed(seed=1)
        fitting_results = []
        n_images = len(fitting_images)
        if verbose:
            perc1 = 0.
            perc2 = 0.
            perc3 = 0.
        for j, i in enumerate(fitting_images):
            # fit
            if group is not None:
                gt_s = i.landmarks[group.__name__].lms
            else:
                gt_s = i.landmarks['PTS'].lms
            s = fitter.perturb_shape(gt_s,
                                     noise_std=fitting_options['noise_std'])
            fr = fitter.fit(i, s, gt_shape=gt_s,
                            max_iters=fitting_options['max_iters'])
            fitting_results.append(fr)

            # verbose
            if verbose:
                final_error = fr.final_error(error_type='me_norm')
                initial_error = fr.initial_error(error_type='me_norm')
                if final_error <= 0.03:
                    perc1 += 1.
                if final_error <= 0.04:
                    perc2 += 1.
                if final_error <= 0.05:
                    perc3 += 1.
                print_dynamic('- {0} - [<=0.03: {1:.1f}%, <=0.04: {2:.1f}%, '
                              '<=0.05: {3:.1f}%] - Image {4}/{5} (error: '
                              '{6:.3f} --> {7:.3f})'.format(
                              progress_bar_str(float(j + 1.) / n_images,
                                               show_bar=False),
                              perc1 * 100. / n_images, perc2 * 100. / n_images,
                              perc3 * 100. / n_images, j + 1, n_images,
                              initial_error, final_error))
        if verbose:
            print_dynamic('- Fitting completed: [<=0.03: {0:.1f}%, <=0.04: '
                          '{1:.1f}%, <=0.05: {2:.1f}%]\n'.format(
                          perc1 * 100. / n_images, perc2 * 100. / n_images,
                          perc3 * 100. / n_images))

        errors = []
        errors.append([fr.final_error() for fr in fitting_results])
        errors.append([fr.initial_error() for fr in fitting_results])
        pickle_dump(errors, save_path)
    return fitting_results, filename


def file_exists(filename):
    return os.path.isfile(filename)


def parse_features(features, fast):
    if features == 'no_op':
        if fast:
            from menpofast.feature import no_op
        else:
            from menpo.feature import no_op
        return no_op
    elif features == 'igo':
        if fast:
            from menpofast.feature import igo
        else:
            from menpo.feature import igo
        return igo
    elif features == 'double_igo':
        if fast:
            from menpofast.feature import double_igo
        else:
            from menpo.feature import double_igo
        return double_igo
    elif features == 'sift':
        if fast:
            from menpofast.feature import aam_dsift
        else:
            raise ValueError('SIFT only exist in the menpofast repo.')
        return aam_dsift
    else:
        raise ValueError('Invalid feature str provided')


def parse_algorithm(fast, algorithm):
    if fast:
        from antonakoscvpr2015.menpofast.fitter import LucasKanadeAPSFitter
        if algorithm == 'forward':
            from antonakoscvpr2015.menpofast.algorithm import Forward
            algorithm_cls = Forward
        elif algorithm == 'inverse':
            from antonakoscvpr2015.menpofast.algorithm import Inverse
            algorithm_cls = Inverse
        else:
            raise ValueError('Algorithm can be either forward or inverse.')
    else:
        from antonakoscvpr2015.menpo.fitter import LucasKanadeAPSFitter
        if algorithm == 'forward':
            from antonakoscvpr2015.menpo.algorithm import Forward
            algorithm_cls = Forward
        elif algorithm == 'inverse':
            from antonakoscvpr2015.menpo.algorithm import Inverse
            algorithm_cls = Inverse
        else:
            raise ValueError('Algorithm can be either forward or inverse.')
    return LucasKanadeAPSFitter, algorithm_cls


def model_filename(training_images_options, training_options, group, fast,
                   graph_deformation_str, graph_appearance_str,
                   graph_shape_str):
    filename = training_images_options['db_name']
    if group is not None:
        filename += '_' + group.__name__
    else:
        filename += '_PTS'
    filename += '_' + training_options['features'].__name__ + \
                '_def-' + graph_deformation_str + \
                '_app-' + graph_appearance_str + \
                '_sha-' + graph_shape_str + \
                '_patch' + str(training_options['patch_shape'][0]) + \
                '_norm' + str(training_options['normalization_diagonal']) + \
                '_lev' + str(training_options['n_levels']) + \
                '_sc' + str(int(training_options['downscale'] * 10)) + \
                '_app' + str(training_options['n_appearance_parameters'])
    if training_options['scaled_shape_models']:
        filename += '_scaledShape'
    else:
        filename += '_noScaledShape'
    if training_options['use_procrustes']:
        filename += '_procrustes'
    else:
        filename += '_noProcrustes'
    if fast:
        filename += '_menpofast'
    else:
        filename += '_menpo'
    filename += '.pickle'
    return filename


def results_filename(fitting_images_options, fitting_options, group, fast):
    filename = fitting_images_options['db_name']
    if group is not None:
        filename += '_' + group.__name__
    else:
        filename += '_PTS'
    filename += '_' + fitting_options['algorithm'] + '_n_sh' + \
                str(fitting_options['n_shape'])
    if fitting_options['use_deformation']:
        filename += '_def'
    else:
        filename += '_noDef'
    filename += '_noise' + str(int(fitting_options['noise_std'] * 100))
    filename += '_iters' + str(fitting_options['max_iters'])
    if fast:
        filename += '_menpofast'
    else:
        filename += '_menpo'
    filename += '.pickle'
    return filename
