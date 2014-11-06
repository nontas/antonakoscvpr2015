import numpy as np


def parse_deformation_graph(graph_type):
    if graph_type == 'mst_68':
        # MST 68
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5], [ 5,  6],
             [ 6,  7], [ 7,  8], [ 8,  9], [ 8, 57], [ 9, 10], [57, 58],
             [57, 56], [57, 66], [10, 11], [58, 59], [56, 55], [66, 67],
             [66, 65], [11, 12], [65, 63], [12, 13], [63, 62], [63, 53],
             [13, 14], [62, 61], [62, 51], [53, 64], [14, 15], [61, 49],
             [51, 50], [51, 52], [51, 33], [64, 54], [15, 16], [49, 60],
             [33, 32], [33, 34], [33, 29], [60, 48], [32, 31], [34, 35],
             [29, 30], [29, 28], [28, 27], [27, 22], [27, 21], [22, 23],
             [21, 20], [23, 24], [20, 19], [24, 25], [19, 18], [25, 26],
             [25, 44], [18, 17], [18, 37], [44, 43], [44, 45], [37, 38],
             [45, 46], [38, 39], [46, 47], [39, 40], [47, 42], [40, 41],
             [41, 36]])
        root_vertex = 0
    elif graph_type == 'star_tree_68':
        # STAR 68
        adjacency_array = np.empty((67, 2), dtype=np.int32)
        for i in range(68):
            if i < 34:
                adjacency_array[i, 0] = 34
                adjacency_array[i, 1] = i
            elif i > 34:
                adjacency_array[i-1, 0] = 34
                adjacency_array[i-1, 1] = i
        root_vertex = 34
    elif graph_type == 'star_tree_51':
        # STAR 51
        adjacency_array = np.empty((50, 2), dtype=np.int32)
        for i in range(51):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        root_vertex = 16
    elif graph_type == 'star_tree_49':
        # STAR 49
        adjacency_array = np.empty((48, 2), dtype=np.int32)
        for i in range(49):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        root_vertex = 16
    else:
        raise ValueError('Invalid graph_deformation str provided')
    return adjacency_array, root_vertex


def parse_appearance_graph(graph_type):
    if graph_type == 'full':
        # FULL
        adjacency_array = None
        gaussian_per_patch = False
    elif graph_type == 'diagonal':
        # DIAGONAL
        adjacency_array = None
        gaussian_per_patch = True
    elif graph_type == 'full_graph_68':
        kk = 67*67 - sum(range(67))
        adjacency_array = np.empty((kk, 2), dtype=np.int)
        e = -1
        for i in range(67):
            k = i+1
            for j in range(k, 68, 1):
                e += 1
                adjacency_array[e, 0] = i
                adjacency_array[e, 1] = j
        gaussian_per_patch = True
    elif graph_type == 'per_facial_area':
        # FULL COV FOR EACH FACIAL AREA
        jaw = np.empty((16, 2), dtype=np.int)
        for i in range(16):
            jaw[i, 0] = i
            jaw[i, 1] = i + 1
        rbrow = np.array([[17, 18], [18, 19], [19, 20], [20, 21]], dtype=np.int)
        lbrow = np.array([[22, 23], [23, 24], [24, 25], [25, 26]], dtype=np.int)
        nose = np.array([[27, 28], [28, 29], [29, 30], [30, 33], [33, 32],
                         [32, 31], [33, 34], [34, 35]], dtype=np.int)
        reye = np.array([[36, 37], [37, 38], [38, 39], [39, 40], [40, 41],
                         [41, 36]], dtype=np.int)
        leye = np.array([[42, 43], [43, 44], [44, 45], [45, 46], [46, 47],
                         [47, 42]], dtype=np.int)
        mouth = np.array([[48, 49], [49, 50], [50, 51], [51, 52], [52, 53],
                         [53, 54], [54, 55], [55, 56], [56, 57], [57, 58],
                         [58, 59], [59, 48]], dtype=np.int)
        mouth2 = np.array([[60, 61], [61, 62], [62, 63], [63, 64], [64, 65],
                           [65, 66], [66, 67], [67, 60]], dtype=np.int)
        adjacency_array = np.concatenate((jaw, rbrow))
        adjacency_array = np.concatenate((adjacency_array, lbrow))
        adjacency_array = np.concatenate((adjacency_array, nose))
        adjacency_array = np.concatenate((adjacency_array, reye))
        adjacency_array = np.concatenate((adjacency_array, leye))
        adjacency_array = np.concatenate((adjacency_array, mouth))
        adjacency_array = np.concatenate((adjacency_array, mouth2))
        gaussian_per_patch = True
    elif graph_type == 'mst_68':
        # MST 68
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5], [ 5,  6],
             [ 6,  7], [ 7,  8], [ 8,  9], [ 8, 57], [ 9, 10], [57, 58],
             [57, 56], [57, 66], [10, 11], [58, 59], [56, 55], [66, 67],
             [66, 65], [11, 12], [65, 63], [12, 13], [63, 62], [63, 53],
             [13, 14], [62, 61], [62, 51], [53, 64], [14, 15], [61, 49],
             [51, 50], [51, 52], [51, 33], [64, 54], [15, 16], [49, 60],
             [33, 32], [33, 34], [33, 29], [60, 48], [32, 31], [34, 35],
             [29, 30], [29, 28], [28, 27], [27, 22], [27, 21], [22, 23],
             [21, 20], [23, 24], [20, 19], [24, 25], [19, 18], [25, 26],
             [25, 44], [18, 17], [18, 37], [44, 43], [44, 45], [37, 38],
             [45, 46], [38, 39], [46, 47], [39, 40], [47, 42], [40, 41],
             [41, 36]])
        gaussian_per_patch = True
    elif graph_type == 'star_tree_68':
        # STAR 68
        adjacency_array = np.empty((67, 2), dtype=np.int32)
        for i in range(68):
            if i < 34:
                adjacency_array[i, 0] = 34
                adjacency_array[i, 1] = i
            elif i > 34:
                adjacency_array[i-1, 0] = 34
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    elif graph_type == 'star_tree_51':
        # STAR 51
        adjacency_array = np.empty((50, 2), dtype=np.int32)
        for i in range(51):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    elif graph_type == 'star_tree_49':
        # STAR 49
        adjacency_array = np.empty((48, 2), dtype=np.int32)
        for i in range(49):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    else:
        raise ValueError('Invalid graph_appearance str provided')
    return adjacency_array, gaussian_per_patch
