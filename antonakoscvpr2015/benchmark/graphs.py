import numpy as np


def parse_graph(graph_type):
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
        raise ValueError('Invalid graph str provided')
    return adjacency_array, root_vertex