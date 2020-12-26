"""Ordinal Graph-based Oversampling via Shortest Paths

An implementation of the method described in:

Pérez-Ortiz, María, Pedro Antonio Gutiérrez, César Hervás-Martínez, and Xin Yao.
“Graph-Based Approaches for Over-Sampling in the Context of Ordinal Regression.”
IEEE Transactions on Knowledge and Data Engineering 27, no. 5 (May 2015): 1233–45.
https://doi.org/10.1109/TKDE.2014.2365780.

This implementation lets the user provide their own probability
distribution for the generation of synthetic samples in the inter-class edges.
"""

import random

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import pdist


def oversample_class(x, y, q, n, k, dist):
    """
    Oversample the class using the OGO-SP method (Perez-Ortiz et al, 2015)

    Parameters
    ----------
    x
        The dataset samples (n_samples, n_features)
    y
        The oridnal label of each sample as an integer (n_samples,)
    q
        The class label to augment
    n
        The number of samples to generate
    k
        The number of neighbours considered to construct the graph
    dist
        Distribution to use for inter-class edges generation

    Returns
    -------
    new_x: ndarray of shape (new_n_samples, n_features)
        The generated samples
    new_y: ndarray of shape (new_n_samples,)
        The corresponding labels
    """
    graph_edges = construct_graph(x, y, q, k)
    return oversample_class_with_graph(x, y, q, n, dist, graph_edges)


def oversample_class_with_graph(x, y, q, n, dist, graph_edges):
    """
    Oversample the class using the OGO-SP method (Perez-Ortiz et al, 2015)

    Parameters
    ----------
    x
        The dataset samples (n_samples, n_features)
    y
        The oridnal label of each sample as an integer (n_samples,)
    q
        The class label to augment
    n
        The number of samples to generate
    dist
        Distribution to use for inter-class edges generation
    graph_edges
        The edges of the graph formed by the projections

    Returns
    -------
    new_x: ndarray of shape (new_n_samples, n_features)
        The generated samples
    new_y: ndarray of shape (new_n_samples,)
        The corresponding labels
    """
    n_samples, n_features = x.shape
    q_indexes = np.arange(n_samples)[y == q]
    new_x = np.empty((n, n_features), dtype=x.dtype)
    new_y = np.repeat(q, n)
    for i, (xi_idx, xj_idx) in enumerate(random.choices(graph_edges, k=n)):
        assert (xi_idx in q_indexes) or (xj_idx in q_indexes)
        xi = x[xi_idx, :]
        xj = x[xj_idx, :]

        if (xi_idx in q_indexes) and (xj_idx in q_indexes):
            # Intra-class augmentation, uniform distribution
            ratio = np.random.random()
        else:
            # Class frontier augmentation, gamma distribution
            # Make sure xi is the one on X_q
            if xi_idx not in q_indexes:
                xi, xj = xj, xi
            # Saturate values >1 to just 1
            ratio = dist()
        new_x[i, :] = xi + ratio * (xj - xi)
    return new_x, new_y


def construct_graph(x, y, q, k):
    classes = np.unique(y)
    assert set(classes) == set(np.arange(classes.min(), classes.max()+1))
    assert q in classes

    n_samples = x.shape[0]

    def get_distance_function():
        pair_distances = pdist(x, 'euclidean')

        def _d(i, j):
            if i == j:
                # Distance to itself defined as infinite so that
                # nearest neighbors does not include itself
                return np.inf
            i, j = min(i, j), max(i, j)
            index = i*n_samples + j - (i+1)*(i+2)//2
            return pair_distances[index]
        return _d
    distance = get_distance_function()

    all_indices = np.arange(n_samples)
    this_class = all_indices[y == q]

    # E_{q,q}
    this_edges = _k_neighbor_edges(distance, this_class, this_class, k)
    this_verts = set(this_class)
    # Note that, because distance from a node to itself is defined as infinite,
    # this_edges will not include edges looping to the same node

    if (q-1) not in classes:
        next_class = all_indices[y == (q + 1)]

        # E_{q, q+1}
        next_edges = _k_neighbor_edges(distance, this_class, next_class, k)

        # Vertexes on the frontier for each adjacent class
        next_verts = frozenset.union(*next_edges) & set(next_class)

        edges = this_edges | next_edges

        src_verts = this_verts
        dst_verts = next_verts
    elif (q+1) not in classes:
        prev_class = all_indices[y == (q - 1)]

        # E_{q-1, q}
        prev_edges = _k_neighbor_edges(distance, this_class, prev_class, k)

        # Vertexes on the frontier for each adjacent class
        prev_verts = frozenset.union(*prev_edges) & set(prev_class)

        edges = prev_edges | this_edges

        src_verts = prev_verts
        dst_verts = this_verts
    else:
        prev_class = all_indices[y == (q-1)]
        next_class = all_indices[y == (q+1)]

        # E_{q-1, q}
        prev_edges = _k_neighbor_edges(distance, this_class, prev_class, k)
        # E_{q, q+1}
        next_edges = _k_neighbor_edges(distance, this_class, next_class, k)

        # Vertexes on the frontier for each adjacent class
        prev_verts = frozenset.union(*prev_edges) & set(prev_class)
        next_verts = frozenset.union(*next_edges) & set(next_class)

        edges = prev_edges | this_edges | next_edges
        src_verts = prev_verts
        dst_verts = next_verts

    adjacency_matrix = _adjacency_matrix_from_edges(edges, n_samples, distance)

    _, predecessors = dijkstra(adjacency_matrix, directed=False, return_predecessors=True)
    final_graph_edges = _core_edges(predecessors, src_verts, dst_verts)

    return [tuple(e) for e in final_graph_edges]


def _k_neighbor_edges(distance, x1, x2, k):
    """Find the edges defining the frontier of x1 and x2"""
    # N_d(X_1, X_2, k)
    edges12 = _k_neighborhood(distance, x1, x2, k)
    # N_d(X_2, X_1, k)
    edges21 = _k_neighborhood(distance, x2, x1, k)

    # Only edges in both sets are considered
    # These are the edges connecting the frontier between X_1 and X_2
    # E_{1, 2}
    return edges12 & edges21


def _k_neighborhood(distance, x1, x2, k):
    """Find the edges in the frontier of x1 with respect to x2"""
    edges = set()
    for i in x1:
        k_nearest_in_x2 = x2[np.argpartition([distance(i, j) for j in x2], k-1)[:k]]
        edges = edges | {frozenset({i, j}) for j in k_nearest_in_x2}
    return edges


def _adjacency_matrix_from_edges(edges, n_samples, distance):
    """Construct the sparse adjacency matrix from the edges and distances"""
    edges = [tuple(e) for e in edges]
    data = [distance(i, j) for i, j in edges]
    row_ind, col_ind = zip(*edges)
    return csr_matrix((data, (row_ind, col_ind)), shape=(n_samples, n_samples))


def _core_edges(predecessors, src_verts, dst_verts):
    """
    Obtain the edges that join prev_verts with next_verts
    through the shortest path in the graph
    """
    edges = set()
    for p in src_verts:
        for q in dst_verts:
            edges = edges | _edges_shortest_path(p, q, predecessors)
    return edges


def _edges_shortest_path(src, dst, predecessors):
    """
    Obtains the set of edges composing the shortest path
    between src and dst according to the predecessors matrix
    obtained with `scipy.sparse.csgraph.dijkstra`

    If there is no path, returns the empty set
    """
    path = list()
    step = dst
    while step != src:
        if step < 0:
            return set()
        path.append(step)
        step = predecessors[src, step]
    path.append(src)
    edges = set(frozenset(pair) for pair in zip(path[:-1], path[1:]))
    return edges
