# General
import typing
import sys
import numpy as np
import itertools

#Networkx
import networkx as nx
# from stellargraph import StellarGraph
# from stellargraph.data import UniformRandomMetaPathWalk

# Sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score

# Pytorch
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import one_hot

PAD_VALUE = 0
def get_border_nodes(graph, subgraph):
    '''
    Returns (1) an array containing the border nodes of the subgraph (i.e. all nodes that have an edge to a node not in the subgraph, but are themselves in the subgraph)
    and (2) an array containing all of the nodes in the base graph that aren't in the subgraph
    '''

    # get all of the nodes in the base graph that are not in the subgraph
    non_subgraph_nodes = np.array(list(set(graph.nodes()).difference(set(subgraph.nodes()))))

    subgraph_nodes = np.array(list(subgraph.nodes()))
    A = nx.adjacency_matrix(graph).todense()

    # subset adjacency matrix to get edges between subgraph and non-subgraph nodes
    border_A = A[np.ix_(subgraph_nodes - 1,non_subgraph_nodes - 1)] # NOTE: Need to subtract 1 bc nodes are indexed starting at 1

    # the nodes in the subgraph are border nodes if they have at least one edge to a node that is not in the subgraph
    border_edge_exists = (np.sum(border_A, axis=1) > 0).flatten()
    border_nodes = subgraph_nodes[np.newaxis][border_edge_exists]
    return border_nodes, non_subgraph_nodes
