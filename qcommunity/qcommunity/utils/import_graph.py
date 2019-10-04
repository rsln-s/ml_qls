#!/usr/bin/env python

# Imports KONECT graph as NetworkX undirected graph

import networkx as nx
import matplotlib.pyplot as plt
import argparse
import logging
import qcommunity.modularity.graphs as gm
import sys
import numpy as np
from networkx.utils.decorators import preserve_random_state
import random

def import_konect(fpath):
    G = nx.convert_node_labels_to_integers(
        nx.read_edgelist(
            fpath, comments='%', data=False, create_using=nx.OrderedGraph()))
    logging.info("Imported graph: {}".format(nx.info(G)))
    return G


def import_pajek(fpath):
    G = nx.convert_node_labels_to_integers(nx.Graph(nx.read_pajek(fpath)))
    logging.info("Imported graph: {}".format(nx.info(G)))
    return G


def import_edgelist(fpath):
    G = nx.convert_node_labels_to_integers(nx.Graph(nx.read_edgelist(fpath)))
    logging.info("Imported graph: {}".format(nx.info(G)))
    return G

def import_ilya_edgelist(fpath):
    graph = nx.OrderedDiGraph()
    with open(fpath, 'r') as f:
        header = f.readline().split()
        no_nodes = int(header[2])
        no_edges = int(header[3])
        for row in f:
            row_split = row.split()
            node1 = int(row_split[1])
            node2 = int(row_split[2])
            graph.add_edge(node1, node2, weight=1)            
    graph = nx.convert_node_labels_to_integers(graph)
    assert nx.number_of_nodes(graph) == no_nodes
    assert nx.number_of_edges(graph) == no_edges
    assert sorted(graph.nodes()) == list(range(nx.number_of_nodes(graph)))  # very very important! 
    return nx.OrderedGraph(graph)

@preserve_random_state
def generate_graph(graph_generator_name, left, right, seed=None, weight=False, remove_edge=False, p=None):
    try:
        graph_generator = getattr(gm, graph_generator_name)
    except AttributeError:
        print("Incorrect graph generator function: {}".format(
            graph_generator_name))
        sys.exit(0)
    try:
        if seed is None:
            G, solution_bitstring = graph_generator(left, right, p=p)
        else:
            G, solution_bitstring = graph_generator(left, right, p=p, seed=seed)
    except (TypeError, ValueError) as err:
        print("Incorrect community sizes received.", err)
        sys.exit(0)
    if weight:
        np.random.seed(seed)
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = np.random.randint(-1,2) # right side in exclusive: [-1,2)
    if remove_edge:
        state = random.getstate()
        random.seed(seed)
        gm.remove_random_edge(G, edge_num=remove_edge)
        random.setstate(state)
    return G, solution_bitstring


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fpath", type=str, help="path to KONECT edgelist (out.graphname file)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    G = import_pajek(args.fpath)
    import os
    nx.write_edgelist(G, os.path.basename(args.fpath))
