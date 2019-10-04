#!/usr/bin/env python

import numpy as np
import networkx as nx
import argparse
import logging
from qcommunity.utils.import_graph import generate_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--graph-generator",
        type=str,
        default="get_barbell_graph",
        help="graph generator function")
    parser.add_argument(
        "-l",
        type=int,
        default=3,
        help="number of vtx in the left (first) community")
    parser.add_argument(
        "-r",
        type=int,
        default=3,
        help="number of vtx in the right (second) community")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--graph-generator-seed",
        type=int,
        default=None,
        help="random seed, only used for graph generator")
    parser.add_argument(
        "--sobol", help="use sobol sequence as the random sequence", action="store_true")
    parser.add_argument(
        "--niter", type=int, default=100, help="number of iterations")
    parser.add_argument(
        "--label",
        type=str,
        help=
        "description of this version of the script. The description is prepended to the filename, so it should not contain any spaces. Default: time stamp"
    )
    parser.add_argument(
        "--weighted", help="if raised, the graph will be randomly weighted", action="store_true")
    parser.add_argument(
        "--remove-edge", 
        help="remove random edge from the graph", 
        action="store",
        const=-1, # hack! '-1' means remove random edge
        default=None,
        nargs='?',
        type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    test_all = True

    G_original, _ = generate_graph(args.graph_generator, args.l, args.r, seed=args.graph_generator_seed, weight=args.weighted, remove_edge=None)
    spectrum_original = nx.linalg.spectrum.laplacian_spectrum(G_original)
    print(spectrum_original)

    if test_all:
        max_norm = -1
        max_edge_ind = -1
        for edge_ind in range(G_original.number_of_edges()):
            G, _ = generate_graph(args.graph_generator, args.l, args.r, seed=args.graph_generator_seed, weight=args.weighted, remove_edge=edge_ind)
            spectrum = nx.linalg.spectrum.laplacian_spectrum(G)
            curr_norm = np.linalg.norm(spectrum_original - spectrum) 
            print("Removed edge\t{}\t with difference in Laplacian spectrum\t {}".format(edge_ind, curr_norm))
            if curr_norm > max_norm:
                max_norm = curr_norm
                max_edge_ind = edge_ind
        print("\n\nFor graph {} l={}, r={}, seed={} found optimal edge to remove: {} with difference in Laplacian spectrum: {}".format(args.graph_generator, args.l, args.r, args.graph_generator_seed, max_edge_ind, max_norm))
    else:
        G, _ = generate_graph(args.graph_generator, args.l, args.r, seed=args.graph_generator_seed, weight=args.weighted, remove_edge=args.remove_edge)
        spectrum = nx.linalg.spectrum.laplacian_spectrum(G)
        curr_norm = np.linalg.norm(spectrum_original - spectrum) 
        print("Difference in Laplacian spectrum: {}".format(curr_norm))
