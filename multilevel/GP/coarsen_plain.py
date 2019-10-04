#!/usr/bin/env python

"""
This script is for coarsening only

"""

import re, os, sys
import numpy as np
import networkx as nx
import random, copy
from os import listdir
from os.path import isfile, join

mydir = os.environ['ML_QUANTUM_WORKING_DIR']


def get_graphs():
    mypath_graphs = mydir + "multilevel_quantum/coarsening/coarsened_graphs/graphs/"
    graph_files = [f for f in listdir(mypath_graphs) \
    if isfile(join(mypath_graphs, f))]
    all_graphs = {}
    for graph_file in graph_files:
        if graph_file.endswith('.txt'):
            mygraph = read_metis_graph(mypath_graphs + graph_file)
            level = int(graph_file[5:-4])
            all_graphs[level] = mygraph
    return all_graphs


def get_uncoarsen_maps():
    mypath = mydir + "multilevel_quantum/coarsening/coarsened_graphs/maps/"
    map_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    all_maps = {}
    for map_file in map_files:
        if map_file.endswith('.txt'):
            mymap = get_map(mypath + map_file)
            level = int(map_file[3:-4])
            all_maps[level] = mymap
    return all_maps


def read_metis_graph(filename):
    """ read metis graph with edge weight and node volume"""
    graph = nx.Graph()
    myfile = open(filename, 'r')
    line_number = 0
    for line in myfile:
        line_number += 1
        line_split = [int(i) for i in line.split()]
        if line_number == 1:
            no_nodes = line_split[0]
            no_edges = line_split[1]
        else:
            node_weight = int(line_split[0])
            node = line_number - 2 # node index start at 0
            graph.add_node(node)
            graph.node[node]['volume'] = node_weight
            for i in xrange(1, len(line_split)-1, 2):
                neigh = int(line_split[i])-1 # minus 1 for node indexing
                edge_weight = int(line_split[i+1])
                graph.add_edge(node, neigh, weight=edge_weight)            
    assert nx.number_of_nodes(graph) == no_nodes
    assert nx.number_of_edges(graph) == no_edges
    return graph


def get_map(filename):
    myfile = open(filename, 'r')
    uncoarsen_map = {}
    node = 0
    for line in myfile:
        u = int(line)
        if u in uncoarsen_map:
            uncoarsen_map[u].append(node)
        else:
            uncoarsen_map[u] = [node]
        node +=1
    myfile.close()
    return uncoarsen_map
    
  
def coarsen_graph_with_kaffpa():
    graph_file = sys.argv[1]
    run_kahip = "./../coarsening/KaHiP/deploy/kaffpa_coarsen"
    kahip_config = "--k=2 --preconfiguration=fast"
    os.system(run_kahip + " " + graph_file + " " + kahip_config)
    # get maps
    all_maps = get_uncoarsen_maps()
    all_graphs = get_graphs()
    for level in all_graphs:
        print('\t level %i; %i nodes' %(level, nx.number_of_nodes(all_graphs[level])))
    Levels = []
    for level in all_graphs:
        Levels.append(level)
    Levels = sorted(Levels)
    coarsest_level = Levels.pop()
    finest_graph = all_graphs[0]
    coarsest_graph = all_graphs[coarsest_level]
    orig_graph_size = nx.number_of_nodes(finest_graph)
    print( "finest graph has %d nodes" %orig_graph_size)
    print("coarsest graph has %d nodes" %nx.number_of_nodes(coarsest_graph))
    return all_graphs, all_maps, coarsest_level

  
def main():
    os.system('./clean_data.sh')
    coarsen_graph_with_kaffpa()


if __name__ == '__main__':
    main()


