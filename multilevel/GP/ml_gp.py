#!/usr/bin/env python

import re, os, sys
import numpy as np
import networkx as nx
from numpy import linalg as la
from networkx.generators.atlas import *
import random, copy
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import time as mytime
import math
import pickle
import copy
from pyomo.environ import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'optimal/'))
import minimize_ising_model
from networkx.utils import reverse_cuthill_mckee_ordering
import os.path
import logging
import argparse
import configparser
from pathlib import Path
from state_watcher import StateWatcher

mydir = os.environ['ML_QUANTUM_WORKING_DIR']

results_dir = os.environ['ML_QUANTUM_RESULTS_DIR']
checkpoints_dir = os.environ['ML_QUANTUM_CHECKPOINTS_DIR']

state_watcher = None

def save_checkpoint():
    checkpoint_path = Path(checkpoints_dir, state_watcher.outpath.stem + \
                           '_lvl_{}_nevals_{}'.format(state_watcher.current_level, \
                            state_watcher.num_solver_calls) + state_watcher.outpath.suffix)
    print('Saving checkpoint to ', checkpoint_path)
    pickle.dump(state_watcher, open(checkpoint_path, 'wb'))

def get_graphs():
    mypath_graphs = mydir + "multilevel/coarsening/coarsened_graphs/graphs/"
    graph_files = [f for f in listdir(mypath_graphs) \
    if isfile(join(mypath_graphs, f))]
    all_graphs = {}
    for graph_file in graph_files:
        if graph_file.endswith('.txt'):
            mygraph = read_metis_graph(mypath_graphs + graph_file)
            level = int(graph_file[5:-4])
            #if level > 20:
            #    exit()
            all_graphs[level] = mygraph
    return all_graphs


def get_uncoarsen_maps():
    mypath = mydir + "multilevel/coarsening/coarsened_graphs/maps/"
    map_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    all_maps = {}
    for map_file in map_files:
        if map_file.endswith('.txt'):
            mymap = get_map(mypath + map_file)
            level = int(map_file[3:-4])
            all_maps[level] = mymap
    return all_maps

def compute_imbalances(part0, part1):
    if part0 == 0.0:
        imbalance10 = 1.0
        imbalance01 = 0.0
    elif part1 == 0.0:
        imbalance10 = 0.0
        imbalance01 = 1.0
    else:
        imbalance01 = part0/part1
        imbalance10 = part1/part0
    return imbalance01, imbalance10

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
            #if no_nodes >= 8000:
            #    exit()
        else:
            node_weight = int(line_split[0])
            node = line_number - 2 # node index start at 0
            graph.add_node(node)
            graph.node[node]['volume'] = node_weight
            for i in range(1, len(line_split)-1, 2):
                neigh = int(line_split[i])-1 # minus 1 for node indexing
                edge_weight = int(line_split[i+1])
                graph.add_edge(node, neigh, weight=edge_weight)            
    assert nx.number_of_nodes(graph) == no_nodes
    assert nx.number_of_edges(graph) == no_edges
    assert sorted(graph.nodes()) == list(range(nx.number_of_nodes(graph)))  # very very important! 
    return graph



def read_unweighted_metis_graph(filename):
    """ read unweighted metis graph"""
    graph = nx.OrderedGraph()
    myfile = open(filename, 'r')
    line_number = 0
    for line in myfile:
        line_number += 1
        line_split = [int(i) for i in line.split()]
        if line_number == 1:
            no_nodes = line_split[0]
            no_edges = line_split[1]
        else:
            node = line_number - 2 # node index start at 0
            graph.add_node(node)
            for i in range(len(line_split)):
                neigh = int(line_split[i])-1 # minus 1 for metis node indexing
                graph.add_edge(node, neigh, weight=1)            
    graph = nx.convert_node_labels_to_integers(graph)
    assert nx.number_of_nodes(graph) == no_nodes
    assert nx.number_of_edges(graph) == no_edges
    assert sorted(graph.nodes()) == list(range(nx.number_of_nodes(graph)))  # very very important! 
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
    
  
def coarsen_graph_with_kaffpa(seed):
    if state_watcher.args.coarsen_deg:
    #if state_watcher.args.problem_type == 'mod_via_GP' or state_watcher.args.problem_type == 'modularity':
        graph_file = create_deg_volume_graph_file()
    else:
        graph_file = state_watcher.args.fpath
    return run_kaffpa(seed, graph_file)


def run_kaffpa(seed, graph_file):
    run_kahip = "./../coarsening/KaHiP/deploy/kaffpa_coarsen"
    kahip_config = "--k=2 --preconfiguration=fast --seed=" + str(seed)
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
    print("finest graph has %d nodes" %orig_graph_size)
    print("coarsest graph has %d nodes" %nx.number_of_nodes(coarsest_graph))
    return all_graphs, all_maps, coarsest_level, graph_file


def initial_solution(coarsest_graph, objective = 'GP'):
    if objective == 'GP':
        return initial_bisection(coarsest_graph)
    else:
        raise ValueError('Only implemented for GP so far')


def initial_bisection(coarsest_graph, alpha, beta, hardware_size):
    if nx.number_of_nodes(coarsest_graph) > hardware_size:
        raise ValueError("ERROR: coarsest graph should fit in hardware")
        
    else:

        B_matrix, bias = formulate_GP_ising(coarsest_graph,
                                            alpha,
                                            beta,
                                            ptn_variables)
        solution_dict = solve_ising(B_matrix,
                                     bias,
                                     ising_obj = 'min',
                                     solver = 'qbsolv')
        return solution_dict
        

def formulate_GP_ising_old(graph,
                       alpha,
                       beta,
                       ptn_variables):
                       
    A = nx.adjacency_matrix(graph, nodelist=range(nx.number_of_nodes(graph)))
    vol_list = [graph.node[i]['volume'] for i in range(nx.number_of_nodes(graph))]
    vol_vec = np.array(vol_list).reshape(nx.number_of_nodes(graph), 1)
    vol_mat = vol_vec.transpose() * vol_vec
    B_matrix = alpha * vol_mat - beta * A

    
    # get sub hamiltonian
    free_var = []
    fixed_var = []
    fixed_vec = []
    free_nodes = []
    fixed_nodes = []
    
  
    for node in range(nx.number_of_nodes(graph)):
        if ptn_variables[node] == 'free':
            free_var.append(True)
            fixed_var.append(False)
            free_nodes.append(node)
        else:
            free_var.append(False)
            fixed_var.append(True)
            fixed_vec.append(ptn_variables[node])
            fixed_nodes.append(node)

    free_var = np.array(free_var)
    sub_B_matrix = B_matrix[free_var][:, free_var]
    bias = []
    for node_i in free_nodes:
        bias_i = 0
        for node_j in fixed_nodes:
            s_j = ptn_variables[node_j]
            bias_i += 2 * s_j * B_matrix.item((node_i, node_j))
        bias.append(bias_i)
    fixed_var = np.array(fixed_var)
    C = B_matrix[fixed_var][:, fixed_var]
    n = C.shape[0]
    vec = np.array(fixed_vec).reshape(n, 1)
    constant = vec.transpose() * C * vec

    return sub_B_matrix, bias, constant.item(0)

def scale_list(mylist, max_value):
    m = max(mylist) + 0.0
    for i in range(len(mylist)):
        mylist[i] = (mylist[i]/m) * max_value  # smooth
        #mylist[i] = min(mylist[i], max_value)  # values look like [1, 2,,45, 50, 50, ...]
    return


def formulate_GP_ising(graph,
                       alpha,
                       beta,
                       ptn_variables):
    """ Description in refinement section in multilevel paper"""
    if 'free' not in ptn_variables.values():
        raise ValueError('At least one node must be a variable')
    n = nx.number_of_nodes(graph)
    A = nx.adjacency_matrix(graph, nodelist=range(n))
    vol_list = []
    for i in range(n):
        vol = graph.node[i]['volume']
        vol_list.append(vol)
    if state_watcher.args.solver == 'dwave':
        scale_list(vol_list, 50)

    is_free = [True if ptn_variables[i] == 'free' else False for i in range(n)]


    is_fixed = [False if ptn_variables[i] == 'free' else True for i in range(n)]
    is_free = np.array(is_free, dtype=bool)
    is_fixed = np.array(is_fixed, dtype=bool)
    s_f_list = [ptn_variables[i] for i in range(n) if ptn_variables[i] != 'free']
    s_f = np.array(s_f_list).reshape(len(s_f_list), 1)
    vol_vec = np.array(vol_list).reshape(1, n)
    v_v = vol_vec[:, is_free]
    v_v = v_v.transpose()  # in order to match paper
    v_f = vol_vec[:, is_fixed]
    v_f = v_f.transpose()  # in order to match paper
    A_vv = A[is_free][:, is_free]
    A_vf = A[is_free][:, is_fixed]
    sub_B_mat = alpha * v_v.dot(v_v.transpose())
    sub_B_mat -= beta*A_vv
    if s_f_list != []:
        sub_bias =  2 * alpha * v_v.dot(v_f.transpose()) 
        sub_bias = sub_bias.dot(s_f)

        sub_bias -= 2 * beta* A_vf.dot(s_f)
    else:
        num_free = n - len(s_f_list)
        #sub_bias = np.array([0]*num_free).reshape(1,num_free)
        sub_bias = np.array([0]*num_free).reshape(num_free, 1)

    return sub_B_mat, sub_bias


def formulate_modularity_ising(graph, ptn_variables):
    n = nx.number_of_nodes(graph)
    A = nx.adjacency_matrix(graph, nodelist=range(n), weight='weight')
    #total_edges = 0
    #for u, v in graph.edges():
    #   total_edges += graph[u][v]['weight']
    #total_edges = sum([graph.node[i]['volume'] for i in range(n)])
    #deg_list = [graph.degree(i, weight='weight') for i in range(n)]
    deg_list = []
    for i in range(n):
        vol = graph.node[i]['volume']  # in modularity, volume of node = degree of node
        deg_list.append(vol)
    #if state_watcher.args.solver == 'dwave':
    #    scale_list(deg_list, 50)
    total_edges = sum(deg_list)/2.0
    is_free = [True if ptn_variables[i] == 'free' else False for i in range(n)]
    is_fixed = [False if ptn_variables[i] == 'free' else True for i in range(n)]
    is_free = np.array(is_free, dtype=bool)
    is_fixed = np.array(is_fixed, dtype=bool)
    s_f_list = [ptn_variables[i] for i in range(n) if ptn_variables[i] != 'free']
    s_f = np.array(s_f_list).reshape(len(s_f_list), 1)
    deg_vec = np.array(deg_list).reshape(1, n)
    k_v = deg_vec[:, is_free]
    
    k_v = k_v.transpose()  # in order to match paper
    k_f = deg_vec[:, is_fixed]
    k_f = k_f.transpose()  # in order to match paper
    A_vv = A[is_free][:, is_free]
    A_vf = A[is_free][:, is_fixed]
    alpha = 1.0/(2*total_edges)
    beta = 1.0
    sub_B_mat = alpha * k_v.dot(k_v.transpose())
    sub_B_mat -= beta*A_vv
    if s_f_list != []:
        sub_bias = 2 * alpha * k_v.dot(k_f.transpose())
        sub_bias = sub_bias.dot(s_f)
        sub_bias -= 2 * beta * A_vf.dot(s_f)
    else:
        num_free = n - len(s_f_list)
        sub_bias = np.array([0]*num_free).reshape(num_free, 1)
    
    return sub_B_mat, sub_bias


def solve_ising(B_matrix,
                bias,
                ising_obj = 'min',
                solver = 'qbsolv'):
    # TODO @Hayato: incrementing num_solver calls should be
    #  moved into state_watcher.record_refinement_step
    # everything for record_refinement_step should be computed
    #  right here in this function and not in refine_GP_solution, as it is now
    state_watcher.num_solver_calls += 1  # global variable
    print(B_matrix.shape, bias.shape)
    if solver == 'qbsolv':
        Q_matrix, Q_bias = ising_to_qubo(B_matrix, bias)
        solution_dict = solve_qubo_qbsolv(Q_matrix, Q_bias, qubo_obj = ising_obj)
    elif solver == 'dwave':
        solution_dict = solve_ising_dwave(B_matrix, bias, ising_obj = ising_obj)
    elif solver == 'qaoa':
        solution_dict = solve_ising_qaoa(B_matrix, bias, ising_obj = ising_obj)
    elif solver == 'pyomo':       
        solution_dict = solve_ising_pyomo(B_matrix, bias, ising_obj = ising_obj)
    return solution_dict
    


def ising_to_qubo(B_matrix, bias):
    """
    From https://arxiv.org/pdf/1705.03082.pdf equation (21) [page 8/20]
    However, also add bias term converted to QUBO
    sT*M*s + sT*b= 4xT*M*x  - 4xT*M*ones + ones*M*ones + 2xT*b - onesT*b
    or
    sT*M*s + sT*b = 4xT*M*x - 4xT*M*ones + 2xT*b + ones*M*ones - onesT*b
    Q_matrix = 4*M
    Q_bias = -4*M*ones + 2 * bias
    constant =  ones*M*ones - onesT*b  -- we shall ignore this for now
    """
    Q_matrix = 4 * B_matrix
    n = B_matrix.shape[0]
    ones = np.ones((n,1))
    _bias = np.array(bias).reshape(n, 1)
    _Q_bias = -4 * B_matrix.dot(ones) + 2 * _bias
    Q_bias = _Q_bias.transpose().tolist()
    return Q_matrix, Q_bias[0]



def get_qubo_solution():
    myFile = open("qubo/qbsolv_output.out", 'r')
    line_count = 0
    for lines in myFile:
        line_count += 1
        if line_count == 2:
            bit_string = lines
            break
    return bit_string.strip()


def qubo_to_file(Q_matrix, Q_bias):
    n = Q_matrix.shape[0]
    maxDiagonals = n
    nDiagonals = n
    nElements = 0
    qubo_string = ""

    qubo_string += "c nodes first \n"
    for i in range(n):
        if  Q_bias[i] > 0.0000001 or Q_bias[i] < -0.0000001:
            qubo_string += " ".join([str(i), str(i), str(Q_bias[i]), '\n'])
        
    qubo_string += "c couplers \n"
    for i in range(n):
        for j in range(i,n):
            if i != j:
                if Q_matrix[i, j] > 0.0000001 or Q_matrix[i, j] < -0.0000001:
                    temp = [i, j, 2 * Q_matrix[i,j]]
                    qubo_string += " ".join([str(_) for _ in temp])
                    qubo_string += '\n'
                    nElements += 1
    qfile = open("qubo/graph.qubo", 'w')
    initial = "p qubo 0"
    out = " ". join([initial, str(maxDiagonals), str(nDiagonals), str(nElements), "\n"])
    qfile.write(out)
    qfile.write(qubo_string)
    qfile.close()
    
    
    
def solve_qubo_qbsolv(Q_matrix, Q_bias, qubo_obj = 'min'):

    if qubo_obj == 'max':
        Q_matrix = -Q_matrix
        Q_bias = [-i for i in Q_bias]
    # write QUBO to file
    qubo_to_file(Q_matrix, Q_bias)
    # solve with qbsolv
    seed = '0'  # needs to be input, although not big difference in results seen
    os.system("qbsolv -i qubo/graph.qubo -r "+seed+" > qubo/qbsolv_output.out")
    bit_string = get_qubo_solution()
    n = Q_matrix.shape[0]
    solution_dict = {}
    for i in range(n):
        solution_dict[i] = 2*int(bit_string[i]) - 1
    return solution_dict


def solve_ising_dwave(B_matrix, B_bias, ising_obj = 'min'):
    raise ValueError("Not supported!")
    return solution_dict


def solve_ising_qaoa(B_matrix, B_bias, ising_obj = 'min'):
    from qcommunity.optimization.optimize import optimize_ising
    from ibmqxbackend.ansatz import IBMQXVarForm
    backend='IBMQX'
    backend_params={
        'depth': 1,
        'var_form':'RYRZ',
        'backend_device': state_watcher.args.qaoa_backend,
        'n_iter': state_watcher.args.niter_qaoa,
        #'opt_method': 'COBYLA_NLOPT'
        'opt_method': 'SBPLX_NLOPT'
    }
    from qcommunity.utils.ising_obj import ising_objective as rs_ising_objective
    problem_description = {
                            'name': 'ising',
                            'B_matrix':B_matrix,
                            'B_bias':B_bias,
                            'ising_obj':ising_obj,
                            'objective_function':rs_ising_objective,
                            'num_qubits':B_matrix.shape[0]
                        }

    if backend_params['backend_device'] is None or "simulator" in backend_params['backend_device']:
        target_backend_name = None
    else:
        target_backend_name = backend_params['backend_device']

    if problem_description['num_qubits'] in state_watcher.var_forms:
        var_form = state_watcher.var_forms[problem_description['num_qubits']]
    else:
        var_form = IBMQXVarForm(
            problem_description, 
            depth=backend_params['depth'], 
            var_form=backend_params['var_form'], 
            target_backend_name=target_backend_name)
        state_watcher.var_forms[problem_description['num_qubits']] = var_form
    qaoa_solution = optimize_ising(problem_description, backend=backend, backend_params=backend_params, var_form_obj=var_form)
    print("QAOA solution : ", qaoa_solution)
    print("QAOA energy : ", rs_ising_objective({'B_matrix':B_matrix, 'B_bias':B_bias}, qaoa_solution))

    try:
        pyomo_solution = solve_ising_pyomo(B_matrix, B_bias, ising_obj = 'min')
        print("pyomo solution : ", pyomo_solution)
        print("pyomo energy : ", rs_ising_objective({'B_matrix':B_matrix, 'B_bias':B_bias}, pyomo_solution))
    except (KeyboardInterrupt, SystemExit):
        raise
    except: 
        # not really all that important
        print("pyomo failed, ignoring")

    return qaoa_solution


def solve_ising_pyomo(sub_B_matrix, bias, ising_obj = 'min'):
    constant = 0
    ising_to_file(sub_B_matrix, bias, ising_obj)
    instance = minimize_ising_model.model.create_instance("ising.dat")
    solver = SolverFactory("gurobi")
    solver.options['mipgap'] = 0.00000001
    solver.options['parallel'] = -1
    solver.options['timelimit'] = state_watcher.args.pyomo_timelimit
    results = solver.solve(instance, tee=False)
    energy = instance.min_ising()
    #print('energy', energy)
    # Get partition
    varobject = getattr(instance, 'x')
    part0 = []
    part1 = []
    n = sub_B_matrix.shape[0]
    #ising_partition = ['unset' for i in range(n)]
    solution_dict = {}
    for index in sorted(varobject):
        x_val = varobject[index].value
        s_val = 2 * x_val - 1
        if s_val < 0:
            s_val = -1
        else:
            s_val = 1
        assert s_val != 0
        #ising_partition[index] = s_val
        solution_dict[index] = s_val
    return solution_dict


def ising_to_file(B_matrix, bias, ising_obj):
    data_var = {}
    data_var['couplers'] = 'set couplers :=\n'
    data_var['nodes'] = 'set nodes :=\n'
    data_var['bias'] = 'param bias := \n'
    data_var['weight'] = 'param w := \n'
    mygraphfile = open('ising.dat', 'w')
    n = B_matrix.shape[0]
    # Take negative values because we max modularity
    if ising_obj == 'max':
        B_matrix = -B_matrix
        bias = -bias
    #print(bias)
    bias = bias.reshape(1, B_matrix.shape[0])
    bias = list(bias)[0]
    for i in range(n - 1):
        for j in range(i, n):
            w = B_matrix.item((i, j))
            data_var['couplers'] += ' '.join([str(i), str(j), '\n'])
            data_var['weight'] += ' '.join([str(i), str(j), str(w), '\n'])
    i, j = n - 1, n - 1
    w = B_matrix.item((i, j))
    data_var['couplers'] += ' '.join([str(i), str(j), '\n'])
    data_var['weight'] += ' '.join([str(i), str(j), str(w), '\n'])
    for i in range(n):
        data_var['nodes'] += str(i) + '\n'
        data_var['bias'] += str(i) + ' ' + str(bias[i]) + '\n'
    data_var['nodes'] += ';\n'
    data_var['bias'] += ';\n'
    data_var['weight'] += ';\n'
    data_var['couplers'] += ';\n'
    for item in data_var:
        mygraphfile.write(data_var[item])

def get_coarsest_level(all_graphs, hardware_size):
    coarsest_level = 0
    finest_level = len(all_graphs) - 1
    if nx.number_of_nodes(all_graphs[finest_level]) > hardware_size:
        return finest_level
    for level in all_graphs:
        if nx.number_of_nodes(all_graphs[level]) <= hardware_size:
            coarsest_level = level
            break
    return coarsest_level


def project_solution(uncoarse_soln,
                    coarse_graph,
                    fine_graph,
                    uncoarsen_map,
                    coarsen_algorithm = 'Matching'):
    """Project coarse solution to next fine graph"""

    fine_soln = dict.fromkeys(fine_graph.nodes(), 'unset')
    if not coarsen_algorithm == 'Matching':
        for c_node in sorted(coarse_graph.nodes()):
            fine_soln[uncoarsen_map[c_node]] = uncoarse_soln[c_node]
        unset_nodes = []
        part0 = []
        part1 = []
        for node in fine_graph:
            if fine_soln[node] == 'unset':
                unset_nodes.append(node)
            elif fine_soln[node] == '0':
                part0.append(node)
            elif fine_soln[node] == '1':
                part1.append(node)
            else:
                raise ValueError("must be 0, 1, or unset")
        fine_soln = assign_unset_nodes_ratio(fine_graph,
                                                fine_soln,
                                                unset_nodes,
                                                part0,
                                                part1)
    else:
        for node in sorted(coarse_graph.nodes()):
            for i in uncoarsen_map[node]:
                fine_soln[i] = uncoarse_soln[node]
    
    assert 'unset' not in [fine_soln[node] for node in fine_soln]

    return fine_soln


def get_random_boundary_nodes(graph, ptn_variables, hardware_size):

    boundary_nodes = set()
    for u, v in graph.edges():
        if ptn_variables[u] != ptn_variables[v]:
            boundary_nodes.add(u)
            boundary_nodes.add(v)
    return random.sample(boundary_nodes, hardware_size)
        
    
    
def get_free_nodes(graph,
                    ptn_variables,
                    alpha,
                    beta,
                    hardware_size,
                    problem_type,  # GP or modularity
                    refine_method,
                    ptn_properties):
    assert 'free' not in ptn_variables.values()
    if refine_method == 'boundary':
        print(' \n \t \t ATTN: BOUNDARY NODE SELECTION \n \n')
        nodes = get_random_boundary_nodes(graph, ptn_variables, hardware_size)
        nodes.sort()
        #for node in nodes:
        #    ptn_variables[node] = 'free'
    elif refine_method == 'top_gain':
        nodes = get_top_gain_nodes(graph,
                                    ptn_variables,
                                    alpha,
                                    beta,
                                    hardware_size,
                                    problem_type,
                                    ptn_properties)
    elif refine_method == 'top_gain_pairs':
        nodes = get_top_gain_nodes_PAIRS(graph,
                                        ptn_variables,
                                        alpha,
                                        beta,
                                        hardware_size,
                                        problem_type,
                                        ptn_properties)
    elif refine_method == 'top_gain_pair_and_single':
        nodes = get_top_gain_nodes_PAIRS_Singles(graph,
                                                ptn_variables,
                                                alpha,
                                                beta,
                                                hardware_size,
                                                problem_type,
                                                ptn_properties)
    else:
        raise ValueError('unknown refine method')
    nodes.sort()  # code depends on sorted nodes
    return nodes


def get_top_gain_nodes(graph, ptn_variables, alpha,beta, hardware_size,
                        problem_type, ptn_properties):
    if problem_type == 'GP':
        return get_GP_top_gain(graph, ptn_variables, alpha,beta, hardware_size,
                                                                 ptn_properties)
    elif problem_type == 'modularity':
        return get_modularity_top_gain(graph, ptn_variables, hardware_size,
                                                                 ptn_properties)
    else:
        raise ValueError('Only implemented for GP and Modularity')


def get_top_gain_nodes_PAIRS(graph, ptn_variables, alpha,beta, hardware_size,
                        problem_type, ptn_properties):
    if problem_type == 'GP':
        return get_GP_top_gain_PAIRS(graph, ptn_variables, alpha,beta, hardware_size,
                                                                 ptn_properties)
    elif problem_type == 'modularity':
        return get_modularity_top_gain_PAIRS(graph, ptn_variables, hardware_size,
                                                                 ptn_properties)
    else:
        raise ValueError('Only implemented for GP and Modularity')

    
def get_GP_top_gain_PAIRS(graph, ptn_variables, alpha,beta, hardware_size,
                            ptn_properties):                     
    '''return pair of nodes such that swaping them reduces energy the most'''
    
    swap_method = 'use_cut'  # gain is computed based on ext_deg - int_deg
    #swap_method = 'use_energy'  # gain is computed based on cut and balance
    if swap_method == 'use_cut':
        all_gain_pairs = []
        top_nodes = set()
        for u, v in graph.edges():
            if ptn_variables[u] != ptn_variables[v]:
                gain = cut_reduction(u, v, graph, ptn_variables)
                all_gain_pairs.append((gain, u, v))
            
        all_gain_pairs.sort(reverse = True)
        for _, u , v in all_gain_pairs:
            if u not in top_nodes and v not in top_nodes:
                if len(top_nodes) <= hardware_size - 2:
                    top_nodes.add(u)
                    top_nodes.add(v)
                else:
                    break
        return sorted(list(top_nodes))
        
        
def get_top_gain_nodes_PAIRS_Singles(graph,
                                    ptn_variables,
                                    alpha,
                                    beta,
                                    hardware_size,
                                    problem_type,
                                    ptn_properties):
                                    
    if problem_type == 'GP':
        return get_GP_top_gain_PAIRS_n_singles(graph,
                                                ptn_variables, 
                                                alpha,beta,
                                                hardware_size,
                                                ptn_properties)
    elif problem_type == 'modularity':
        return get_modularity_top_gain_PAIRS_n_singles(graph,
                                                       ptn_variables,
                                                       hardware_size,
                                                       ptn_properties)
    else:
        raise ValueError('Only implemented for GP and Modularity')



def get_GP_top_gain_PAIRS_n_singles(graph,
                                    ptn_variables, 
                                    alpha,beta,
                                    hardware_size,
                                    ptn_properties):
    swap_method = 'use_cut'
    #swap_method = 'use_energy' 
    all_gain_pairs = []
    top_nodes = set()

    for u, v in graph.edges():
        if ptn_variables[u] != ptn_variables[v]:
            gain = cut_reduction(u, v, graph, ptn_variables)
            all_gain_pairs.append((gain, u, v))
    for u in graph.nodes():
        gain = D_value(u, graph, ptn_variables)
        all_gain_pairs.append((gain, u, u))
    all_gain_pairs.sort(reverse = True)
    for _, u , v in all_gain_pairs:
        if len(top_nodes) < hardware_size:
            top_nodes.add(u)
            if len(top_nodes) < hardware_size and v != u:
                top_nodes.add(v)
        else:
            break
                
        
    return sorted(list(top_nodes))
                
    


def get_modularity_top_gain_PAIRS_n_singles(graph,
                                            ptn_variables,
                                            hardware_size,
                                            ptn_properties):
    pass


def D_value(node, graph, ptn_variables):
    ''' gain in cut from moving from one part to another'''
    ext_deg = 0
    int_deg = 0
    mypart = ptn_variables[node]
    for v in graph.neighbors(node):
        v_part = ptn_variables[v]
        assert v_part == -1 or v_part == 1  # check that all nodes assigned part
        if mypart == v_part:
            int_deg += graph[node][v]['weight']
        else:
            ext_deg += graph[node][v]['weight']
    return ext_deg - int_deg


def cut_reduction(node_a, node_b, graph, ptn_variables):
    if ptn_variables[node_a] == ptn_variables[node_b]:
        return 0
    else:
        D_a = D_value(node_a, graph, ptn_variables)
        D_b = D_value(node_b, graph, ptn_variables)    
        if graph.has_edge(node_a, node_b):
            c_ab = graph[node_a][node_b]['weight']
        else:
            c_ab = 0
        return D_a + D_b - 2*c_ab
                            
def get_modularity_top_gain_PAIRS(graph, ptn_variables, hardware_size,
                                  ptn_properties):
    pass


def get_modularity_top_gain(graph, ptn_variables, hardware_size, ptn_properties):              
    node_gain = []
    for node in range(nx.number_of_nodes(graph)):
        noise = 0
        gain = _node_gain_modularity(graph, node, ptn_variables,
                                        ptn_properties) + noise
        node_gain.append((gain, node))
        #gain2 = verify_gain(node, graph, mod_matrix, ptn)
    sort_gain = sorted(node_gain, reverse=True)
    nodes = [node for _, node in sort_gain]
    return nodes[:hardware_size]


def get_GP_top_gain(graph, ptn_variables, alpha,beta, hardware_size,
                    ptn_properties):
                    
    node_gain = []
    for node in range(nx.number_of_nodes(graph)):
        noise = 0
        gain = _node_gain_GP(graph, node, alpha, beta,
                                     ptn_variables, ptn_properties) + noise
        node_gain.append((gain, node))
        #gain2 = verify_gain(node, graph, mod_matrix, ptn)
    sort_gain = sorted(node_gain, reverse=True)
    nodes = [node for _, node in sort_gain]
    return nodes[:hardware_size]


def _node_gain_GP(graph, node, alpha, beta, ptn_variables, ptn_properties):
    v_i = graph.node[node]['volume']
    current_part = ptn_variables[node]
    vol_current_part = ptn_properties[current_part]['sum_of_vols']
    if current_part == -1:
        next_part = 1
    elif current_part == 1:
        next_part = -1
    else:
        raise ValueError('part must be either -1 or 1')
         
    vol_next_part = ptn_properties[next_part]['sum_of_vols']
    int_degree = ext_degree = 0
    for i in graph.neighbors(node):
        if ptn_variables[i] == current_part:
            int_degree += graph[node][i]['weight']
        else:
            ext_degree += graph[node][i]['weight']
    # from formula in multilevel paper
    gain = 2*alpha*v_i*(vol_current_part - v_i - vol_next_part)
    gain -= 2* beta * (int_degree - ext_degree)
    return gain


def _node_gain_modularity(graph, node, ptn_variables, ptn_properties):
    k_i = graph.degree(node, weight = 'weight')
    current_part = ptn_variables[node]
    weight_current_part = ptn_properties[current_part]['sum_of_weigh_degrees']
    if current_part == -1:
        next_part = 1
    elif current_part == 1:
        next_part = -1
    else:
        raise ValueError('part must be either -1 or 1')
         
    weight_next_part = ptn_properties[next_part]['sum_of_weigh_degrees']
    int_degree = ext_degree = 0
    for i in graph.neighbors(node):
        if ptn_variables[i] == current_part:
            int_degree += graph[node][i]['weight']
        else:
            ext_degree += graph[node][i]['weight']
    # from formula in multilevel paper
    alpha  = 0.5/(weight_current_part + weight_next_part)  # 1/|E|
    beta = 1
    gain = 2*alpha*k_i*(weight_current_part - k_i - weight_next_part)
    gain -= 2* beta * (int_degree - ext_degree)
    return gain



def formulate_and_solve_GP_ising(graph,
                                 alpha,
                                 beta,
                                 ptn_variables,
                                 solver,
                                 ising_obj):
    B_matrix, bias = formulate_GP_ising(graph,
                                        alpha,
                                        beta,
                                        ptn_variables)
    solution_dict = solve_ising(B_matrix,
                                bias,
                                ising_obj = 'min',
                                solver = solver)
    return solution_dict           

 
def formulate_and_solve_MODULARITY_ising(graph,
                                 ptn_variables,
                                 solver,
                                 ising_obj):
    B_matrix, bias = formulate_modularity_ising(graph, ptn_variables)
    solution_dict = solve_ising(B_matrix,
                                bias,
                                ising_obj = 'min',
                                solver = solver)
    return solution_dict  

                               
def refine_solution(graph,
                     ptn_variables,
                     solver,
                     refine_method,
                     hardware_size,
                     ising_obj,
                     ptn_properties,
                     problem_type,
                     objective,
                     alpha = 1,
                     beta = 1):
    converge_criterion = 3
    stop = 0
    prev_energy = float('inf')
    for myiter in range(100):

        solution_before_refinement = copy.deepcopy(ptn_variables)
        free_nodes_list = get_free_nodes(graph,
                                        ptn_variables,
                                        alpha,
                                        beta,
                                        hardware_size,
                                        problem_type,  # GP or modularity
                                        refine_method,
                                        ptn_properties)


        for i in free_nodes_list:
            ptn_variables[i] = 'free'
        if problem_type == 'GP':
            solution_dict  = formulate_and_solve_GP_ising(graph,
                                                             alpha,
                                                             beta,
                                                             ptn_variables,
                                                             solver,
                                                             ising_obj)
        elif problem_type == 'modularity':
            solution_dict  = formulate_and_solve_MODULARITY_ising(graph,
                                                             ptn_variables,
                                                             solver,
                                                             ising_obj)

        ptn = [solution_dict[_] for _ in sorted(solution_dict)]
        # ptn[node] = assigned_part 
        # ptn = [0, 1, 0, 1, ...,]
        # free_nodes_list = [0, 1, 10, 11, 15, ..., ], sorted free nodes
        for i in range(len(ptn)):
            ptn_variables[free_nodes_list[i]] = ptn[i]
        solution_after_refinement = copy.deepcopy(ptn_variables)
        ptn_properties = compute_ptn_properties(graph, ptn_variables)
        energy = compute_energy(graph, alpha, beta, ptn_properties, ptn_variables, objective)
        obj = compute_objective(graph, ptn_properties, ptn_variables, objective)
        part0 = ptn_properties[-1]['sum_of_vols']
        part1 = ptn_properties[1]['sum_of_vols']
        
        imbalance01, imbalance10 = compute_imbalances(part0, part1)
        print('gp energy', energy, imbalance01, imbalance10, obj)
        state_watcher.record_refinement_step(obj, imbalance01, imbalance10, 
        energy, solution_before_refinement, copy.deepcopy(free_nodes_list), solution_after_refinement)

        if energy < prev_energy:
            prev_energy = energy
        else:
            stop += 1
        if stop >= converge_criterion:
            print("\t Refine converge \n")
            break

    return ptn_variables


def compute_balance_ratio(graph, ptn_variables):
    part0 = part1 = 0
    for node in graph.nodes():
        if ptn_variables[node] == 0 or ptn_variables[node] == -1:
            part0 += graph.node[node]['volume']
        elif ptn_variables[node] == 1:
            part1 += graph.node[node]['volume']
        else:
            raise ValueError('Each node must be assigned a part')
    #part0_vol = 0.0 + sum([graph.node[node]['volume'] for node in part0])
    #part1_vol = sum([graph.node[node]['volume'] for node in part1])
    #print '\t \t vol', part0_vol, part1_vol
    return round(part0/part1,2)


def compute_ptn_properties(graph, ptn_variables):
    ptn_properties = {-1: {'sum_of_vols':0.0, 'sum_of_weigh_degrees':0.0},
                       1: {'sum_of_vols':0.0, 'sum_of_weigh_degrees':0.0},
                       'edge_cut':0,
                       'total_edges':0}
    for node in graph.nodes():
        part = ptn_variables[node]
        ptn_properties[part]['sum_of_vols'] += graph.node[node]['volume']
        ptn_properties[part]['sum_of_weigh_degrees'] += graph.degree(node,
                                                              weight = 'weight')
    for u, v in graph.edges():
        ptn_properties['total_edges'] += graph[u][v]['weight']
        if ptn_variables[u] != ptn_variables[v]:
            ptn_properties['edge_cut'] += graph[u][v]['weight']
    return ptn_properties


def compute_energy(graph, alpha, beta, ptn_properties, ptn_variables, objective):
    if objective == "GP":
        vol_1 = ptn_properties[-1]['sum_of_vols']
        vol_2 = ptn_properties[1]['sum_of_vols']
        E = ptn_properties['total_edges']
        cut = ptn_properties['edge_cut']
        return alpha * (vol_1 - vol_2)**2 - 2 * beta * (E - 2*cut)
    elif objective == "modularity":
        return compute_objective(graph, ptn_properties, ptn_variables, objective)
    else:
        raise ValueError("Unknown objective: {}".format(objective))

def compute_objective(graph, ptn_properties, ptn_variables, objective):
    if objective == "GP":
        cut = 0
        for u, v in graph.edges():
            if ptn_variables[u] != ptn_variables[v]:
                cut += graph[u][v]['weight']
        return cut
    elif objective == "modularity":
        deg_1 = ptn_properties[-1]['sum_of_weigh_degrees']
        deg_2 = ptn_properties[1]['sum_of_weigh_degrees']
        E = ptn_properties['total_edges']
        cut = ptn_properties['edge_cut']
        alpha = 0.5/E
        scale = 0.25/E
        return scale * (alpha * (deg_1 - deg_2)**2 - 2 * (E - 2*cut))
    else:
        raise ValueError("Unknown objective: {}".format(objective))

def assemble_outpath():
    if state_watcher.args.label:
        state_watcher.label = state_watcher.args.label
    else:
        import time
        state_watcher.label = time.strftime("%Y%m%d-%H%M%S")

    problem_name = '{}_{}_{}_seed_{}_hwsize_{}_refine_{}_niterqaoa_{}_a_{}_b_{}_backend_{}.p'.format(state_watcher.label,
            state_watcher.args.problem_type,
            Path(state_watcher.args.fpath).stem,
            state_watcher.args.seed, 
            state_watcher.args.hardware_size, 
            state_watcher.args.refine_method, 
            state_watcher.args.niter_qaoa,
            state_watcher.args.alpha,
            state_watcher.args.beta,
            state_watcher.args.qaoa_backend)
    outpath = Path(results_dir, state_watcher.args.solver, problem_name)
    if outpath.exists():
        print("Found pickle at {}.\nOur job here is done, exiting".format(outpath))
        sys.exit(0)
    return outpath

def coarsen_solve_coarse(seed, ising_obj):
    # Delete previous coarsened graph data
    os.system('./clean_data.sh')
    graphname = state_watcher.args.fpath.split('/')
    graphname = graphname[-1]
    state_watcher.graphname = graphname

    # coarsen graph
    state_watcher.all_graphs, state_watcher.up_maps, \
        state_watcher.lowest_level, state_watcher.graph_file = coarsen_graph_with_kaffpa(seed)

    coarsest_level = get_coarsest_level(state_watcher.all_graphs, state_watcher.args.hardware_size)

    # initial solution
    state_watcher.ptn_variables = {}
    for node in state_watcher.all_graphs[coarsest_level].nodes():
        state_watcher.ptn_variables[node] = 'free'
    if nx.number_of_nodes(state_watcher.all_graphs[coarsest_level]) > state_watcher.args.hardware_size:

        # Get random initial solution

        for node in state_watcher.all_graphs[coarsest_level].nodes():
            state_watcher.ptn_variables[node] = random.choice([-1,+1])
        ptn_properties = compute_ptn_properties(state_watcher.all_graphs[coarsest_level], state_watcher.ptn_variables)

        # Run QLS

        part0 = ptn_properties[-1]['sum_of_vols']
        part1 = ptn_properties[1]['sum_of_vols']

        # refine
        state_watcher.ptn_variables = refine_solution(state_watcher.all_graphs[coarsest_level],
                                         state_watcher.ptn_variables,
                                         state_watcher.args.solver,
                                         state_watcher.args.refine_method,
                                         state_watcher.args.hardware_size,
                                         ising_obj,
                                         ptn_properties,
                                         state_watcher.args.problem_type,
                                         state_watcher.args.objective,
                                         state_watcher.args.alpha,
                                         state_watcher.args.beta)

    else:
        if state_watcher.args.problem_type == 'GP':
            state_watcher.ptn_variables  = formulate_and_solve_GP_ising(state_watcher.all_graphs[coarsest_level],
                                                             state_watcher.args.alpha,
                                                             state_watcher.args.beta,
                                                             state_watcher.ptn_variables,
                                                             state_watcher.args.solver,
                                                             ising_obj)
        elif state_watcher.args.problem_type == 'modularity':
            state_watcher.ptn_variables  = formulate_and_solve_MODULARITY_ising(state_watcher.all_graphs[coarsest_level],
                                                             state_watcher.ptn_variables,
                                                             state_watcher.args.solver,
                                                             ising_obj)
    ptn_properties = compute_ptn_properties(state_watcher.all_graphs[coarsest_level], state_watcher.ptn_variables)
    obj = compute_objective(state_watcher.all_graphs[coarsest_level], ptn_properties, state_watcher.ptn_variables, state_watcher.args.objective)
    print('coarsest objective (cut or modularity)', obj)
    part0 = ptn_properties[-1]['sum_of_vols']
    part1 = ptn_properties[1]['sum_of_vols']
    imbalance01, imbalance10 = compute_imbalances(part0, part1)
    print('coarsest vols ratio', imbalance01)

    state_watcher.current_level = coarsest_level
    state_watcher.coarsest_level = coarsest_level
    state_watcher.record_refinement_step(obj, imbalance01, imbalance10, None, None, None, copy.deepcopy(state_watcher.ptn_variables))
    if state_watcher.args.save:
        try:
            if not state_watcher.args.no_checkpoint:
                save_checkpoint()
        except AttributeError:
            # old checkpoint that doesn't have no_checkpoint in the namespace
            save_checkpoint()



def project_and_refine(ising_obj, current_level):
    # project solution and refine
    for _i in range(current_level, state_watcher.coarsest_level + 1):
        state_watcher.current_level = state_watcher.coarsest_level - _i
        print('\n-----level', state_watcher.current_level)
        state_watcher.ptn_variables = project_solution(state_watcher.ptn_variables,
                                        state_watcher.all_graphs[state_watcher.current_level + 1], # prev state_watcher.current_level
                                        state_watcher.all_graphs[state_watcher.current_level],
                                        state_watcher.up_maps[state_watcher.current_level + 1],
                                        coarsen_algorithm = 'Matching')
        ptn_properties = compute_ptn_properties(state_watcher.all_graphs[state_watcher.current_level], state_watcher.ptn_variables)
        projected_obj = compute_objective(state_watcher.all_graphs[state_watcher.current_level], ptn_properties, state_watcher.ptn_variables, state_watcher.args.objective)
        print('projected obj', projected_obj)
        part0 = ptn_properties[-1]['sum_of_vols']
        part1 = ptn_properties[1]['sum_of_vols']

        # refine
        state_watcher.ptn_variables = refine_solution(state_watcher.all_graphs[state_watcher.current_level],
                                         state_watcher.ptn_variables,
                                         state_watcher.args.solver,
                                         state_watcher.args.refine_method,
                                         state_watcher.args.hardware_size,
                                         ising_obj,
                                         ptn_properties,
                                         state_watcher.args.problem_type,
                                         state_watcher.args.objective,
                                         state_watcher.args.alpha,
                                         state_watcher.args.beta)
        ptn_properties = compute_ptn_properties(state_watcher.all_graphs[state_watcher.current_level], state_watcher.ptn_variables)
        obj = compute_objective(state_watcher.all_graphs[state_watcher.current_level], ptn_properties, state_watcher.ptn_variables, state_watcher.args.objective)
        print('obj', obj)
        part0 = ptn_properties[-1]['sum_of_vols']
        part1 = ptn_properties[1]['sum_of_vols']
        imbalance01, imbalance10 = compute_imbalances(part0, part1)
        print('vols ratio', imbalance01)
        print('NUMBER_SOLVER_CALLS:', state_watcher.num_solver_calls)  
        #   
        state_watcher.obj_value = obj
        state_watcher.imbalance01 =  imbalance01
        state_watcher.imbalance10 =  imbalance10            
        if state_watcher.args.save:
            try:
                if not state_watcher.args.no_checkpoint:
                    save_checkpoint()
            except AttributeError:
                # old checkpoint that doesn't have no_checkpoint in the namespace
                save_checkpoint()

def experiment_to_file(seed):
    # Deprecated! Please use the pickles produced by state_watcher! 
    obj, num, ratio, graphname, hardware_size = main(seed)
    #print obj, num, ratio, graphname
    expfile = 'myexp.txt'
    if os.path.exists(expfile):
        myfile = open(expfile, 'a')
    else:
        myfile = open(expfile, 'w')
    out = [graphname, obj, ratio, num, hardware_size]
    out = [str(i) for i in out]
    out = " ".join(out)
    out += '\n'
    myfile.write(out)


def deg_volume_graph(graph):
    for node in graph.nodes():
        vol = nx.degree(graph, node, weight='weight')
        if vol > 0:  # isolated nodes do not affect modularity so ignore them
            graph.node[node]['volume'] = vol
        else:
            graph.remove_node(node)  # remove isolated nodes
    graph = nx.convert_node_labels_to_integers(graph, first_label=0)
    return graph

def write_weighted_graph_to_metis_file(graph):
    if list(range(nx.number_of_nodes(graph))) == sorted(graph.nodes()):
        start_node_index = 0
    elif list(range(1, nx.number_of_nodes(graph) + 1)) == sorted(graph.nodes()):
        start_node_index = 1
    else:
        raise ValueError('Inconsistent node indexing')
    gfile = open("deg_volume.graph", "w")
    n = str(nx.nx.number_of_nodes(graph))
    m = str(nx.nx.number_of_edges(graph))
    out = " ".join([n, m, "11", "\n"])
    gfile.write(out)
    for u in sorted(graph.nodes()):
        neighbors = [i for i in nx.neighbors(graph,u)]
        volume = graph.node[u]['volume']
        out = str(volume)+" "
        for v in neighbors:
            weight = graph[u][v]['weight']
            out += str(v+ (1-start_node_index) )+" "+str(weight)+" "  # v+1 because metis starts from 1
        out += "\n"
        gfile.write(out)
    gfile.close() 


def create_deg_volume_graph_file():
    graph_file = state_watcher.args.fpath
    graph = read_unweighted_metis_graph(graph_file)
    graph = deg_volume_graph(graph)
    write_weighted_graph_to_metis_file(graph)
    graph_file = "deg_volume.graph"
    return graph_file


if __name__ == '__main__':
    # solution for parsing both command line parameters and config file proudly copied from https://stackoverflow.com/questions/3609852/which-is-the-best-way-to-allow-configuration-options-be-overridden-at-the-comman 
    # adding fake header to INI file: https://stackoverflow.com/questions/2885190/using-pythons-configparser-to-read-a-file-without-section-name

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    # Note: for defaults to work correctly, need to put them here!
    defaults = { 
            "niter_qaoa" : 100,
            "pyomo_timelimit" : 1000,
            "alpha" : 1,
            "beta" : 1,
            "solver" : "pyomo",
            "qaoa_backend" : "qasm_simulator",
            "problem_type" : "GP",
            "seed" : 42,
            "hardware_size" : 20,
            "refine_method" : "top_gain_pair_and_single"
        }

    if args.conf_file:
        config = configparser.SafeConfigParser()
        try:
            config.read([args.conf_file])
        except configparser.MissingSectionHeaderError:
            with open(args.conf_file) as f:
                config.read_string("[Defaults]\n" + f.read())
        defaults.update(dict(config.items("Defaults")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
        )
    parser.set_defaults(**defaults)

    parser.add_argument(
        "fpath", type=str, nargs='?', help="path to graph file")
    parser.add_argument(
        "--niter-qaoa", type=int, help="number of qaoa training iterations")
    parser.add_argument(
        "--pyomo-timelimit", type=int, help="timelimit for pyomo (seconds)")
    parser.add_argument(
        "--alpha", type=float, help="pseudo-lagrangian parameter for balance")
    parser.add_argument(
        "--beta", type=float, help="pseudo-lagrangian parameter for cut")
    parser.add_argument(
        "--solver",
        type=str,
        choices=[
            "qbsolv", "pyomo",
            "qaoa", "dwave"
        ],
        help="optimization method")
    parser.add_argument(
        "--qaoa-backend",
        type=str,
        default="qasm_simulator",
        help="optimization method")
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=[
            "GP", "modularity"
        ],
        help="Problem being solved")
    parser.add_argument(
        "--objective",
        type=str,
        choices=[
            "GP", "modularity"
        ],
        help="Objective to be minimized/maximized. Defaults to problem_type")
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed")
    parser.add_argument(
        "--hardware-size",
        type=int,
        help="Hardware size (e.g. number of qubits). Limit the size of the subproblem")
    parser.add_argument(
        "--refine-method",
        type=str,
        choices=[
            "top_gain_pair_and_single", "top_gain_pairs",
            "top_gain", "boundary" 
        ],
        help="Problem being solved")
    save_params = parser.add_mutually_exclusive_group()
    save_params.add_argument(
        "--save", help="flag to save results", action="store_true")
    save_params.add_argument(
        "--save-hayato-format", help="flag to save results using Hayato`s format", action="store_true")
    parser.add_argument(
        "--no-checkpoint", help="do not save checkpoints", action="store_true")
    parser.add_argument(
        "--label",
        type=str,
        help=
        "description of this version of the script. The description is prepended to the filename,\
         so it should not contain any spaces. Default: time stamp"
    )
    parser.add_argument(
        "--verbose", help="sets logging level to INFO", action="store_true")
    parser.add_argument(
        "--restore", type=str, help="path to pickle of StateWatcher instance",
        default=None)
    parser.add_argument(
        "--coarsen_deg", help="if raised, coarsen degree volume graph",
        action="store_true")  
    args = parser.parse_args(remaining_argv)
    restore = args.restore is not None
    # Min or max problem. 
    # Modularity is formulated as a min problem here, so it's always min
    ising_obj = 'min'

    if restore:
        state_watcher = pickle.load(open(args.restore, 'rb'))
    else:
        state_watcher = StateWatcher()
        state_watcher.args = args
        del args
        state_watcher.outpath = assemble_outpath()
    # set seed
    seed = state_watcher.args.seed
    random.seed(seed)

    if ((state_watcher.args.problem_type == 'GP' and state_watcher.args.refine_method != 'top_gain_pair_and_single')
        or (state_watcher.args.problem_type == 'modularity' and state_watcher.args.refine_method != 'top_gain')):
            raise ValueError("Incorrect combination of problem_type: {} \
                              and refine_method: {}".format(state_watcher.args.problem_type, \
                              state_watcher.args.refine_method)
                            )
    try:
        if state_watcher.args.objective is None:
            state_watcher.args.objective = state_watcher.args.problem_type
    except AttributeError:
        # compatibility with old checkpoints
        state_watcher.args.objective = state_watcher.args.problem_type


    if state_watcher.args.solver == 'qaoa':
        from qcommunity.optimization.optimize import optimize_ising
    if state_watcher.args.solver == 'dwave':
        from dwave_sapi2.local import local_connection
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import solve_ising as sapi_solve_ising
        from dwave_sapi2.embedding import find_embedding
        from dwave_sapi2.util import get_chimera_adjacency, get_hardware_adjacency
        from dwave_sapi2.embedding import embed_problem, unembed_answer
        from chimera_embedding import processor
        from collections import Counter
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),'dwave/'))
        import dwave_params
        state_watcher.dwave_solver = dwave_params.start_sapi()
        state_watcher.dwave_embedding = dwave_params.get_native_embedding(state_watcher.dwave_solver)
        

    if state_watcher.args.verbose:
        logging.basicConfig(level=logging.INFO)

    if not restore:
        coarsen_solve_coarse(state_watcher.args.seed, ising_obj)

    current_level =  state_watcher.coarsest_level - state_watcher.current_level + 1
    project_and_refine(ising_obj, current_level)
    
    state_watcher.print_records()
    if state_watcher.args.save:
        state_watcher.save_final_res_to_disk(state_watcher.outpath)
    elif state_watcher.args.save_hayato_format:
        state_watcher.save_final_res_to_disk_hayato(state_watcher.outpath)


