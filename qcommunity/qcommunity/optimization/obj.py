#!/usr/bin/env python

# Returns obj_val function to be used in an optimizer
# A better and updated version of qaoa_obj.py

import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
from networkx.generators.classic import barbell_graph
import copy
import sys
import warnings
import logging
from collections import namedtuple

import qcommunity.modularity.graphs as gm
from qcommunity.utils.import_graph import generate_graph
from ibmqxbackend.ansatz import IBMQXVarForm

def get_obj(problem_description=None,
            obj_params='ndarray',
            sign=1,
            backend='IBMQX',
            backend_params={'depth':3, 'var_form':'RYRZ', 'backend_device':None},
            statistic='mean',
            samples_per_eval=8000,
            var_form_obj=None,
            seed=None,
            return_x=0):
            # TODO: problem description as a dictionary with fields corresponding to params, like: problem_description = {'name':'modularity', 'B':B, 'C':C}
    """
    :param obj_params: defines the signature of obj_val function. 'beta gamma' or 'ndarray' (added to support arbitrary number of steps and scipy.optimize.minimize.) 

    :return: obj_val function, number of variational parameters
    :rtype: tuple
    """
    if return_x >= 1:
        all_x = []
        all_vals = []
    if return_x >= 2:
        all_samples_tuple = []
        all_samples_energies = []
    # TODO refactor, remove code duplication
    if backend == 'IBMQX':
        if 'simulator' in backend_params['backend_device']:
            target_backend_name = None
        else:
            target_backend_name=backend_params['backend_device']
        if var_form_obj is not None:
            var_form = var_form_obj
        else:
            var_form = IBMQXVarForm(
                problem_description, 
                backend_params=backend_params,
                target_backend_name=target_backend_name)
        num_parameters = var_form.num_parameters
        if obj_params == 'ndarray':

            def obj_val(x):
                if len(x) == 1:
                    # assuming it's Sami's weird array
                    x = x[0]
                if not all(np.isfinite(x)):
                    raise FloatingPointError("Received non finite x={}".format(x))
                resstrs = var_form.run(x, samples=samples_per_eval, seed=seed)
                energies = [
                    problem_description['objective_function'](problem_description, x) for x in resstrs
                ]
                y = {}
                y['mean'] = np.mean(energies)
                y['min'] = min(energies)
                y['max'] = max(energies)
                if return_x >= 1:
                    all_x.append(copy.deepcopy(x))
                    all_vals.append({'max': y['max'], 'mean': y['mean'], 'min': y['min']})
                if return_x >= 2:
                    all_samples_tuple.append(copy.deepcopy(resstrs))
                    all_samples_energies.append(copy.deepcopy(energies))
                logging.info("Actual energy max\t{:.4f}\tmean\t{:.4f}\tmin\t{:.4f}\tat x={}".format(y['max'], y['mean'], y['min'], x))
                return float(sign * y[statistic])  # cast to float to please nlopt, see: https://stackoverflow.com/questions/28564976/nlopt-invalid-argument-python 
        else:
            raise ValueError(
                "obj_params '{}' not compatible with backend '{}'".format(
                    obj_params, backend))
    else:
        raise ValueError("Unsupported backend: {}".format(backend))

    if return_x >= 2:
        return obj_val, num_parameters, all_x, all_vals, all_samples_tuple, all_samples_energies
    elif return_x >= 1:
        return obj_val, num_parameters, all_x, all_vals
    else:
        return obj_val, num_parameters


def get_obj_val(graph_generator_name,
                left,
                right,
                p=None, # just for erdos-renyi graph
                seed=None,
                obj_params='ndarray',
                sign=1,
                backend='IBMQX',
                backend_params={'depth':3, 'var_form':'RYRZ', 'backend_device':None},
                return_x=False,
                problem_name = 'modularity',
                statistic='mean', # statistic that objective function returns and outer-loop optimizer optimizes for
                weight=False,
                simulator_seed=None, # used to see Qiskit Aer
                remove_edge=False):
    # Generate the graph
    G, _ = generate_graph(graph_generator_name, left, right, seed=seed, weight=weight, remove_edge=remove_edge, p=p)
    if problem_name == 'modularity':
        B = nx.modularity_matrix(G, weight='weight').A
        problem_description = {'name': problem_name, 'B':B, 'n_nodes':G.number_of_nodes(), 'objective_function':gm.compute_modularity_dict}
    elif problem_name == 'maxcut':
        L = nx.laplacian_matrix(G, weight='weight').A
        A = nx.adjacency_matrix(G, weight='weight').A
        problem_description = {'name': problem_name, 'A':A, 'L':L, 'n_nodes':G.number_of_nodes(), 'objective_function':gm.compute_cut_dict}
    else:
        raise ValueError("Incorrect problem name: {}".format(problem_name))
    return get_obj(
        problem_description=problem_description,
        obj_params=obj_params,
        sign=sign,
        backend=backend,
        seed=simulator_seed,
        backend_params=backend_params,
        return_x=return_x)


if __name__ == "__main__":
    p = 2
    obj_val, num_parameters = get_obj_val(
            "get_connected_caveman_graph", 2, 6, seed=1, obj_params='ndarray', backend='IBMQX', backend_params={'backend_device': 'qasm_simulator', 'depth': p, 'var_form':'QAOA'}, sign=-1)
    y = np.random.uniform(-np.pi, np.pi, num_parameters)
    print(obj_val(y))
