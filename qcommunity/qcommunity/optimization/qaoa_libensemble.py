#!/usr/bin/env python
# Libensemble for parameter optimization

from __future__ import division
from __future__ import absolute_import

from qcommunity.optimization.run_with_angles import run_angles

from mpi4py import MPI  # for libE communicator
import sys, os  # for adding to path
import numpy as np
import socket

#sys.path.append('/home/rshaydu/quantum/libensemble/libensemble')
#sys.path.append('/home/rshaydu/quantum/libensemble/libensemble/gen_funcs')

from libensemble.libE import libE
from libensemble.gen_funcs.aposmm import aposmm_logic
from libensemble.alloc_funcs.fast_alloc_to_aposmm import give_sim_work_first as alloc_f


def optimize_obj(obj_val, num_parameters, params=None):

    print("Using localopt_method: {}".format(
        params['localopt_method']))

    def sim_func(H, gen_info, sim_specs, libE_info):
        del libE_info  # Ignored parameter

        batch = len(H['x'])
        O = np.zeros(batch, dtype=sim_specs['out'])

        for i, x in enumerate(H['x']):
            O['f'][i] = obj_val(x)

        print(O)
        sys.stdout.flush()
        return O, gen_info

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    #State the objective function, its arguments, output, and necessary parameters (and their sizes)
    sim_specs = {
        'sim_f':
            sim_func,  # This is the function whose output is being minimized
        'in': ['x'],  # These keys will be given to the above function
        'out': [
            ('f',
             float),  # This is the output from the function being minimized
        ],
    }
    gen_out = [
        ('x', float, num_parameters),
        ('x_on_cube', float, num_parameters),
        ('sim_id', int),
        ('priority', float),
        ('local_pt', bool),
        ('known_to_aposmm',
         bool),  # Mark known points so fewer updates are needed.
        ('dist_to_unit_bounds', float),
        ('dist_to_better_l', float),
        ('dist_to_better_s', float),
        ('ind_of_better_l', int),
        ('ind_of_better_s', int),
        ('started_run', bool),
        ('num_active_runs', int),  # Number of active runs point is involved in
        ('local_min', bool),
        ('paused', bool),
    ]

    if params['ansatz'] == 'QAOA':
        lb = np.array([0, 0] * params['ansatz_depth'])
        ub = np.array([np.pi / 2] * params['ansatz_depth'] + [2*np.pi] * params['ansatz_depth'])
    elif params['ansatz'] == 'RYRZ':
        lb = np.array([-np.pi] * num_parameters)
        ub = np.array([np.pi] * num_parameters)

    # State the generating function, its arguments, output, and necessary parameters.
    gen_specs = {
        'gen_f': aposmm_logic,
        'in': [o[0] for o in gen_out] + ['f', 'returned'],
        #'mu':0.1,   # limit on dist_to_bound: everything closer to bound than mu is thrown out
        'out': gen_out,
        'lb': lb,
        'ub': ub,
        'initial_sample_size': params[
            'init_points'],  # num points sampled before starting opt runs
        'localopt_method': params['localopt_method'],
        'xtol_rel': params['xtol_rel'], # TODO: give these two to local methods
        'ftol_rel': params['ftol_rel'],
        'min_batch_size': 1,
        'batch_mode': True,
        'num_active_gens': 1,
        'high_priority_to_best_localopt_runs': True,
        'sample_points': params['sample_points'],
        'max_active_runs': params['max_active_runs'],
        'dist_to_bound_multiple':
            1.0,  # NEVER more than 1! (increase max_active_runs instead)
    }

    # Tell libEnsemble when to stop
    exit_criteria = {'sim_max': params['n_iter'] + params['init_points']}

    persis_info = {'next_to_give': 0}
    persis_info['total_gen_calls'] = 0
    persis_info['last_worker'] = 0
    persis_info[0] = {
        'active_runs': set(),
        'run_order': {},
        'old_runs': {},
        'total_runs': 0,
        'rand_stream': np.random.RandomState(1)
    }

    for i in range(1, MPI.COMM_WORLD.Get_size()):
        persis_info[i] = {'rand_stream': np.random.RandomState(i)}

    alloc_specs = {'out': [('allocated', bool)], 'alloc_f': alloc_f}

    H, persis_info, flag = libE(
        sim_specs,
        gen_specs,
        exit_criteria,
        persis_info=persis_info,
        alloc_specs=alloc_specs)
    if MPI.COMM_WORLD.Get_rank() == 0:
        return (H, persis_info)
