#!/usr/bin/env python

# QAOA parameter optimization

# Example: mpirun -np 2 python -m mpi4py optimize.py -g get_connected_watts_strogatz_graph -l 12 -r 5 --method libensemble --mpi --backend IBMQX
# Example: ./optimize.py -g get_connected_watts_strogatz_graph -l 12 -r 5 --method neldermead
# Example: ./optimize.py -g get_connected_watts_strogatz_graph -l 12 -r 5 --method COBYLA --niter 100 --backend IBMQX

import pickle
import numpy as np
import os.path
import sys
import argparse
import warnings
import random
import logging
import nlopt
from operator import itemgetter
from SALib.sample import saltelli
from qcommunity.utils.ising_obj import ising_objective
import qcommunity.modularity.graphs as gm
from qcommunity.optimization.obj import get_obj_val, get_obj
from qcommunity.optimization.run_with_angles import run_angles, test_angles, run_and_get_best
from qcommunity.optimization.sample_points import get_fourier_init_points, get_past_best_sample_points, add_extra_points_uniform
import qcommunity.optimization.learning as ml
import qcommunity.optimization.cobyla_nlopt as cobyla_nlopt
import qcommunity.optimization.bobyqa_nlopt as bobyqa_nlopt
import qcommunity.optimization.newuoa_nlopt as newuoa_nlopt
import qcommunity.optimization.praxis_nlopt as praxis_nlopt
import qcommunity.optimization.sbplx_nlopt as sbplx_nlopt
import qcommunity.optimization.neldermead_nlopt as neldermead_nlopt
import qcommunity.optimization.mlsl_nlopt as mlsl_nlopt
import qcommunity.optimization.sami_bayes as sami_bayes
import qcommunity.optimization.sami_seqopt as sami_seqopt
import qcommunity.optimization.pass_through as pass_through


def optimize_modularity(n_nodes,
                        B,
                        C=None,
                        params=None,
                        method='COBYLA',
                        backend='IBMQX',
                        backend_params={
                            'backend_device': None,
                            'depth': 3
                        }):
    if method == 'COBYLA':
        problem_description = {'name': 'modularity', 'B':B, 'n_nodes':n_nodes, 'objective_function':gm.compute_modularity_dict, 'C':C}
        obj_val, num_parameters = get_obj(
            problem_description=problem_description,
            obj_params='ndarray',
            sign=-1,
            backend=backend,
            backend_params=backend_params)  # sign = -1 because COBYLA minimizes
        res = cobyla.optimize_obj(obj_val, num_parameters, params)
        optimized = run_angles(
            n_nodes,
            B,
            res.x,
            C=C,
            backend=backend,
            backend_params=backend_params)
    else:
        raise ValueError('Incorrect method: {}'.format(method))
    return optimized

def optimize_ising(problem_description=None,
                   backend='IBMQX',
                   backend_params={
                       'depth': 1,
                       'var_form':'RYRZ',
                       'backend_device': 'qasm_simulator',
                       'n_iter': 5000,
                       'opt_method': 'COBYLA_NLOPT'
                   },
                   var_form_obj=None,
                   seed=None):
    statistic = 'min' # statistic on qaoa sample to optimize for
    if problem_description['ising_obj'] == 'min':
        sign = 1
    elif problem_description['ising_obj'] == 'max':
        sign = -1
    else:
        raise ValueError("Unsupported ising_obj={}".format(ising_obj))

    if seed == None:
        import datetime, time
        # real hacky, ideally optimize_ising is always called with a seed
        seed = int(time.mktime(datetime.datetime.now().timetuple()))
    obj_val, num_parameters, all_x, all_vals = get_obj(
            problem_description=problem_description,
            sign=sign, # because we minimize ising_obj_function, which is exactly what the methods do
            backend=backend,
            backend_params=backend_params,
            statistic='mean', # train on mean, choose best based on statistic
            samples_per_eval=5000,
            var_form_obj=var_form_obj,
            return_x=1)
    # 1. Find optimal parameters (in simulator, with realistic device noise!)
    # Use nlopt SUBPLX, 10k iterations
    # or COBYLA? 
    n_iter = backend_params['n_iter']
    method = backend_params['opt_method']
    params = {
        'n_iter': n_iter,
        'n_iter_local': n_iter,
        'init_points': 0, # compatibility reasons
        'ansatz': backend_params['var_form'],
        'ansatz_depth': backend_params['depth'],
        'seed': seed,
        'xtol_rel': 1e-23, 
        'ftol_rel': 1e-22
    }
    if params['ansatz'] == 'RYRZ':
        params['sample_points'] = np.split(
            np.random.uniform(-np.pi + 0.25, np.pi - 0.25,
                              n_iter * num_parameters), n_iter)
    if params['ansatz'] == 'QAOA':
        ub = ( [np.pi / 2 - 0.01] * params['ansatz_depth'] + [2*np.pi - 0.01] * params['ansatz_depth'] ) * n_iter
        lb = [0.01]  * 2 * params['ansatz_depth'] * n_iter
        params['sample_points'] = np.split(np.random.uniform(lb,ub, 2 * params['ansatz_depth'] * n_iter), n_iter)
    # TODO progress bar for training? 
    while len(all_vals) < n_iter:
        print("Starting {}, current nevals {}".format(method, len(all_vals)))
        try:
            if method == 'COBYLA_NLOPT':
                res = cobyla_nlopt.optimize_obj(obj_val, num_parameters, params)
            elif method == 'BOBYQA_NLOPT':
                res = bobyqa_nlopt.optimize_obj(obj_val, num_parameters, params)
            elif method == 'NELDERMEAD_NLOPT':
                res = neldermead_nlopt.optimize_obj(obj_val, num_parameters, params)
            elif method == 'NEWUOA_NLOPT':
                res = newuoa_nlopt.optimize_obj(obj_val, num_parameters, params)
            elif method == 'PRAXIS_NLOPT':
                res = praxis_nlopt.optimize_obj(obj_val, num_parameters, params)
            elif method == 'SBPLX_NLOPT':
                res = sbplx_nlopt.optimize_obj(obj_val, num_parameters, params)
            else:
                raise ValueError('Incorrect method: {}'.format(args.method))
        except (nlopt.RoundoffLimited, FloatingPointError) as e:
            print("Encountered {}, recovering results at iter {}".format(e, len(all_vals)))
            res = None
        assert len(all_x) == len(all_vals)
        params['sample_points'] = params['sample_points'][1:]

    # 2. Run optimal parameters on actual device with 5k samples, pick best of those 5k samples
    min_index = np.argmin([x[statistic] for x in all_vals]) # this is pretty inefficient
    optimal_parameters = all_x[min_index]
    logging.info("Using min_index={}, corresponding to {} energy={}".format(min_index, statistic, all_vals[min_index][statistic]))
    best_energy, best_string = run_and_get_best(optimal_parameters, 
            problem_description=problem_description,
            objective=problem_description['ising_obj'],
            var_form_obj=var_form_obj,
            backend=backend,
            backend_params=backend_params)
    # 3. Convert to Hayato's solution_dict format  
    solution_dict = {}
    for i, val in enumerate(best_string):
        solution_dict[i] = 2*val - 1
    return solution_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "-p",
        type=float,
        help="probability of edge creation p (only for erdos-reniy)")
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
        "--niter-local", type=int, default=-1, help="number of iterations per local optimizer run (default: niter)")
    parser.add_argument(
        "--label",
        type=str,
        help=
        "description of this version of the script. The description is prepended to the filename, so it should not contain any spaces. Default: time stamp"
    )
    parser.add_argument(
        "-g",
        "--graph-generator",
        type=str,
        default="get_barbell_graph",
        help="graph generator function")
    parser.add_argument(
        "--method",
        type=str,
        default="BOBYQA_NLOPT",
        choices=[
            "libensemble", "COBYLA_NLOPT",
            "BOBYQA_NLOPT", "MLSL_NLOPT", "NELDERMEAD_NLOPT", "NEWUOA_NLOPT", 
            "PRAXIS_NLOPT", "SBPLX_NLOPT", "SAMI", 'SEQ_OPT', 
            "PASS_THROUGH"
        ],
        help="optimization method")
    parser.add_argument(
        "--localopt-method",
        type=str,
        default="LN_BOBYQA",
        choices=["LN_BOBYQA", "LN_COBYLA"],
        help="libensemble local optimization method")
    parser.add_argument(
        "--backend",
        type=str,
        default="IBMQX",
        choices=["IBMQX"],
        help="backend simulator to be used")
    parser.add_argument(
        "--problem",
        type=str,
        default="modularity",
        choices=["modularity", "maxcut"],
        help="the problem to be solved on the graph")
    parser.add_argument(
        "--ansatz-depth", type=int, default=1, help="variational ansatz depth")
    parser.add_argument(
        "--ansatz",
        type=str,
        default="RYRZ",
        choices=["RYRZ", "QAOA"],
        help="ansatz (variational form) to be used")
    init_points_group = parser.add_mutually_exclusive_group()
    init_points_group.add_argument(
        "--sample-points",
        type=str,
        help="path to the pickle with sample points (produced by get_optimal_sample_points)")
    init_points_group.add_argument(
        "--fourier",
        type=str,
        help="path to the pickle with best points for fourier (produced by approximation_ratio.py)")
    parser.add_argument(
        "--fourier-method",
        type=str,
        default="FOURIER",
        choices=["INTERP", "FOURIER"],
        help="ansatz (variational form) to be used")
    parser.add_argument(
        "--fourier-no-extra-points",
        action="store_true",
        help="if this flag is raised, not extra points are added to the ones produced by fourier")
    parser.add_argument(
        "--max-active-runs",
        type=int,
        default=10,
        help="maximal number of active runs in libensemble")
    parser.add_argument(
        "--noise", help="flag to add noise", action="store_true")
    parser.add_argument(
        "--weighted", help="if raised, the graph will be randomly weighted", action="store_true")
    parser.add_argument(
        "--save", help="flag to save results", action="store_true")
    parser.add_argument(
        "--mpi", help="note that optimize is run with mpi", action="store_true")
    parser.add_argument(
        "--verbose", help="sets logging level to INFO", action="store_true")
    parser.add_argument(
        "--restart-local", help="restarts local nlopt methods after convergence with a new initial point", 
        action="store_true")
    parser.add_argument(
        "--zero-tol", help="pass zero tolerance to local optimizers (only use for no restart comparison!)", 
        action="store_true")
    parser.add_argument(
        "--remove-edge", 
        help="remove random edge from the graph", 
        action="store",
        const=-1, # hack! '-1' means remove random edge
        default=None,
        nargs='?',
        type=int)
    parser.add_argument(
        "--savedir",
        type=str,
        default='/zfs/safrolab/users/rshaydu/quantum/data/for_jeff/',
        help="folder where to save the results")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        
    init_points = int(args.niter * 0.1)  # for backwards compatibility
    n_iter = args.niter - init_points
    if args.niter_local == -1:
        args.niter_local = args.niter
    if args.backend == "IBMQX":
        backend_params = {'backend_device': 'qasm_simulator', 'depth': args.ansatz_depth, 'var_form':args.ansatz}
    else:
        raise ValueError("Illegal backend: {}".format(args.backend))

    if args.graph_generator_seed is None:
        graph_generator_seed = args.seed
    else:
        graph_generator_seed = args.graph_generator_seed

    np.random.seed(args.seed)
    random.seed(args.seed)

    main_proc = True
    if args.mpi:
        if args.method == 'libensemble':
            import qcommunity.optimization.qaoa_libensemble as qaoa_libensemble
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() != 0:
            main_proc = False

    if args.label:
        label = args.label
    else:
        import time
        label = time.strftime("%Y%m%d-%H%M%S")

    if not args.save:
        warnings.warn("save flag not raised, the results will be thrown out!",
                      UserWarning)
    params = {
        'init_points': init_points,
        'n_iter': n_iter,
        'n_iter_local': args.niter_local,
        'localopt_method': args.localopt_method,
        'max_active_runs': args.max_active_runs,
        'ansatz': args.ansatz,
        'ansatz_depth': args.ansatz_depth,
        'seed': args.seed
    }
    
    if args.zero_tol:
        params['xtol_rel'] = 0 
        params['ftol_rel'] = 0 
    else:
        params['xtol_rel'] = 1e-3 
        params['ftol_rel'] = 1e-4


    H = {
        'method': args.method,
        'problem': {
            'id':
                '{}_left_{}_right_{}_seed_{}'.format(args.graph_generator,
                                                     args.l, args.r, args.seed),
            'graph_generator':
                args.graph_generator,
            'left':
                args.l,
            'right':
                args.r,
            'seed':
                args.seed,
            'graph_generator_seed':
                graph_generator_seed
        }
    }

    obj_val, num_parameters, all_x, all_vals = get_obj_val(
    # obj_val, num_parameters, all_x, all_vals, all_samples_tuple, all_samples_energies = get_obj_val(
        args.graph_generator,
        args.l,
        args.r,
        p = args.p,
        seed=graph_generator_seed,
        obj_params='ndarray',
        sign=-1,
        backend=args.backend,
        backend_params=backend_params,
        problem_name=args.problem,
        return_x=1,
        weight=args.weighted,
        simulator_seed=args.seed,
        remove_edge=args.remove_edge)  # sign = -1 because all methods minimize

    outname = os.path.join(
        args.savedir, args.method,
        "{}_{}_l_{}_r_{}_p_{}_nparam_{}_noise_{}_init_pts_{}_nit_{}_seed_{}_graph_gen_seed_{}_max_active_runs_{}_sobol_{}_lopt_method_{}_rmedge_{}"
        .format(label, args.graph_generator, args.l, args.r, args.p, num_parameters,
                args.noise, init_points, n_iter, args.seed,
                graph_generator_seed, args.max_active_runs, args.sobol, args.localopt_method, args.remove_edge))
    if args.fourier:
        outname += "_fourier_{}".format(args.fourier_method)
    outname += ".p"
    print(outname)
    if os.path.isfile(outname):
        print('Output file {} already exists! Our job here is done'.format(
            outname))
        sys.exit(0)

    if args.ansatz == 'RYRZ':
        if args.sobol:
            raise ValueError("Sobol is not implemented yet for RYRZ ansatz")
        elif args.sample_points:
            raise ValueError("Sample points are not tested yet for RYRZ ansatz")
        else:
            sample_points = np.split(
                np.random.uniform(-np.pi + 0.25, np.pi - 0.25,
                                  args.niter * num_parameters), args.niter)
    elif args.ansatz == 'QAOA':
        if args.sobol:
            problem = {
              'num_vars': 2 * args.ansatz_depth,
              'bounds': [np.pi / 2 - 0.01] * args.ansatz_depth + [2*np.pi - 0.01] * args.ansatz_depth
            }
            sample_points = saltelli.sample(problem, args.niter*10)
            sample_points = sample_points[args.seed % 10 : ]
        elif args.sample_points:
            sample_points = get_past_best_sample_points(args.sample_points, args.ansatz_depth)
            add_extra_points_uniform(sample_points, args.niter, args.ansatz_depth)
        elif args.fourier:
            sample_points = get_fourier_init_points(args.fourier, args.ansatz_depth, method=args.fourier_method)
            if not args.fourier_no_extra_points:
                add_extra_points_uniform(sample_points, args.niter, args.ansatz_depth)
        else:
            ub = ( [np.pi / 2 - 0.01] * args.ansatz_depth + [2*np.pi - 0.01] * args.ansatz_depth ) * args.niter
            lb = [0.01]  * 2 * args.ansatz_depth * args.niter
            sample_points = np.split(np.random.uniform(lb,ub, 2 * args.ansatz_depth * args.niter), args.niter)
    else:
        raise ValueError("Unsupported ansatz: {}".format(args.ansatz))

    params['sample_points'] = sample_points
    restart_points = []
    restart_positions = [] # restart position in all_x
    if args.method == 'libensemble':
        res_tuple = qaoa_libensemble.optimize_obj(obj_val, num_parameters,
                                                  params)
        # libensemble is kinda problematic
        if main_proc:
            # cannot simply put two things to the left of equality since None is not iterable
            res = res_tuple[0]
            persis_info = res_tuple[1]
            all_x = res['x']
            H['persis_info'] = persis_info
            all_vals = [{'mean': -x, 'max': None} for x in res['f']]
    elif args.method == 'MLSL_NLOPT':
        res = mlsl_nlopt.optimize_obj(obj_val, num_parameters, params)
    elif args.method == 'PASS_THROUGH':
        res = pass_through.optimize_obj(obj_val, num_parameters, params)
    else:
        # assuming the method is local
        # for local methods, once they converge, restart until exhausting the eval limit
        while len(all_vals) < args.niter and len(params['sample_points']) != 0:
            restart_points.append(params['sample_points'][0])
            restart_positions.append(len(all_vals))
            print("Starting {}, current nevals {}".format(args.method, len(all_vals)))
            try:
                if args.method == 'COBYLA_NLOPT':
                    res = cobyla_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'BOBYQA_NLOPT':
                    res = bobyqa_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'NELDERMEAD_NLOPT':
                    res = neldermead_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'NEWUOA_NLOPT':
                    res = newuoa_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'PRAXIS_NLOPT':
                    res = praxis_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'SBPLX_NLOPT':
                    res = sbplx_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'SAMI':
                    res = sami_bayes.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'SEQ_OPT':
                    res = sami_seqopt.optimize_obj(obj_val, num_parameters, params)
                else:
                    raise ValueError('Incorrect method: {}'.format(args.method))
            except (nlopt.RoundoffLimited, FloatingPointError) as e:
                print("Encountered {}, recovering results at iter {}".format(e, len(all_vals)))
                res = None
            assert len(all_x) == len(all_vals)
            params['sample_points'] = params['sample_points'][1:]
            if not args.restart_local:
                break

    # If in running the try-except run over the limit, truncate the result to just first niter
    if len(all_x) > args.niter or len(all_vals) > args.niter:
        all_x = all_x[:args.niter]
        all_vals = all_vals[:args.niter]

    if main_proc:
        print("Total nevals", len(all_x))
        H['x'] = all_x
        H['values'] = all_vals
        H['restart_points'] = restart_points
        H['restart_positions'] = restart_positions
        H['num_parameters'] = num_parameters
        H['raw_output'] = res
        print("energy from optimizer: {}".format(
            max([x['mean'] for x in H['values']])))

        if args.save:
            print("\n\n\nRun completed.\nSaving results to file: " + outname)
            params.update({
                'l': args.l,
                'r': args.r,
                'graph_generator': args.graph_generator,
                'seed': args.seed,
                'ansatz_depth': backend_params['depth'],
                'args':args
            })
            pickle.dump((H, params), open(outname, "wb"))
    if args.mpi:
        MPI.COMM_WORLD.Barrier()
