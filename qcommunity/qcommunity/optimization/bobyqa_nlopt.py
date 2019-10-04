from qcommunity.optimization.obj import get_obj_val, get_obj
from scipy.optimize import minimize
import numpy as np
import nlopt


def optimize_obj(obj_val, num_parameters, params=None):
    options = {}
    try:
        init_points = params['sample_points'][0]
    except (KeyError, TypeError):
        init_points = np.random.uniform(-np.pi, np.pi, num_parameters)
    try:
        options['maxiter'] = params['n_iter_local']
    except (KeyError, TypeError):
        options['maxiter'] = 100

    def objective(x, grad):
        f = obj_val(x)
        return f

    nlopt.srand(params['seed'])
    opt = nlopt.opt(nlopt.LN_BOBYQA, num_parameters)
    opt.set_min_objective(objective)
    opt.set_maxeval(options['maxiter'])
    
    if params['ansatz'] == 'QAOA':
        lb = np.array([0, 0] * params['ansatz_depth'])
        ub = np.array([np.pi / 2] * params['ansatz_depth'] + [2*np.pi] * params['ansatz_depth'])
    elif params['ansatz'] == 'RYRZ':
        lb = np.array([-np.pi] * num_parameters)
        ub = np.array([np.pi] * num_parameters)

    #dist_to_bound = min(min(ub-init_points),min(init_points-lb))
    #opt.set_initial_step(dist_to_bound)
    opt.set_ftol_rel(params['ftol_rel'])     
    opt.set_xtol_rel(params['xtol_rel'])

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    x = opt.optimize(init_points)
    return x
