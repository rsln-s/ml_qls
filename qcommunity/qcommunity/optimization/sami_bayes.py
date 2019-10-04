# QAOA parameter optimization using COBYLA

from qcommunity.optimization.obj import get_obj_val, get_obj
from scipy.optimize import minimize
import numpy as np
import nlopt
import GPyOpt
import GPy
import sobol_seq

def optimize_obj(obj_val, num_parameters, params=None):

    #params['ansatz']
    #params['init_points'] = number of points that will be used to initialize the model
    #params['n_iter'] = number of function evaluation
    #params['xtol_rel'] = minimum distance between two consecutive x's to keep running the model,1e-8

    space = []
    if params['ansatz'] == 'QAOA':
        #initialize p=num_parameters/2 betas
        for n in range(0,int(num_parameters/2)):
            temp_var = {
                'name': 'var_'+str(n),
                'type': 'continuous',
                'domain': (0,np.pi / 2)
                }
            space.append(temp_var)
        #initialize p=num_parameters/2 gammas
        for n in range(int(num_parameters/2),num_parameters):
            temp_var = {
                'name': 'var_'+str(n),
                'type': 'continuous',
                'domain': (0,2*np.pi)
                }
            space.append(temp_var)
    elif params['ansatz'] == 'RYRZ':
        for n in range(0,num_parameters):
            temp_var = {
                'name': 'var_'+str(n),
                'type': 'continuous',
                'domain': (-np.pi,np.pi)
                }
            space.append(temp_var)

    feasible_region = GPyOpt.Design_space(space = space, constraints = None)
    #initial points where the function is evaluated so that model can be initialized
    initial_design = GPyOpt.experiment_design.initial_design('sobol', feasible_region, params['init_points'])

    # define the objective
    objective = GPyOpt.core.task.SingleObjective(obj_val)

    # define the model type
    kernel = GPy.kern.Matern52(input_dim=num_parameters)
    model = GPyOpt.models.GPModel(exact_feval=False,optimize_restarts=10,verbose=False, kernel=kernel)

    # define the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region, optimizer='lbfgs')

    # define the type of acquisition function
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

    # define collection method (sequential or batch)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    # BO object
    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

    # --- Stop conditions
    try:
        max_iter = params['n_iter_local']
    except (KeyError, TypeError):
        max_iter = 100

    # Run the optimization
    bo.run_optimization(max_iter = max_iter, max_time = None, eps = params['xtol_rel'], verbosity=False)
    return bo.x_opt
