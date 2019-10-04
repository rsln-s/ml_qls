# A do-nothing optimizer
# Simply passes through the sample points

def optimize_obj(obj_val, num_parameters, params=None):
    if 'sample_points' not in params:
        raise ValueError("pass_through: No sample points provided in params") 
    
    points = params['sample_points']
    nsamples = params['n_iter'] + params['init_points']
    for i in range(nsamples):
        if i >= len(points):
            raise ValueError("Not enough points provided to pass_through: found {}, need {}".format(len(points), nsamples))
        obj_val(points[i])
