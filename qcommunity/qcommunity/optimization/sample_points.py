import pickle
import numpy as np
from scipy.spatial import distance
from scipy.fftpack import dst, dct, idst, idct
from collections import defaultdict

def get_best_points_enforce_distance(approximation_info, eps, p):
    best_pts = []
    for _, l in approximation_info.items():
        for d in l:
            if d['ansatz_depth'] != p-1:
                continue
            for point in d['QAOA best points']:
                if len(best_pts) == 0 or not any([distance.euclidean(x, point) < eps for x in best_pts]):
                    best_pts.append(point)
                    if len(best_pts) >= 10:
                        return best_pts
    return best_pts

def enforce_bounds(point, period, start, end):
    for i in range(start, end):
        if point[i] > period:
            point[i] -= period
            continue
        if point[i] < 0:
            point[i] += period


# Uses fourier transform to produce initial points for higher dimensions
# as described in https://arxiv.org/abs/1812.01041

def fourier_interpolation(best_pts, p):
    sample_points = []
    for pt in best_pts:
        betas = pt[:p-1] 
        gammas = pt[p-1:]
        # u,v for p
        u_p = np.hstack([idst(gammas, type=4, norm='ortho'), 0])
        v_p = np.hstack([idct(betas, type=4, norm='ortho'), 0])

        new_gammas = dst(u_p, type=4, norm='ortho')
        new_betas = dct(v_p, type=4, norm='ortho')
        new_sample_point = np.hstack([new_betas, new_gammas])
        enforce_bounds(new_sample_point, np.pi/2, 0, p)
        enforce_bounds(new_sample_point, 2*np.pi, p, 2*p)
        sample_points.append(new_sample_point)
    return sample_points

def naive_interpolation(best_pts, p):
    sample_points = []
    for pt in best_pts:
        betas = pt[:p-1] 
        gammas = pt[p-1:]
        new_sample_point = np.hstack([betas, np.zeros(1), gammas, np.zeros(1)])
        print(new_sample_point)
        sample_points.append(new_sample_point)
    return sample_points

def get_fourier_init_points(fpath, p, eps=0.2, method='FOURIER'):
    approximation_info = pickle.load(open(fpath, 'rb'))

    if 'best points period enforced' in approximation_info:
        best_pts = approximation_info['best points period enforced'][p-1]
    else:
        best_pts = get_best_points_enforce_distance(approximation_info, eps, p)

    if len(best_pts) == 0:
        raise ValueError("No candidate points found! Possibly, incorrect ansatz depth specified")

    if method == 'FOURIER':
        return fourier_interpolation(best_pts, p)
    if method == 'INTERP':
        return naive_interpolation(best_pts, p)
    else:
        raise ValueError("Incorrect method received: {}".format(method))

def get_past_best_sample_points(fpath, p):
    t = pickle.load(open(fpath, 'rb'))
    if type(t) == tuple:
        # legacy
        return t[0]
    elif type(t) == defaultdict:
        # assuming produced by  approximation_ratio.py
        return t['best points period enforced'][p]


# if not enough sample points, throw in some extra just uniformly:
def add_extra_points_uniform(sample_points, npoints_total_expected, p):
    if len(sample_points) < npoints_total_expected:
        nextra_points = npoints_total_expected - len(sample_points)
        ub = ( [np.pi / 2 - 0.01] * p + [2*np.pi - 0.01] * p ) * nextra_points
        lb = [0.01]  * 2 * p * nextra_points
        sample_points.extend(np.split(np.random.uniform(lb,ub, 2 * p * nextra_points), nextra_points))

