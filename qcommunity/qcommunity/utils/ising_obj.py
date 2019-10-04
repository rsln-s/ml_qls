import numpy as np

def ising_objective(problem_description, x):
    # only takes 0/1 strings 
    B_matrix = problem_description['B_matrix']
    B_bias = problem_description['B_bias']

    if isinstance(x, dict):
        # dict is assumed to be correct (i.e. +-1)
        print("Converting from dict form: ", x)
        x = np.asarray([x[i] for i in range(len(x))])
        print("to list form ", x)
    else:
        if -1 in x:
            raise ValueError("Oops, expected 0/1 string!")
        x = np.asarray([2*xi - 1 for xi in x]) 

    if B_bias.shape[0] == 1:
        # if bias is row vector -- flip
        B_bias = B_bias.T
    return (x.dot(B_matrix)).dot(x) + B_bias.T.dot(x)
