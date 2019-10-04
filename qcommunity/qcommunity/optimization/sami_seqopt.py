# QAOA parameter optimization using a Coordinate Search Based Method (Compass Search)
from scipy.optimize import minimize
import numpy as np
import sys

#Requirements:
#params['n_iter'] = number of function evaluation
#
#Sweep through angles randomly
#For each angle k,
#    evaluate the circuit 3 times
#        using initial angle set
#        using initial angle set, but change angle[k] to angle[k] + perturbation
#        using initial angle set, but change angle[k] to angle[k] - perturbation
#    using these evaluations
#        fit a function f(theta) = a1*cos(theta - a2) + a3
#        theta_new = argmin f(theta)
#    if angleset & theta_new improve objective
#        angle[k] = theta_new


class sequential_optimizer:

    def __init__(self,black_box_objective, num_parameters, params=None):
        self.coef = np.zeros((3,))
        self.objective = black_box_objective
        self.num_parameters=num_parameters

        # --- Stop conditions
        try:
            self.maxeval = params['n_iter']
        except (KeyError, TypeError):
            self.maxeval = 100
        self.objV = sys.maxsize
        self.opt_p = self.sequential_opt()

    # Main Optimization Function
    # Performs Compass Search / Coordinate Search Method
    def sequential_opt(self):
        #Randomly initialize the angles
        angles = np.random.uniform(-np.pi + 0.25, np.pi - 0.25, self.num_parameters)

        while (self.maxeval > 0):
            #iterate over parameters, in a random order
            for p in np.random.permutation(self.num_parameters):

                #add/subtract perturbation on parameter p
                pert = np.random.uniform(-np.pi + 0.25, np.pi - 0.25, 1)[0]
                angleP_add_pert = np.remainder(np.asarray([angles[p]+pert]), np.pi*2)
                angleP_sub_pert = np.remainder(np.asarray([angles[p]-pert]), np.pi*2)

                #EVALUATE CIRCUIT AT THREE POINTS, with ANGLE[P]
                #updated as: ANGLE[P], ANGLE[P]+Perturbation, ANGLE[P]-Perturbation
                angles_1 = angles
                angles_2 = np.concatenate((angles[0:p],angleP_add_pert,angles[p+1:self.num_parameters]),axis = 0)
                angles_3 = np.concatenate((angles[0:p],angleP_sub_pert,angles[p+1:self.num_parameters]),axis = 0)

                f1 = self.objective(angles_1,None)
                f2 = self.objective(angles_2,None)
                f3 = self.objective(angles_3,None)
                #We have done three evaluations to estimate a1,a2,a3
                self.maxeval -= 3

                #Using the three evaluations, we fit a function of the form f(theta) = a1*cos(theta-a2)+a3
                #and min_theta f(theta)
                newAngle = self.fit_minimize([f1,f2,f3],[angles[p],angles_2[p],angles_3[p]])
                newAngle = np.remainder(newAngle, np.pi*2)

                #if process succeeded
                if (newAngle != -sys.maxsize):
                    newAngles = np.concatenate((angles[0:p],newAngle,angles[p+1:self.num_parameters]),axis = 0)
                    newObjV = self.objective(newAngles,None)
                    #We spent one evaluation to check if we have improved the obj funtion value
                    self.maxeval -= 1
                    if (newObjV<self.objV):
                        #set angle[p] to new theta because this update has reduced the objective value
                        angles[p]=newAngle
                        self.objV=newObjV

            #print(self.objV)
            #if sweeping through the parameters did not update any of them, terminate optimization
            #if (angles==newAngles).all():
            #    break
        return angles


    #this function finds the sum of squared loss
    # \sigma_i (f_i - a1*cos(x_i - a2)+a3)**2
    def squared_loss_model(self,F,X,a1,a2,a3):
        temp = np.asarray(F) - self.cosine_model(X,a1,a2,a3)
        return np.dot(temp,temp)

    #the sinusoidal model we fit
    def cosine_model(self,X,a1,a2,a3):
        return a1*np.cos(np.asarray(X)-a2) + a3

    #the sinuosoidal model evaluated at theta, and optimal a1,a2,a3
    def cosine_function(self,theta):
        return self.cosine_model(theta,self.coef[0],self.coef[1],self.coef[2])


    def fit_minimize(self,F, X):
        #F is a 3x1 array with three evaluations of the circuit at different anlge_P
        #We will fit a function of the form f(x) = a1*cos(x-a2)+a3
        #And then we will minimize f(x) returning argmin_x f(x)
        def coef_loss(A):
            return self.squared_loss_model(F,X,A[0],A[1],A[2])

        #randomly select initial values for a1,a2,a3
        x0 = np.random.uniform(-np.pi + 0.25, np.pi - 0.25, 3)
        minProb1 = minimize(coef_loss, x0, method='BFGS', tol=1e-6)

        if minProb1.success:
            self.coef = minProb1.x


        #having fitted the function, we now have a1*cos(x-a1)+a3, which we minimize w.r.t x
        ang0 = np.random.uniform(-np.pi + 0.25, np.pi - 0.25, 1)
        minProb2 = minimize(self.cosine_function, ang0, method='BFGS', tol=1e-6)

        if minProb2.success:
            ang_opt =  minProb2.x
        else:
            ang_opt = -sys.maxsize

        return ang_opt


def optimize_obj(obj_val, num_parameters, params=None):
    def objective(x, grad):
        f = obj_val(x)
        return f
    optm_problem = sequential_optimizer(objective, num_parameters, params)
    optm_problem.opt_p
    return None
