import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from . import utils
from . import nexp
from . import ng
from . import nhn
from . import ntn
np.seterr(all='ignore')

def estimate(data,model='nhn',cost=False,startingvalues=None,algorithm='BFGS',gtol=1e-5):
    n = data.shape[0]
    k = data.shape[1]
    list = {'nexp': nexp,
            'ng': ng,
            'nhn': nhn,
            'ntn': ntn}
    if model in list:
        module = list[model]
    else:
        raise ValueError("Unknown model:", model)
    if startingvalues is None:
        startingvalues = module.startvals(data, cost)
    if algorithm == 'BFGS':
        result = minimize(module.minuslnlikelihood,startingvalues,
                          method=algorithm,args=(data,cost),
                          options={'gtol': gtol*n,'disp': 2})
    else:
        result = minimize(module.minuslnlikelihood,startingvalues,
                          method=algorithm,args=(data,cost),
                          options={'disp': 2})
    frontier = module.Frontier(lnlikelihood = -result.fun,
                               k = k,
                               theta = result.x,
                               score = -result.jac,
                               hess_inv = result.hess_inv,
                               iterations = result.nit,#
                               func_evals = result.nfev,
                               score_evals = result.njev,
                               status = result.status,
                               success = result.success,
                               message = result.message,
                               yhat = utils.yhat(result.x,data),
                               residual = utils.residual(result.x,data),
                               eff_bc = module.efficiency(result.x,data,predictor='bc',cost=cost),
                               eff_jlms = module.efficiency(result.x,data,predictor='jlms',cost=cost),
                               eff_mode = module.efficiency(result.x,data,predictor='mode',cost=cost))
    return(frontier)