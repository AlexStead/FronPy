from fronpy import np
from fronpy import pkg_resources

def dataset(filename):
    path = pkg_resources.resource_filename('fronpy', 'data/' + filename)
    with open(path, 'r') as file:
        return np.genfromtxt(file, delimiter=',')

def yhat(params,data):
    k = data.shape[1]-1
    X = data[:,1:k+1]
    b = params[0:k]
    epsilon = (X @ b)
    return epsilon

def residual(params,data):
    epsilon = (data[:,0] - yhat(params,data))
    return epsilon

def calculate_star(pval):
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.1:
        return '*'
    else:
        return ''

calculate_stars = np.vectorize(calculate_star)

def estimate(data,cost=False,startingvalues=None,algorithm='BFGS',gtol=1e-5):
    n = data.shape[0]
    k = data.shape[1]
    if startingvalues is None:
        startingvalues = startvals(data, cost)
    if algorithm == 'BFGS':
        result = minimize(minuslnlikelihood,startingvalues,
                          method=algorithm,args=(data,cost),
                          options={'gtol': gtol*n,'disp': 2})
    else:
        result = minimize(minuslnlikelihood,startingvalues,
                          method=algorithm,args=(data,cost),
                          options={'disp': 2})
    frontier = Frontier(lnlikelihood = -result.fun,
                        k = k,
                        theta = result.x,
                        score = -result.jac,
                        hess_inv = result.hess_inv,
                        iterations = result.nit,
                        func_evals = result.nfev,
                        score_evals = result.njev,
                        status = result.status,
                        success = result.success,
                        message = result.message,
                        yhat = utils.yhat(result.x,data),
                        residual = utils.residual(result.x,data),
                        eff_bc = efficiency(result.x,data,predictor='bc',cost=cost),
                        eff_jlms = efficiency(result.x,data,predictor='jlms',cost=cost),
                        eff_mode = efficiency(result.x,data,predictor='mode',cost=cost))
    return(frontier)