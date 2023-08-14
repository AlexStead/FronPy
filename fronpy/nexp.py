from numpy import array,append,diag,exp,log,cbrt,sqrt,sum,max,maximum
from statsmodels.api import OLS
from scipy.stats import norm.pdf,norm.cdf,norm.logcdf
from . import utils

def density(epsilon,lnsigmav,lnsigmau,cost=False):
    density = exp(lndensity(epsilon,lnsigmav,lnsigmau,cost))
    return density

def efficiency(params,data,predictor='bc',cost=False):
    k = data.shape[1]-1
    y = data[:,0]
    X = data[:,1:k+1]
    b = params[0:k]
    epsilon = (y - X @ b)
    sigmav = exp(params[-2])
    sigmau = exp(params[-1])
    if cost:
        s = -1
    else:
        s = 1
    z = (s*epsilon/sigmav + sigmav/sigmau)
    if predictor == 'bc':
        efficiency = ((1 - norm.cdf(z + sigmav))/(1-norm.cdf(z))*
                      exp(s*epsilon+sigmav**2/sigmau+sigmav**2/2))
    if predictor == 'jlms':
        efficiency = exp(-sigmav*(norm.pdf(z)/norm.cdf(-z)-z))
    if predictor == 'mode':
        efficiency = exp(-maximum(0,-sigmav*z))
    return(efficiency)

def lndensity(epsilon,lnsigmav,lnsigmau,cost=False):
    sigmav = exp(lnsigmav)
    sigmau = exp(lnsigmau)
    if cost:
        s = -1
    else:
        s = 1
    lndensity = (- lnsigmau+1/2*(sigmav/sigmau)**2 + s*epsilon/sigmau +
                 norm.logcdf(-s*epsilon/sigmav-sigmav/sigmau))
    return lndensity

def lnlikelihood(params,data,cost=False):
    k = data.shape[1]-1
    y = data[:,0]
    X = data[:,1:k+1] #remember 1:n is not inclusive of n, so this really gives 1:k
    b = params[0:k] #see comment above
    lnsigmav = params[k]
    lnsigmau = params[k+1]
    epsilon=(y - X @ b)
    return lndensity(epsilon,lnsigmav,lnsigmau,cost).sum()

def minuslnlikelihood(params,data,cost=False):
    minuslnlikelihood = -lnlikelihood(params,data,cost)
    return minuslnlikelihood

def startvals(data,cost=False):
    y = data[:,0]
    k = data.shape[1]-1
    X = data[:,1:k+1]
    b_ols = (OLS(y,X).fit()).params
    ehat_ols = (OLS(y,X).fit()).resid
    n = X.shape[0]
    if cost:
        s = -1
    else:
        s = 1
    m2 = 1/n*sum(ehat_ols**2)
    m3 = s/n*sum(ehat_ols**3)
    m4 = 1/n*sum(ehat_ols**4)
    sigmau_cols = cbrt(-m3/2)
    sigmav_cols = sqrt(max((m2-sigmau_cols**2,1.0e-20)))
    cons_cols = b_ols[-1] + sigmau_cols
    e_params =  array([log(sigmav_cols),
                          log(sigmau_cols)])
    b_cols = append(b_ols[0:-1],cons_cols)
    theta_cols = append(b_cols,e_params, axis=None)
    return theta_cols

class Frontier:
    def __init__(self,lnlikelihood,k,theta,score,hess_inv,iterations,
                 func_evals,score_evals,status,success,message,
                 yhat,residual,eff_bc,eff_jlms,eff_mode):
        self.lnlikelihood = lnlikelihood
        self.theta = theta
        self.theta_se = sqrt(diag(hess_inv))
        self.theta_pval =  2*norm.cdf(-abs(theta/sqrt(diag(hess_inv))))
        self.theta_star = utils.calculate_stars(2*norm.cdf(-abs(theta/
                                                         sqrt(diag(hess_inv)))))
        self.beta = theta[0:k-1]
        self.beta_se = sqrt(diag(hess_inv))[0:k-1]
        self.beta_pval =  2*norm.cdf(-abs(theta[0:k-1]/sqrt(diag(hess_inv))[0:k-1]))
        self.beta_star = utils.calculate_stars(2*norm.cdf(-abs(theta[0:k-1]/
                                                         sqrt(diag(hess_inv))[0:k-1])))
        self.sigmav = exp(theta[k-1])
        self.sigmav_se = sqrt(exp(theta[k-1])**2*diag(hess_inv)[k-1])
        self.sigmau = exp(theta[k])
        self.sigmau_se = sqrt(exp(theta[k])**2*diag(hess_inv)[k])
        self.score = score
        self.hess_inv = hess_inv
        self.iterations = iterations
        self.func_evals = func_evals
        self.score_evals = score_evals
        self.status = status
        self.success = success
        self.message = message
        self.yhat = yhat
        self.residual = residual
        self.eff_bc = eff_bc
        self.eff_jlms = eff_jlms
        self.eff_mode = eff_mode

    def __repr__(self):
        return (f"lnlikelihood: {self.lnlikelihood}\n"
                f"beta: {self.beta}\n"
                f"beta_se: {self.beta_se}\n"
                f"beta_pval: {self.beta_pval}\n"
                f"beta_star: {self.beta_star}\n"
                f"sigmav: {self.sigmav}\n"
                f"sigmav_se: {self.sigmav_se}\n"
                f"sigmau: {self.sigmau}\n"
                f"sigmau_se: {self.sigmau_se}\n"
                f"theta: {self.theta}\n"
                f"theta_se: {self.theta_se}\n"
                f"theta_pval: {self.theta_pval}\n"
                f"theta_star: {self.theta_star}\n"
                f"score: {self.score}\n"
                f"hess_inv: {self.hess_inv}\n"
                f"iterations: {self.iterations}\n"
                f"func_evals: {self.func_evals}\n"
                f"score_evals: {self.score_evals}\n"
                f"status: {self.status}\n"
                f"message: {self.message}\n"
                f"yhat: {self.yhat}\n"
                f"residual: {self.residual}\n"
                f"eff_bc: {self.eff_bc}\n"
                f"eff_jlms: {self.eff_jlms}\n"
                f"eff_mode: {self.eff_mode}\n")