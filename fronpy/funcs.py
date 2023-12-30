import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.special
import mpmath as mp
import pkg_resources
np.seterr(all='ignore')

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

def density(epsilon,lnsigmav,lnsigmau,lnmu,mu,model='nhn',cost=False,mpmath=False):
    if model in ('nhn','ntn','nexp','ng','nn'):
        return np.exp(lndensity(epsilon,lnsigmav,lnsigmau,lnmu,mu,model,cost))
    else:    
        raise ValueError("Invalid model:", model)

def lndensity(epsilon,lnsigmav,lnsigmau,lnmu,mu,model='nhn',cost=False,mpmath=False):
    if model in ('nhn','ntn','nexp','ng','nn'):
        if mpmath:
            sigmav = mp.log(lnsigmav)
            sigmau = mp.log(lnsigmau)
            mu = mp.exp(lnmu)
        else:
            sigmav = np.exp(lnsigmav)
            sigmau = np.exp(lnsigmau)
        if cost:
           s = -1
        else:
           s = 1
        if model in ('nhn','ntn'):
            sigma = np.sqrt(sigmav**2+sigmau**2)
            lambda_ = sigmau/sigmav
            if model == 'nhn':
                return (np.log(2) - np.log(sigma) + norm.logpdf(epsilon/sigma) +
                        norm.logcdf(-s*epsilon*lambda_/sigma))
            elif model == 'ntn':
                return (-np.log(sigma) + norm.logpdf((s*epsilon+mu)/sigma) +
                        norm.logcdf(mu/(lambda_*sigma)-s*epsilon*lambda_/sigma) -
                        norm.logcdf(mu/sigmau))
        elif model == 'nexp':
            return (- lnsigmau + 1/2*(sigmav/sigmau)**2 + s*epsilon/sigmau +
                    norm.logcdf(-s*epsilon/sigmav-sigmav/sigmau))
        elif model in ('ng','nn'):
            if mpmath == True:
                mu = mp.exp(lnmu)
                if model == 'ng':
                    return float((mu-1)*lnsigmav - 1/2*mp.log(2) - 1/2*mp.log(mp.pi) 
                                 - mu*lnsigmau - mp.power(1/2*(epsilon/sigmav),2) 
                                 + mp.power(1/4*(epsilon/sigmav+sigmav/sigmau),2)
                                 + mp.log(mp.pcfd(-mu,epsilon/sigmav+sigmav/sigmau)))
                elif model == 'nn':
                    return (mp.loggamma(2*mu) - mp.loggamma(mu) + 1/2*mp.log(2)
                            - 1/2*mp.log(mp.pi) + mu*lnmu + (2*mu-1)*lnsigmav
                            - mu*mp.log(sigmasq) - mp.power(1/2*(epsilon/sigmav),2)
                            + mp.power(1/4*((epsilon*sigmau/sigmav)/mp.sqrt(sigmasq)),2)
                            + mp.log(mp.pcfd(-2*mu,(epsilon*sigmau/sigmav)/mp.sqrt(sigmasq))))
            else:
                mu = np.exp(lnmu)
                if model == 'ng':
                    return ((mu-1)*lnsigmav - 1/2*np.log(2) - 1/2*np.log(np.pi)
                            - mu*lnsigmau - 1/2*(epsilon/sigmav)**2 
                            + 1/4*(s*epsilon/sigmav+sigmav/sigmau)**2
                            + np.log(scipy.special.pbdv(-mu,s*epsilon/sigmav+sigmav/sigmau)[0]))
                elif model == 'nn':
                    sigmasq = 2*mu*sigmav**2+sigmau**2
                    return (scipy.special.loggamma(2*mu) - scipy.special.loggamma(mu)
                            + 1/2*np.log(2) - 1/2*np.log(np.pi) + mu*lnmu
                            + (2*mu-1)*lnsigmav - mu*np.log(sigmasq) - 1/2*(epsilon/sigmav)**2
                            + 1/4*((epsilon*sigmau/sigmav)/np.sqrt(sigmasq))**2
                            + np.log(scipy.special.pbdv(-2*mu,(s*epsilon*sigmau/sigmav)/np.sqrt(sigmasq))[0]))
    else:
        raise ValueError("Unknown model:", model)

def efficiency(params,data,model='nhn',predictor='bc',cost=False,mpmath=False):
    if model in ('nhn','ntn','nexp','ng','nn'):
        if cost:
            s = -1
        else:
            s = 1
        k = data.shape[1]-1
        y = data[:,0]
        X = data[:,1:k+1]
        b = params[0:k]
        epsilon = (y - X @ b)
        if model in ('nhn','nexp'):
            sigmav = np.exp(params[-2])
            sigmau = np.exp(params[-1])
        elif model in ('ntn','ng','nn'):
            sigmav = np.exp(params[-3])
            sigmau = np.exp(params[-2])
            if model == 'ntn':
                mu = params[-1]
            elif model in ('ng','nn'):
                mu = np.exp(params[-1])
        if model in ('nhn','ntn'):
            sigma = np.sqrt(sigmav**2+sigmau**2)
            sigmastar = sigmau*sigmav/sigma
            if model == 'nhn':
                mustar = -s*epsilon*(sigmau/sigma)**2
            elif model == 'ntn':
                mustar = (sigmav**2*mu-s*epsilon*sigmau**2)/(sigma**2)
            if predictor == 'bc':
                return ((1-norm.cdf(sigmastar-mustar/sigmastar))/
                        (1-norm.cdf(-mustar/sigmastar))*
                        np.exp(-mustar+1/2*sigmastar**2))
            elif predictor == 'jlms':
                return (np.exp(-mustar-sigmastar*norm.pdf(-mustar/sigmastar)/
                               norm.cdf(mustar/sigmastar)))
            elif predictor == 'mode':
                    return np.exp(-np.maximum(0,mustar))
            else:
                raise ValueError("Unknown predictor:", predictor)
        elif model == 'nn':
            sigma = np.sqrt(2*mu*sigmav**2+sigmau**2)
            z = (s*epsilon*sigmau/sigmav)
            if predictor == 'bc':
                return ((np.exp(1/4*((z+sigmav*sigmau)/sigma)**2)*
                         scipy.special.pbdv(-2*mu,(z+sigmav*sigmau)/sigma)[0])/
                         (np.exp(1/4*(z/sigma)**2)*
                          scipy.special.pbdv(-2*mu,z/sigma)[0]))
            elif predictor == 'jlms':
                return np.exp(-sigmav*sigmau/sigma*scipy.special.poch(2*mu,1)*
                              scipy.special.pbdv(-2*mu-1,z/sigma)[0]/
                              scipy.special.pbdv(-2*mu,z/sigma)[0])
            elif predictor == 'mode':
                return np.exp(-sigmav*sigmau/(2*sigma)*np.maximum(0,np.nan_to_num(-z/sigma+np.sqrt((z/sigma)**2+4*(2*mu-1)),nan=0)))
            else:
                raise ValueError("Unknown predictor:", predictor)
        elif model in ('nexp','ng'):
            z = (s*epsilon/sigmav + sigmav/sigmau)
            if model == 'nexp':
                if predictor == 'bc':
                    return ((1 - norm.cdf(z + sigmav))/(1-norm.cdf(z))*
                            np.exp(s*epsilon+sigmav**2/sigmau+sigmav**2/2))
                elif predictor == 'jlms':
                    return np.exp(-sigmav*(norm.pdf(z)/norm.cdf(-z)-z))
                elif predictor == 'mode':
                    return np.exp(-np.maximum(0,-sigmav*z))
                else:
                    raise ValueError("Unknown predictor:", predictor)
            if model == 'ng':
                if predictor == 'bc':
                    if mpmath == True:
                        return ()
                    else:
                        return (np.exp(1/4*(z+sigmav)**2)*scipy.special.pbdv(-mu,z+sigmav)[0]/
                                (np.exp(1/4*(z)**2)*scipy.special.pbdv(-mu,z)[0]))
                elif predictor == 'jlms':
                    if mpmath == True:
                        return ()
                    else:
                        return np.exp(-sigmav*scipy.special.gamma(mu+1)/scipy.special.gamma(mu)*
                                      scipy.special.pbdv(-mu-1,z)[0]/scipy.special.pbdv(-mu,z)[0])
                elif predictor == 'mode':
                    if mpmath == True:
                        return ()
                    else:
                        return np.exp(-sigmav/2*np.maximum(0,np.nan_to_num(-z+np.sqrt(z**2+4*(mu-1)),nan=0)))
                else:
                    raise ValueError("Unknown predictor:", predictor)
    else:
        raise ValueError("Unknown model:", model)
    
def lnlikelihood(params,data,model='nhn',cost=False,mpmath=False):
    k = data.shape[1]-1
    y = data[:,0]
    X = data[:,1:k+1] #remember 1:n is not inclusive of n, so this really gives 1:k
    b = params[0:k] #see comment above
    epsilon=(y - X @ b)
    lnsigmav = params[k]
    lnsigmau = params[k+1]
    if model in ('ntn','ng','nn'):
        if model == 'ntn':
            mu = params[k+2]
            lnmu = None
        elif model in ('ng','nn'):
            mu = None
            lnmu = params[k+2]
    else:
        mu = None
        lnmu = None
    return np.sum(lndensity(epsilon,lnsigmav,lnsigmau,lnmu,mu,model,cost,mpmath))

def minuslnlikelihood(params,data,model='nhn',cost=False,mpmath=False):
    minuslnlikelihood = -lnlikelihood(params,data,model,cost,mpmath)
    return minuslnlikelihood

def startvals(data,model='nhn',cost=False):
    y = data[:,0]
    k = data.shape[1]-1
    X = data[:,1:k+1]
    b_ols = (sm.OLS(y,X).fit()).params
    ehat_ols = (sm.OLS(y,X).fit()).resid
    n = X.shape[0]
    if cost:
        s = -1
    else:
        s = 1
    m2 = 1/n*np.sum(ehat_ols**2)
    m3 = s/n*np.sum(ehat_ols**3)
    m4 = 1/n*np.sum(ehat_ols**4)
    if model in ('nhn','ntn','nn'):
        sigmau_cols = np.cbrt(m3*np.sqrt(np.pi/2)/(1-4/np.pi))
        sigmav_cols = np.sqrt(np.max((m2-(1-2/np.pi)*sigmau_cols**2,1.0e-20)))
        cons_cols = b_ols[-1] + sigmau_cols*np.sqrt(2/np.pi)
        if model == 'nhn':
            e_params =  np.array([np.log(sigmav_cols),
                                  np.log(sigmau_cols)])
        elif model in ('ntn','nn'):
            if model == 'ntn':
                mu = 0
                e_params =  np.array([np.log(sigmav_cols),
                                      np.log(sigmau_cols),
                                      mu])
            if model == 'nn':
                lnmu = np.log(0.5)
                e_params =  np.array([np.log(sigmav_cols),
                                      np.log(sigmau_cols),
                                      lnmu])
    elif model == 'nexp':
        sigmau_cols = np.cbrt(-m3/2)
        sigmav_cols = np.sqrt(np.max((m2-sigmau_cols**2,1.0e-20)))
        cons_cols = b_ols[-1] + sigmau_cols
        e_params =  np.array([np.log(sigmav_cols),
                              np.log(sigmau_cols)])
    elif model == 'ng':
        sigmau_cols = -(m4-3*m2**2)/(3*m3)
        mu_cols = -m3/(2*sigmau_cols**3)
        sigmav_cols = np.sqrt(np.max((m2-mu_cols*sigmau_cols**2,1.0e-20)))
        cons_cols = b_ols[-1]+ mu_cols*sigmau_cols
        e_params =  np.array([np.log(sigmav_cols),
                              np.log(sigmau_cols),
                              np.log(mu_cols)])
    b_cols = np.append(b_ols[0:-1],cons_cols)
    theta_cols = np.append(b_cols,e_params, axis=None)
    return theta_cols

class Frontier:
    def __init__(self,lnlikelihood,k,theta,score,hess_inv,iterations,
                 func_evals,score_evals,status,success,message,
                 yhat,residual,eff_bc,eff_jlms,eff_mode,model):
        self.lnlikelihood = lnlikelihood
        self.theta = theta
        self.theta_se = np.sqrt(np.diag(hess_inv))
        self.theta_pval =  2*norm.cdf(-abs(theta/np.sqrt(np.diag(hess_inv))))
        self.theta_star = calculate_stars(2*norm.cdf(-abs(theta/
                                                          np.sqrt(np.diag(hess_inv)))))
        self.beta = theta[0:k-1]
        self.beta_se = np.sqrt(np.diag(hess_inv))[0:k-1]
        self.beta_pval =  2*norm.cdf(-abs(theta[0:k-1]/np.sqrt(np.diag(hess_inv))[0:k-1]))
        self.beta_star = calculate_stars(2*norm.cdf(-abs(theta[0:k-1]/
                                                         np.sqrt(np.diag(hess_inv))[0:k-1])))
        self.sigmav = np.exp(theta[k-1])
        self.sigmav_se = np.sqrt(np.exp(theta[k-1])**2*np.diag(hess_inv)[k-1])
        self.sigmau = np.exp(theta[k])
        self.sigmau_se = np.sqrt(np.exp(theta[k])**2*np.diag(hess_inv)[k])
        if model == 'ntn':
            self.mu = theta[k+1]
            self.mu_se = np.sqrt(np.diag(hess_inv))[k+1]
        if model in ('ng','nn'):
            self.mu = np.exp(theta[k+1])
            self.mu_se = np.sqrt(np.exp(theta[k+1])**2*np.diag(hess_inv)[k+1])
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
        self.model = model

    def __repr__(self):
        if self.model in ('nhn','nexp'):
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
                    f"eff_mode: {self.eff_mode}\n"
                    f"model: {self.model}\n")
        elif self.model in ('ntn','ng','nn'):
            return (f"lnlikelihood: {self.lnlikelihood}\n"
                    f"beta: {self.beta}\n"
                    f"beta_se: {self.beta_se}\n"
                    f"beta_pval: {self.beta_pval}\n"
                    f"beta_star: {self.beta_star}\n"
                    f"sigmav: {self.sigmav}\n"
                    f"sigmav_se: {self.sigmav_se}\n"
                    f"sigmau: {self.sigmau}\n"
                    f"sigmau_se: {self.sigmau_se}\n"
                    f"mu: {self.mu}\n"
                    f"mu_se: {self.mu_se}\n"
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
                    f"eff_mode: {self.eff_mode}\n"
                    f"model: {self.model}\n")

def estimate(data,model='nhn',cost=False,startingvalues=None,algorithm='BFGS',tol=1e-4,mpmath=False):
    if model in ('nhn','ntn','nexp','ng','nn'):
        n = data.shape[0]
        k = data.shape[1]
        if startingvalues is None:
            startingvalues = startvals(data,model,cost)
        result = minimize(minuslnlikelihood,startingvalues,tol=tol*n,
                          method=algorithm,args=(data,model,cost,mpmath),
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
                            yhat = yhat(result.x,data),
                            residual = residual(result.x,data),
                            eff_bc = efficiency(result.x,data,model,predictor='bc',cost=cost),
                            eff_jlms = efficiency(result.x,data,model,predictor='jlms',cost=cost),
                            eff_mode = efficiency(result.x,data,model,predictor='mode',cost=cost),
                            model = model)
        return(frontier)
    else:
        raise ValueError("Unknown model:", model)