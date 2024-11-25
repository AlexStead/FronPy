import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.special
import scipy.fft
import scipy.interpolate
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

def cf(t,lnsigmav,lnsigmau,lnmu,mu,model='nhn',cost=False):
    sigmav = np.exp(lnsigmav)
    sigmau = np.exp(lnsigmau)
    if cost:
        s = -1
    else:
        s = 1
    if model == "nexp":
        #cf = np.exp(-0.5*sigmav**2*t**2)/(1-sigmau*1j*s*t)
        cf = np.exp(-0.5*sigmav**2*t**2-np.log(1+sigmau*1j*s*t))
    if model == "nhn":
        #cf = np.exp(-0.5*(sigmav**2+sigmau**2)*t**2)*(1+1j*scipy.special.erfi(-s*sigmau*t/np.sqrt(2)))
        cf = np.exp(-0.5*(sigmav**2+sigmau**2)*t**2+np.log(1-scipy.special.erf((s*1j*sigmau**2*t)/(np.sqrt(2)*sigmau)))
                    -np.log(0.5))
    if model == "ntn":
        #cf = np.exp(mu**1j*s*t-(sigmav**2+sigmau**2)*t**2/2)*(1+scipy.special.erfi(1j*s*sigmau*t))/2
        cf = np.exp(-mu**1j*s*t-0.5*(sigmav**2+sigmau**2)*t**2+np.log(1-scipy.special.erf((-mu/sigmau+s*1j*sigmau**2*t)/(np.sqrt(2)*sigmau)))
                    -np.log(1-scipy.special.erf(-mu/(np.sqrt(2)*sigmau))))
    if model in ('ng','nnak'):
        mu = np.exp(lnmu)
    if model == "ng":
        #cf = np.exp(-0.5*sigmav**2*t**2)*(1+sigmau*1j*s*t)**(-mu)
        cf = np.exp(-0.5*sigmav**2*t**2-mu*np.log(1+sigmau*1j*s*t))
    if model == "nnak":
        pcfd_array = np.frompyfunc(pcfd_complex, 2, 1)
        #cf = (sigmau**2/(2*mu))**mu*scipy.special.gamma(2*mu)*np.exp(0.5*t**2*(sigmau**2 /(4 *mu)-sigmav**2))*pcfd_array(-2*mu,-1j*sigmau*t/np.sqrt(2*mu))
        cf = np.exp(2*mu*np.log(sigmau)-mu*np.log(2)-mu*np.log(mu)+scipy.special.loggamma(2*mu)+0.5*t**2*(sigmau**2 /(4 *mu)-sigmav**2))*pcfd_array(-2*mu,s*1j*sigmau*t/np.sqrt(2*mu))
    return cf

def density(epsilon,lnsigmav,lnsigmau,lnmu,mu,model='nhn',cost=False,mpmath=False):
    if model in ('nhn','ntn','nexp','ng','nnak','nr'):
        return np.exp(lndensity(epsilon,lnsigmav,lnsigmau,lnmu,mu,model,cost))
    else:    
        raise ValueError("Invalid model:", model)
    
def efficiency(params,data,model='nhn',predictor='bc',cost=False,mpmath=False):
    if model in ('nhn','ntn','nexp','ng','nnak','nr'):
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
        elif model in ('ntn','ng','nnak'):
            sigmav = np.exp(params[-3])
            sigmau = np.exp(params[-2])
            if model == 'ntn':
                mu = params[-1]
            elif model in ('ng','nnak'):
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
        elif model == 'nnak':
            sigma = np.sqrt(2*mu*sigmav**2+sigmau**2)
            z = (s*epsilon*sigmau/sigmav)
            if predictor == 'bc':
                return ((np.exp(1/4*((z+sigmav*sigmau)/sigma)**2)*
                         scipy.special.pbdv(-2*mu,(z+sigmav*sigmau)/sigma)[0])/
                         (np.exp(1/4*(z/sigma)**2)*
                          scipy.special.pbdv(-2*mu,z/sigma)[0]))
            elif predictor == 'jlms':
                return np.exp(-2*mu*sigmav*sigmau/sigma*
                              scipy.special.pbdv(-2*mu-1,z/sigma)[0]/
                              scipy.special.pbdv(-2*mu,z/sigma)[0])
            elif predictor == 'mode':
                return np.array([np.exp(-sigmav*sigmau/(2*sigma)*np.nan_to_num(-z/sigma+np.sqrt((z/sigma)**2+4*(2*mu-1)),nan=0)),
                                 np.exp(-sigmav*sigmau/(2*sigma)*np.nan_to_num(-z/sigma-np.sqrt((z/sigma)**2+4*(2*mu-1)),nan=0))])
                #return np.exp(-sigmav*sigmau/(2*sigma)*np.maximum(0,
                #                                                  np.nan_to_num(-z/sigma+np.sqrt((z/sigma)**2-4*(2*mu-1)),nan=0),
                #                                                  np.nan_to_num(-z/sigma-np.sqrt((z/sigma)**2-4*(2*mu-1)),nan=0)))
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
                        return np.exp(-mu*sigmav*scipy.special.pbdv(-mu-1,z)[0]/scipy.special.pbdv(-mu,z)[0])
                elif predictor == 'mode':
                    if mpmath == True:
                        return ()
                    else:
                        return np.array([np.exp(-sigmav/2*np.nan_to_num(-z+np.sqrt(z**2+4*(mu-1)),nan=0)),
                                         np.exp(-sigmav/2*np.nan_to_num(-z-np.sqrt(z**2+4*(mu-1)),nan=0))])
                else:
                    raise ValueError("Unknown predictor:", predictor)
        elif model == 'nr':
            if predictor == 'bc':
                return ()
            else:
                    raise ValueError("Unknown predictor:", predictor)
    else:
        raise ValueError("Unknown model:", model)

def lndensity(epsilon,lnsigmav,lnsigmau,lnmu,mu,model='nhn',cost=False,mpmath=False,approximation=None):
    if model in ('nhn','ntn','nexp','ng','nnak','nr'):
        if mpmath:
            sigmav = mp.exp(lnsigmav)
            sigmau = mp.exp(lnsigmau)
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
        elif model in ('ng','nnak'):
            if mpmath == True:
                mu = mp.exp(lnmu)
                if model == 'ng':
                    return mp.mpf((mu-1)*lnsigmav - 1/2*mp.log(2) - 1/2*mp.log(mp.pi) 
                                  - mu*lnsigmau - 1/2*mp.power((epsilon/sigmav),2) 
                                  + 1/4*mp.power((s*epsilon/sigmav+sigmav/sigmau),2)
                                  + mp.log(mp.pcfd(-mu,s*epsilon/sigmav+sigmav/sigmau)))
                elif model == 'nnak':
                    return mp.mpf(mp.loggamma(2*mu) - mp.loggamma(mu) + 1/2*mp.log(2)
                                  - 1/2*mp.log(mp.pi) + mu*lnmu + (2*mu-1)*lnsigmav
                                  - mu*mp.log(sigmasq) - 1/2*mp.power((epsilon/sigmav),2)
                                  + 1/4*mp.power(((epsilon*sigmau/sigmav)/mp.sqrt(sigmasq)),2)
                                  + mp.log(mp.pcfd(-2*mu,(s*epsilon*sigmau/sigmav)/mp.sqrt(sigmasq))))
            else:
                mu = np.exp(lnmu)
                if model == 'ng':
                    return ((mu-1)*lnsigmav - 1/2*np.log(2) - 1/2*np.log(np.pi)
                            - mu*lnsigmau - 1/2*(epsilon/sigmav)**2 
                            + 1/4*(s*epsilon/sigmav+sigmav/sigmau)**2
                            + np.log(scipy.special.pbdv(-mu,s*epsilon/sigmav+sigmav/sigmau)[0]))
                elif model == 'nnak':
                    sigmasq = 2*mu*sigmav**2+sigmau**2
                    return (scipy.special.loggamma(2*mu) - scipy.special.loggamma(mu)
                            + 1/2*np.log(2) - 1/2*np.log(np.pi) + mu*lnmu
                            + (2*mu-1)*lnsigmav - mu*np.log(sigmasq) - 1/2*(epsilon/sigmav)**2
                            + 1/4*((epsilon*sigmau/sigmav)/np.sqrt(sigmasq))**2
                            + np.log(scipy.special.pbdv(-2*mu,(s*epsilon*sigmau/sigmav)/np.sqrt(sigmasq))[0]))
        elif model == 'nr':
            sigma = np.sqrt(2*sigmav**2+sigmau**2)
            z =  (s*epsilon*sigmau/sigmav)/sigma
            return (np.log(sigmav)- 2*np.log(sigma) - 1/2*(epsilon/sigmav)**2 + 1/2*z**2
                    + np.log(np.sqrt(2/np.pi)*np.exp(-1/2*z**2) - z*(1-scipy.special.erf(z/np.sqrt(2)))))
    else:
        raise ValueError("Unknown model:", model)

def lndensity_fft(epsilon,lnsigmav,lnsigmau,lnmu,mu,epsilonbar,model='nhn',cost=False,points=13,width=2):
    n = 2**points
    #h = (epsilonmax-epsilonmin)/(n-1)
    h = width*epsilonbar/n
    x = (np.arange(n) * h) - (n * h / 2)
    #x = np.linspace(epsilonmin, epsilonmax, n)
    s = 1 / (h * n)
    t = 2 * np.pi * s * (np.arange(n) - (n / 2))
    sgn = np.ones(n)
    sgn[1::2] = -1
    p = np.log(s*np.abs(np.fft.fft(sgn*cf(t,lnsigmav,lnsigmau,lnmu,mu,model=model,cost=cost))))
    cs = scipy.interpolate.interp1d(x,p,kind='linear',fill_value="extrapolate")
    return cs(epsilon)
    
def meanefficiency(params,model='nhn',p1=0,p2=1,mpmath=False):
    if model in ('nhn','ntn','nexp','ng','nnak','nr'):
        if model in ('nhn','nexp'):
            sigmau = np.exp(params[-1])
        elif model in ('ntn','ng','nnak'):
            sigmau = np.exp(params[-2])
            if model == 'ntn':
                mu = params[-1]
            elif model in ('ng','nnak'):
                mu = np.exp(params[-1])
        if p1 == 0 and p2 == 1:
            if model == 'nhn':
                return np.exp(sigmau**2/2)*scipy.special.erfc(sigmau/np.sqrt(2))
            if model == 'nexp':
                return 1/(1+sigmau)
            if model == 'ntn':
                return (np.exp(sigmau**2/2-mu)*
                        (scipy.special.erfc((sigmau**2-mu)/(np.sqrt(2)*sigmau)))/
                        (scipy.special.erfc(mu/(np.sqrt(2)*sigmau))))
            if model == 'nnak':
                return (2**mu/np.sqrt(np.pi)*scipy.special.gamma(mu+1/2)*np.exp(sigmau**2/(8*mu))
                        *scipy.special.pbdv(-2*mu,sigmau/np.sqrt(2*mu))[0])
            if model == 'ng':
                return 1/(1+sigmau)**mu
        elif p1 == 0 and 0 < p2 < 1:
            if model == 'nnak':
                k = 0
                meaneff = 0
                precision = np.finfo(float).eps
                while True:
                    summand = (1/scipy.special.factorial(k)*(-sigmau/np.sqrt(mu))**k*scipy.special.poch(mu,k/2)*
                               scipy.special.gammainc(mu+k/2,scipy.special.gammaincinv(mu,p2))/p2)
                    if np.abs(summand) < precision:
                        break
                    meaneff += summand
                    k += 1
                return meaneff
            if model == 'nexp':
                return (1-(1-p2)**(1+sigmau))/(p2*(1+sigmau))
            if model == 'ng':
                return scipy.special.gammainc(mu,scipy.special.gammaincinv(mu,p2)*(1+sigmau))/(p2*(1+sigmau)**mu)
            if model == 'ntn':
                return (np.exp(sigmau**2/2-mu)*(scipy.special.erf(scipy.special.erfinv(p2-(1+p2)*
                        scipy.special.erf(mu/(np.sqrt(2)*sigmau)))+sigmau/np.sqrt(2))+
                        scipy.special.erf((mu-sigmau**2)/(np.sqrt(2)*sigmau)))/
                        (p2*scipy.special.erfc(mu/(np.sqrt(2)*sigmau))))
            if model == 'nhn':
                return (np.exp(sigmau**2/2)*(scipy.special.erf(scipy.special.erfinv(p2)+sigmau/np.sqrt(2))-
                        scipy.special.erf(sigmau/(np.sqrt(2))))/p2)
        else:
            ValueError("Invalid percentiles:", p1, p2)
    else:
        raise ValueError("Unknown model:", model)

meanefficiencies = np.vectorize(meanefficiency, excluded=['params', 'model', 'p1', 'mpmath'])
    
def lnlikelihood(params,data,model='nhn',cost=False,mpmath=False,approximation=None,points=13,width=2):
    k = data.shape[1]-1
    y = data[:,0]
    X = data[:,1:k+1] #remember 1:n is not inclusive of n, so this really gives 1:k
    b = params[0:k] #see comment above
    epsilon=(y - X @ b)
    lnsigmav = params[k]
    lnsigmau = params[k+1]
    if model in ('ntn','ng','nnak'):
        if model == 'ntn':
            mu = params[k+2]
            lnmu = None
        elif model in ('ng','nnak'):
            mu = None
            lnmu = params[k+2]
    else:
        mu = None
        lnmu = None
    if mpmath == True:
        lndensityarray = np.frompyfunc(lndensity,8,1)
        return np.sum(lndensityarray(epsilon,lnsigmav,lnsigmau,lnmu,mu,model,cost,mpmath))
    elif approximation == 'fft':
        epsilonbar = np.max(np.abs(epsilon))
        return np.sum(lndensity_fft(epsilon,lnsigmav,lnsigmau,lnmu,mu,epsilonbar,model,cost,points,width))
    else:
        return np.sum(lndensity(epsilon,lnsigmav,lnsigmau,lnmu,mu,model,cost,mpmath))

def minuslnlikelihood(params,data,model='nhn',cost=False,mpmath=False,approximation=None,points=13,width=2):
    minuslnlikelihood = -lnlikelihood(params,data,model,cost,mpmath,approximation,points,width)
    return minuslnlikelihood

def pcfd_complex(v,z):
    return np.complex128(complex(mp.re(mp.pcfd(v,z)),mp.im(mp.pcfd(v,z))))

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
    if model in ('nhn','ntn','nnak'):
        sigmau_cols = np.cbrt(m3*np.sqrt(np.pi/2)/(1-4/np.pi))
        sigmav_cols = np.sqrt(np.max((m2-(1-2/np.pi)*sigmau_cols**2,1.0e-20)))
        cons_cols = b_ols[-1] + sigmau_cols*np.sqrt(2/np.pi)
        if model == 'nhn':
            e_params =  np.array([np.log(sigmav_cols),
                                  np.log(sigmau_cols)])
        elif model in ('ntn','nnak'):
            if model == 'ntn':
                mu = 0
                e_params =  np.array([np.log(sigmav_cols),
                                      np.log(sigmau_cols),
                                      mu])
            if model == 'nnak':
                lnmu = np.log(0.5)
                e_params =  np.array([np.log(sigmav_cols),
                                      np.log(sigmau_cols),
                                      lnmu])
    elif model == 'nexp':
        sigmau_cols = np.cbrt(-m3/2)
        sigmav_cols = np.sqrt(np.maximum(m2-sigmau_cols**2,1.0e-20))
        cons_cols = b_ols[-1] + sigmau_cols
        e_params =  np.array([np.log(sigmav_cols),
                              np.log(sigmau_cols)])
    elif model == 'ng':
        sigmau_cols = -(m4-3*m2**2)/(3*m3)
        mu_cols = -m3/(2*sigmau_cols**3)
        sigmav_cols = np.maximum(np.sqrt((m2-mu_cols*sigmau_cols**2,1.0e-20)))
        cons_cols = b_ols[-1]+ mu_cols*sigmau_cols
        e_params =  np.array([np.log(sigmav_cols),
                              np.log(sigmau_cols),
                              np.log(mu_cols)])
    elif model == 'nr':
        sigmau_cols = np.cbrt(-4*m3/(np.sqrt(np.pi)*(np.pi-3)))
        sigmav_cols = np.sqrt(np.maximum(m2 - sigmau_cols**2*(4-np.pi)/4,1.0e-20))
        cons_cols = b_ols[-1] + sigmau_cols*np.sqrt(np.pi)/2
        e_params =  np.array([np.log(sigmav_cols),
                              np.log(sigmau_cols)])
    b_cols = np.append(b_ols[0:-1],cons_cols)
    theta_cols = np.append(b_cols,e_params, axis=None)
    return theta_cols

class Frontier:
    def __init__(self,lnlikelihood,k,theta,score,hess_inv,iterations,
                 func_evals,score_evals,status,success,message,
                 yhat,residual,mean_eff,supra_pc_mean_eff,eff_bc,eff_jlms,
                 eff_mode,model):
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
        if model in ('ng','nnak'):
            self.mu = np.exp(theta[k+1])
            self.mu_se = np.sqrt(np.exp(theta[k+1])**2*np.diag(hess_inv)[k+1])
        self.mean_eff = mean_eff
        self.supra_pc_mean_eff = supra_pc_mean_eff
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
        if self.model in ('nhn','nexp','nr'):
            return (f"lnlikelihood: {self.lnlikelihood}\n"
                    f"beta: {self.beta}\n"
                    f"beta_se: {self.beta_se}\n"
                    f"beta_pval: {self.beta_pval}\n"
                    f"beta_star: {self.beta_star}\n"
                    f"sigmav: {self.sigmav}\n"
                    f"sigmav_se: {self.sigmav_se}\n"
                    f"sigmau: {self.sigmau}\n"
                    f"sigmau_se: {self.sigmau_se}\n"
                    f"mean_eff: {self.mean_eff}\n"
                    f"mean_eff_top_90: {self.supra_pc_mean_eff}\n"
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
        elif self.model in ('ntn','ng','nnak'):
            return (f"lnlikelihood: {self.lnlikelihood}\n"
                    f"beta: {self.beta}\n"
                    f"beta_se: {self.beta_se}\n"
                    f"beta_pval: {self.beta_pval}\n"
                    f"beta_star: {self.beta_star}\n"
                    f"sigmav: {self.sigmav}\n"
                    f"sigmav_se: {self.sigmav_se}\n"
                    f"sigmau: {self.sigmau}\n"
                    f"sigmau_se: {self.sigmau_se}\n"
                    f"mean_eff: {self.mean_eff}\n"
                    f"mean_eff_top_90: {self.supra_pc_mean_eff}\n"
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

def estimate(data,model='nhn',cost=False,startingvalues=None,algorithm='BFGS',tol=1e-4,mpmath=False,approximation=None,points=13,width=2):
    if model in ('nhn','ntn','nexp','ng','nnak','nr'):
        n = data.shape[0]
        k = data.shape[1]
        if startingvalues is None:
            startingvalues = startvals(data,model,cost)
        result = minimize(minuslnlikelihood,startingvalues,tol=tol*n,
                          method=algorithm,args=(data,model,cost,mpmath,approximation,points,width),
                          options={'disp': 2})
        frontier = Frontier(lnlikelihood = -result.fun,
                            k = k,
                            theta = result.x,
                            mean_eff = meanefficiency(result.x,model,0,1,False),
                            supra_pc_mean_eff = np.column_stack((np.array([0.1,
                                                                           0.2,
                                                                           0.3,
                                                                           0.4,
                                                                           0.5,
                                                                           0.6,
                                                                           0.7,
                                                                           0.8,
                                                                           0.9,
                                                                           0.95,
                                                                           0.99,
                                                                           1.0]),
                                                                meanefficiencies(params=result.x,
                                                                                 model=model,
                                                                                 p1=0,
                                                                                 p2=np.array([0.1,
                                                                                              0.2,
                                                                                              0.3,
                                                                                              0.4,
                                                                                              0.5,
                                                                                              0.6,
                                                                                              0.7,
                                                                                              0.8,
                                                                                              0.9,
                                                                                              0.95,
                                                                                              0.99,
                                                                                              1.0]),
                                                                                  mpmath=False))),
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