import sys
import numpy as np
import pandas as pd
from formulaic import Formula
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.special
import scipy.fft
import scipy.interpolate
import mpmath as mp
import pkg_resources
np.seterr(all='ignore')

def dataset(filename, dataframe=False):
    """
    Load a dataset from a CSV file within the 'data' directory of the 'fronpy' package.

    This function locates the specified file in the 'data' directory of the package,
    reads it as either a NumPy array or a pandas DataFrame based on the 'dataframe' option.

    Parameters:
    filename (str): The name of the CSV file to load.
    dataframe (bool): If True, reads as a pandas DataFrame. Default is False.

    Returns:
    numpy.ndarray or pandas.DataFrame: The data from the file.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    ValueError: If the file cannot be parsed as a valid CSV.

    Example:
    >>> electricity = fronpy.dataset('electricity.csv')
    >>> print(electricity[:5, :])
    [[-2.11193924e+00  1.00000000e+00 -7.77287338e+00  6.04175605e+01
      -3.59426334e-01  1.65018066e-01]
     [-1.00910571e+00  1.00000000e+00 -7.07972620e+00  5.01225230e+01
       3.86433827e-01  4.21877886e-01]]

    >>> electricity_df = fronpy.dataset('electricity_df.txt', dataframe=True)
    >>> print(electricity_df.head())
       Column1  Column2   Column3  Column4  Column5   Column6
    0 -2.111939     1.0 -7.772873    60.42 -0.359426  0.165018
    1 -1.009106     1.0 -7.079726    50.12  0.386434  0.421878
    """
    if dataframe:
        # Adjust filename to point to '_df.txt' if dataframe=True
        path = pkg_resources.resource_filename('fronpy', 'data/' + filename)
        return pd.read_csv(path)
    else:
        path = pkg_resources.resource_filename('fronpy', 'data/' + filename)
        with open(path, 'r') as file:
            return np.genfromtxt(file, delimiter=',')

def yhat(params,data):
    """
    Computes the estimated frontier (y-hat) for observations based 
    on the given parameter estimates and input data.

    Parameters:
    -----------
    params : array-like
        A 1D array containing the regression coefficients (beta values). 
        These represent the estimated effects of the predictors on the outcome.

    data : array-like, shape (n_samples, n_features + 1)
        A 2D array where the first column typically represents a constant or 
        identifier (not used in the prediction). The remaining columns are 
        the predictor variables (X).

    Returns:
    --------
    y : numpy.ndarray
        A 1D array of predicted values, calculated as the linear combination 
        of the predictors and their corresponding coefficients.
    
    Notes:
    ------
    This function assumes that the predictors in `data` are arranged in columns 
    from the second column onward, excluding the first column. The linear model 
    is defined as:

        y = X @ b

    where:
        - X is the matrix of predictor variables (extracted from `data`),
        - b is the vector of regression coefficients (extracted from `params`),
        - y is the resulting vector of predicted values.

    Example:
    --------
    >>> import numpy as np
    >>> params = np.array([0.5, 1.5])
    >>> data = np.array([[1, 2, 3], [1, 4, 5]])
    >>> yhat(params, data)
    array([ 5.5, 11.5 ])
    """
    k = data.shape[1]-1
    X = data[:,1:k+1]
    b = params[0:k]
    y = (X @ b)
    return y

def residual(params,data):
    """
    Computes the residuals (observed minus predicted values) for the stochastic 
    frontier model.

    Parameters:
    -----------
    params : array-like
        A 1D array containing the estimated coefficients (beta values) for the 
        deterministic component of the model.

    data : array-like, shape (n_samples, n_features + 1)
        A 2D array where the first column contains the observed dependent variable 
        (e.g., output), and the remaining columns represent the covariates (X).

    Returns:
    --------
    epsilon : numpy.ndarray
        A 1D array of residuals, calculated as the difference between the observed 
        dependent variable and the predicted deterministic component of the model.

    Example:
    --------
    >>> import numpy as np
    >>> params = np.array([0.5, 1.5])
    >>> data = np.array([[10, 2, 3], [15, 4, 5]])
    >>> residual(params, data)
    array([4.5, 3.5])
    """
    epsilon = (data[:,0] - yhat(params,data))
    return epsilon

def calculate_star(pval):
    """
    Assigns significance stars based on the p-value for hypothesis testing results.

    Parameters:
    -----------
    pval : float
        The p-value from a statistical test, indicating the probability of observing 
        the test statistic under the null hypothesis.

    Returns:
    --------
    str
        A string representing the significance level:
        - '***' for p-values < 0.01 (highly significant),
        - '**' for p-values < 0.05 (moderately significant),
        - '*' for p-values < 0.1 (weakly significant),
        - '' (empty string) for p-values ≥ 0.1 (not significant).

    Examples:
    --------
    >>> calculate_star(0.005)
    '***'
    >>> calculate_star(0.03)
    '**'
    >>> calculate_star(0.08)
    '*'
    >>> calculate_star(0.15)
    ''
    """
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
    
def efficiency(params,data,model='nhn',predictor='bc',lnsigmav_matrix=None,lnsigmau_matrix=None,
               lnmu_matrix=None,mu_matrix=None,cost=False,mpmath=False):
    if model in ('nhn','ntn','nexp','ng','nnak','nr'):
        k1 = data.shape[1]-1
        k2 = lnsigmav_matrix.shape[1]
        k3 = lnsigmau_matrix.shape[1]
        k4 = (mu_matrix.shape[1] if model in ('ntn') else lnmu_matrix.shape[1] if model in ('ng','nnak') else 0)
        k = k1 + k2 + k3 + k4
        y = data[:,0]
        X = data[:,1:k1+1]
        b = params[0:k1]
        epsilon=(y - X @ b)
        d1 = params[k1:k1+k2]
        d2 = params[k1+k2:k1+k2+k3]
        sigmav = np.exp(lnsigmav_matrix@d1)
        sigmau = np.exp(lnsigmau_matrix@d2)
        if model in ('ntn','ng','nnak'):
            if model == 'ntn':
                k4 = mu_matrix.shape[1]
                d3 = params[k1+k2+k3:k1+k2+k3+k4]
                mu = (mu_matrix@d3)
                lnmu = None
            elif model in ('ng','nnak'):
                k4 = lnmu_matrix.shape[1]
                mu = None
                d3 = params[k1+k2+k3:k1+k2+k3+k4]
                mu = np.exp(lnmu_matrix@d3)
        else:
            mu = None
        if cost:
            s = -1
        else:
            s = 1
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
            sigma = np.sqrt(2*sigmav**2+sigmau**2)
            z =  (s*epsilon*sigmau/sigmav)/sigma
            if predictor == 'bc':
                return (np.exp(1/2*(z+sigmav*sigmau/sigma)**2-1/2*z**2)*
                        (np.exp(-1/2*(z+sigmav*sigmau/sigma)**2)-np.sqrt(np.pi/2)*(z+sigmav*sigmau/sigma)*
                         scipy.special.erfc(1/np.sqrt(2)*(z+sigmav*sigmau/sigma)))/
                         (np.exp(-1/2*z**2)-np.sqrt(np.pi/2)*z*scipy.special.erfc(z/np.sqrt(2))))
            elif predictor == 'jlms':
                    return (np.exp(-2*sigmav*sigmau/sigma*
                            ((np.sqrt(np.pi/8)*(1+z**2)*scipy.special.erfc(z/np.sqrt(2))-z/2*np.exp(-z**2/2))/
                            (np.exp(-z**2/2)-np.sqrt(np.pi/2)*z*scipy.special.erfc(z/np.sqrt(2))))))
            elif predictor == 'mode':
                    return (np.exp(-sigmav*sigmau/(2*sigma)*(np.sqrt(z**2+4)-z)))
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
    x = (np.arange(n)*h) - (n*h/2)
    #x = np.linspace(epsilonmin, epsilonmax, n)
    s = 1/(h*n)
    t = 2*np.pi*s*(np.arange(n)-(n/2))
    sgn = np.ones(n)
    sgn[1::2] = -1
    if ((lnsigmav is None or np.all(lnsigmav == lnsigmav[0])) and
        (lnsigmau is None or np.all(lnsigmau == lnsigmau[0])) and
        (lnmu is None or np.all(lnmu == lnmu[0])) and
        (mu is None or np.all(mu == mu[0]))):
        lnsigmav = lnsigmav[0]
        lnsigmau = lnsigmau[0]
        lnmu = None if lnmu is None else lnmu[0]
        mu = None if mu is None else mu[0]
        cf_values = cf(t,lnsigmav,lnsigmau,lnmu,mu,model=model,cost=cost)
        p = np.log(s*np.abs(np.fft.fft(sgn*cf_values)))
        cs = scipy.interpolate.interp1d(x,p,kind='linear',fill_value="extrapolate")
        return cs(epsilon)
    else:
    #p = np.log(s*np.abs(np.fft.fft(sgn*cf(t,lnsigmav,lnsigmau,lnmu,mu,model=model,cost=cost))))
    #cs = scipy.interpolate.interp1d(x,p,kind='linear',fill_value="extrapolate")
    #return cs(epsilon)
        lnmu = np.zeros_like(epsilon) if lnmu is None else lnmu
        mu = np.zeros_like(epsilon) if mu is None else mu
    # p_list = []
    # cf_values = np.array([
    #    cf(t, lnsigmav[i], lnsigmau[i], lnmu[i], mu[i], model=model, cost=cost)
    #    for i in range(len(epsilon))
    #])
    # Compute FFT for all cf values in one operation
    # fft_values = np.fft.fft(sgn * cf_values, axis=1)
    # p = np.log(s * np.abs(fft_values))
    # Interpolate for all epsilon at once
    # interpolators = scipy.interpolate.interp1d(x, p, kind='linear', fill_value="extrapolate", axis=1)
    # return interpolators(epsilon)
        p_list = []
        for i in range(len(epsilon)):
            cf_values = cf(t, lnsigmav[i], lnsigmau[i], lnmu[i], mu[i], model=model, cost=cost)
            fft_values = np.fft.fft(sgn * cf_values)
            p = np.log(s * np.abs(fft_values))
            # Interpolate only for epsilon[i]
            cs = scipy.interpolate.interp1d(x, p, kind='linear', fill_value="extrapolate", axis=0)
            p_list.append(cs(epsilon[i]))
        return np.array(p_list)
    
def meanefficiency(params,model='nhn',lnsigmau_matrix=None,lnmu_matrix=None,mu_matrix=None,p1=0,p2=1,mpmath=False):
    if not isinstance(p2, (list, np.ndarray)):  # Ensure p2 is iterable
        p2 = [p2]
    results = []
    if model in ('nhn', 'ntn', 'nexp', 'ng', 'nnak', 'nr'):
        if model in ('nhn', 'nexp', 'nr'):
            if (lnsigmau_matrix is None or np.all(lnsigmau_matrix == lnsigmau_matrix[0])):
                k1 = 1
                sigmau = np.exp(params[-k1])
            else:
                k1 = lnsigmau_matrix.shape[1]
                sigmau = np.exp(lnsigmau_matrix @ params[-k1:])
        elif model in ('ng', 'nnak'):
            if (lnmu_matrix is None or np.all(lnmu_matrix == lnmu_matrix[0])):
                k2 = 1
                mu = np.exp(params[-k2])
            else:
                k2 = lnmu_matrix.shape[1]
                mu = np.exp(lnmu_matrix @ params[-k2:])
            if (lnsigmau_matrix is None or np.all(lnsigmau_matrix == lnsigmau_matrix[0])):
                k1 = 1
                sigmau = np.exp(params[-k1-k2])
            else:
                k1 = lnsigmau_matrix.shape[1]
                sigmau = np.exp(lnsigmau_matrix @ params[-k1-k2])
        elif model in ('ntn'):
            if (mu_matrix is None or np.all(mu_matrix == mu_matrix[0])):
                k2 = 1
                mu = params[-k2]
            else:
                k2 = mu_matrix.shape[1]
                mu = mu_matrix @ params[-k2:]
            if (lnsigmau_matrix is None or np.all(lnsigmau_matrix == lnsigmau_matrix[0])):
                k1 = 1
                sigmau = np.exp(params[-k1-k2])
            else:
                k1 = lnsigmau_matrix.shape[1]
                sigmau = np.exp(lnsigmau_matrix @ params[-k1-k2])

        for p2_value in p2:
            if p1 == 0 and p2_value == 1:
                if model == 'nhn':
                    results.append(np.exp(sigmau**2 / 2) * scipy.special.erfc(sigmau / np.sqrt(2)))
                elif model == 'nexp':
                    results.append(1 / (1 + sigmau))
                elif model == 'nr':
                    results.append((1 - np.sqrt(np.pi) * sigmau / 2 * np.exp((sigmau / 2)**2) * scipy.special.erfc(sigmau / 2)))
                elif model == 'ntn':
                    results.append((np.exp(sigmau**2 / 2 - mu) * 
                                    scipy.special.erfc((sigmau**2 - mu) / (np.sqrt(2) * sigmau)) /
                                    scipy.special.erfc(mu / (np.sqrt(2) * sigmau))))
                elif model == 'nnak':
                    results.append((2**mu / np.sqrt(np.pi) * scipy.special.gamma(mu + 1/2) *
                                    np.exp(sigmau**2 / (8 * mu)) * 
                                    scipy.special.pbdv(-2 * mu, sigmau / np.sqrt(2 * mu))[0]))
                elif model == 'ng':
                    results.append(1 / (1 + sigmau)**mu)
            elif p1 == 0 and 0 < p2_value < 1:
                if model == 'nnak':
                    k = 0
                    meaneff = 0
                    precision = np.finfo(float).eps
                    while True:
                        summand = (1 / scipy.special.factorial(k) * (-sigmau / np.sqrt(mu))**k *
                                   scipy.special.poch(mu, k / 2) *
                                   scipy.special.gammainc(mu + k / 2, scipy.special.gammaincinv(mu, p2_value)) / p2_value)
                        if np.abs(summand) < precision:
                            break
                        meaneff += summand
                        k += 1
                    results.append(meaneff)
                elif model == 'nr':
                    k = 0
                    meaneff = 0
                    precision = np.finfo(float).eps
                    while True:
                        summand = ((-sigmau)**k / scipy.special.factorial(k) * scipy.special.gamma(1 + k / 2) *
                                   scipy.special.gammainc(1 + k / 2, -np.log(1 - p2_value)) / p2_value)
                        if np.abs(summand) < precision:
                            break
                        meaneff += summand
                        k += 1
                    results.append(meaneff)
                elif model == 'nexp':
                    results.append((1 - (1 - p2_value)**(1 + sigmau)) / (p2_value * (1 + sigmau)))
                elif model == 'ng':
                    results.append(scipy.special.gammainc(mu, scipy.special.gammaincinv(mu, p2_value) * (1 + sigmau)) / 
                                   (p2_value * (1 + sigmau)**mu))
                elif model == 'ntn':
                    results.append((np.exp(sigmau**2 / 2 - mu) * 
                                    (scipy.special.erf(scipy.special.erfinv(p2_value - (1 + p2_value) *
                                                                             scipy.special.erf(mu / (np.sqrt(2) * sigmau))) +
                                                       sigmau / np.sqrt(2)) +
                                     scipy.special.erf((mu - sigmau**2) / (np.sqrt(2) * sigmau))) /
                                    (p2_value * scipy.special.erfc(mu / (np.sqrt(2) * sigmau)))))
                elif model == 'nhn':
                    results.append((np.exp(sigmau**2 / 2) * 
                                    (scipy.special.erf(scipy.special.erfinv(p2_value) + sigmau / np.sqrt(2)) - 
                                     scipy.special.erf(sigmau / (np.sqrt(2)))) / p2_value))
            else:
                raise ValueError("Invalid percentiles:", p1, p2_value)

        return np.array(results) if len(results) > 1 else results[0]
    else:
        raise ValueError("Unknown model:", model)
    
def lnlikelihood(params,data,model='nhn',cost=False,lnsigmav_matrix=None,lnsigmau_matrix=None,
                 lnmu_matrix=None,mu_matrix=None,mpmath=False,approximation=None,points=13,width=2):
    k1 = data.shape[1]-1
    k2 = lnsigmav_matrix.shape[1]
    k3 = lnsigmau_matrix.shape[1]
    k4 = (mu_matrix.shape[1] if model in ('ntn') else lnmu_matrix.shape[1] if model in ('ng','nnak') else 0)
    k = k1 + k2 + k3 + k4
    #params = np.zeros(k)
    y = data[:,0]
    X = data[:,1:k1+1] #remember 1:n is not inclusive of n, so this really gives 1:k
    b = params[0:k1] #see comment above
    epsilon=(y - X @ b)
    d1 = params[k1:k1+k2]
    lnsigmav = (lnsigmav_matrix@d1)
    d2 = params[k1+k2:k1+k2+k3]
    lnsigmau = (lnsigmau_matrix@d2)
    if model in ('ntn','ng','nnak'):
        if model == 'ntn':
            k4 = mu_matrix.shape[1]
            d3 = params[k1+k2+k3:k1+k2+k3+k4]
            mu = (mu_matrix@d3)
            lnmu = None
        elif model in ('ng','nnak'):
            k4 = lnmu_matrix.shape[1]
            mu = None
            d3 = params[k1+k2+k3:k1+k2+k3+k4]
            lnmu = (lnmu_matrix@d3)
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

def minuslnlikelihood(params,data,model='nhn',cost=False,lnsigmav_matrix=None,lnsigmau_matrix=None,
                      lnmu_matrix=None,mu_matrix=None,mpmath=False,approximation=None,points=13,width=2):
    minuslnlikelihood = -lnlikelihood(params,data,model,cost,lnsigmav_matrix,lnsigmau_matrix,lnmu_matrix,
                                      mu_matrix,mpmath,approximation,points,width)
    return minuslnlikelihood

def pcfd_complex(v,z):
    return np.complex128(complex(mp.re(mp.pcfd(v,z)),mp.im(mp.pcfd(v,z))))

def startvals(data,model='nhn',lnsigmav_matrix=None,lnsigmau_matrix=None,
              lnmu_matrix=None,mu_matrix=None,cost=False):
    y = data[:,0]
    k1 = data.shape[1]-1
    X = data[:,1:k1+1]
    n = X.shape[0]
    if lnsigmav_matrix is None:
        lnsigmav_matrix = np.ones((data.shape[0], 1))
    if lnsigmau_matrix is None:
        lnsigmau_matrix = np.ones((data.shape[0], 1))
    if lnmu_matrix is None:
        lnmu_matrix = np.ones((data.shape[0], 1))
    if mu_matrix is None:
        mu_matrix = np.ones((data.shape[0], 1))
    k2 = lnsigmav_matrix.shape[1]
    k3 = lnsigmau_matrix.shape[1]
    b_ols = (sm.OLS(y,X).fit()).params
    ehat_ols = (sm.OLS(y,X).fit()).resid
    if cost:
        s = -1
    else:
        s = 1
    m2 = 1/n*np.sum(ehat_ols**2)
    m3 = s/n*np.sum(ehat_ols**3)
    m4 = 1/n*np.sum(ehat_ols**4)
    if model in ('nhn','ntn','nnak'):
        sigmau_cols = np.cbrt(m3*np.sqrt(np.pi/2)/(1-4/np.pi))
        #sigmav_cols = np.sqrt(np.max((m2-(1-2/np.pi)*sigmau_cols**2,1.0e-20)))
        sigmav_cols = np.sqrt(np.maximum(m2-(1-2/np.pi)*sigmau_cols**2,sys.float_info.min))
        cons_cols = b_ols[-1] + sigmau_cols*np.sqrt(2/np.pi)
        if model == 'nhn':
            e_params = np.array(
                                [np.log(sigmav_cols)] + [0] * (k2 - 1) +
                                [np.log(sigmau_cols)] + [0] * (k3 - 1))
        elif model in ('ntn','nnak'):
            if model == 'ntn':
                mu = 0
                k4 = mu_matrix.shape[1]
                e_params =  np.array(
                                    [np.log(sigmav_cols)] + [0] * (k2 - 1) +
                                    [np.log(sigmau_cols)] + [0] * (k3 - 1) +
                                    [mu] + [0] * (k4 - 1))
            if model == 'nnak':
                lnmu = np.log(0.5)
                k4 = lnmu_matrix.shape[1]
                e_params =  np.array(
                                    [np.log(sigmav_cols)] + [0] * (k2 - 1) +
                                    [np.log(sigmau_cols)] + [0] * (k3 - 1) +
                                    [lnmu] + [0] * (k4 - 1))
    elif model == 'nexp':
        sigmau_cols = np.cbrt(-m3/2)
        #sigmav_cols = np.sqrt(np.maximum(m2-sigmau_cols**2,1.0e-20))
        sigmav_cols = np.sqrt(np.maximum(m2-sigmau_cols**2,sys.float_info.min))
        cons_cols = b_ols[-1] + sigmau_cols
        e_params =  np.array(
                            [np.log(sigmav_cols)] + [0] * (k2 - 1) +
                            [np.log(sigmau_cols)] + [0] * (k3 - 1))
    elif model == 'ng':
        sigmau_cols = -(m4-3*m2**2)/(3*m3)
        mu_cols = -m3/(2*sigmau_cols**3)
        #sigmav_cols = np.maximum(np.sqrt((m2-mu_cols*sigmau_cols**2,1.0e-20)))
        sigmav_cols = np.sqrt(np.maximum(m2-mu_cols*sigmau_cols**2,sys.float_info.min))
        cons_cols = b_ols[-1]+ mu_cols*sigmau_cols
        k4 = lnmu_matrix.shape[1]
        e_params =  np.array(
                            [np.log(sigmav_cols)] + [0] * (k2 - 1) +
                            [np.log(sigmau_cols)] + [0] * (k3 - 1) +
                            [np.log(mu_cols)] + [0] * (k4 - 1))
    elif model == 'nr':
        sigmau_cols = np.cbrt(-4*m3/(np.sqrt(np.pi)*(np.pi-3)))
        #sigmav_cols = np.sqrt(np.maximum(m2 - sigmau_cols**2*(4-np.pi)/4,1.0e-20))
        sigmav_cols = np.sqrt(np.maximum(m2-sigmau_cols**2*(4-np.pi)/4,sys.float_info.min))
        cons_cols = b_ols[-1] + sigmau_cols*np.sqrt(np.pi)/2
        e_params =  np.array(
                            [np.log(sigmav_cols)] + [0] * (k2 - 1) +
                            [np.log(sigmau_cols)] + [0] * (k3 - 1))
    b_cols = np.append(b_ols[0:-1],cons_cols)
    theta_cols = np.append(b_cols,e_params,axis=None)
    return theta_cols

class Frontier:
    def __init__(self,lnlikelihood,frontier,lnsigmav_matrix,lnsigmav,lnsigmau_matrix,
                 lnsigmau,lnmu_matrix,lnmu,mu_matrix,mu,k1,k2,k3,k4,theta,score,hess_inv,
                 iterations,func_evals,score_evals,status,success,message,yhat,
                 residual,mean_eff,supra_pc_mean_eff,eff_bc,eff_jlms,eff_mode,model):
        self.model = model
        self.lnlikelihood = lnlikelihood
        self.frontier_eq = frontier
        self.lnsigmav_matrix = lnsigmav_matrix
        self.lnsigmav_eq = lnsigmav
        self.lnsigmau_matrix = lnsigmau_matrix
        self.lnsigmau_eq = lnsigmau
        self.lnmu_matrix = lnmu_matrix
        self.lnmu_eq = lnmu
        self.mu_matrix = mu_matrix
        self.mu_eq = mu
        self.theta = theta
        self.theta_se = np.sqrt(np.diag(hess_inv))
        self.theta_pval =  2*norm.cdf(-abs(theta/np.sqrt(np.diag(hess_inv))))
        self.theta_star = calculate_stars(2*norm.cdf(-abs(theta/
                                                          np.sqrt(np.diag(hess_inv)))))
        self.beta = theta[0:k1]
        self.beta_se = np.sqrt(np.diag(hess_inv))[0:k1]
        self.beta_pval =  2*norm.cdf(-abs(theta[0:k1]/np.sqrt(np.diag(hess_inv))[0:k1]))
        self.beta_star = calculate_stars(2*norm.cdf(-abs(theta[0:k1]/
                                                         np.sqrt(np.diag(hess_inv))[0:k1])))
        if (self.lnsigmav_matrix is None or np.all(self.lnsigmav_matrix == self.lnsigmav_matrix[0])):
            self.sigmav = np.exp(theta[k1])
            self.sigmav_se = np.sqrt(np.exp(theta[k1])**2*np.diag(hess_inv)[k1])
        else:
            self.deltav = theta[k1:k1+k2]
            self.deltav_se = np.sqrt(np.diag(hess_inv))[k1:k1+k2]
            self.deltav_pval =  2*norm.cdf(-abs(theta[k1:k1+k2]/np.sqrt(np.diag(hess_inv))[k1:k1+k2]))
            self.deltav_star = calculate_stars(2*norm.cdf(-abs(theta[k1:k1+k2]/
                                                         np.sqrt(np.diag(hess_inv))[k1:k1+k2])))
        if (self.lnsigmau_matrix is None or np.all(self.lnsigmau_matrix == self.lnsigmau_matrix[0])):
            self.sigmau = np.exp(theta[k1+k2])
            self.sigmau_se = np.sqrt(np.exp(theta[k1+k2])**2*np.diag(hess_inv)[k1+k2])
        else:
            self.deltau = theta[k1+k2:k1+k2+k3]
            self.deltau_se = np.sqrt(np.diag(hess_inv))[k1+k2:k1+k2+k3]
            self.deltau_pval =  2*norm.cdf(-abs(theta[k1+k2:k1+k2+k3]/
                                                np.sqrt(np.diag(hess_inv))[k1+k2:k1+k2+k3]))
            self.deltau_star = calculate_stars(2*norm.cdf(-abs(theta[k1+k2:k1+k2+k3]/
                                                         np.sqrt(np.diag(hess_inv))[k1+k2:k1+k2+k3])))
        if model == 'ntn':
            if (self.mu_matrix is None or np.all(self.mu_matrix == self.mu_matrix[0])):
                self.mu = theta[k1+k2+k3]
                self.mu_se = np.sqrt(np.diag(hess_inv))[k1+k2+k3]
            else:
                self.deltamu = theta[k1+k2+k3:k1+k2+k3+k4]
                self.deltamu_se = np.sqrt(np.diag(hess_inv))[k1+k2+k3:k1+k2+k3+k4]
                self.deltamu_pval =  2*norm.cdf(-abs(theta[k1+k2+k3:k1+k2+k3+k4]/
                                                     np.sqrt(np.diag(hess_inv))[k1+k2+k3:k1+k2+k3+k4]))
                self.deltamu_star = calculate_stars(2*norm.cdf(-abs(theta[k1+k2+k3:k1+k2+k3+k4]/
                                                                    np.sqrt(np.diag(hess_inv))[k1+k2+k3:k1+k2+k3+k4])))
        if model in ('ng','nnak'):
            if (self.lnmu_matrix is None or np.all(self.lnmu_matrix == self.lnmu_matrix[0])):
                self.mu = np.exp(theta[k1+k2+k3])
                self.mu_se = np.sqrt(np.exp(theta[k1+k2+k3])**2*np.diag(hess_inv)[k1+k2+k3])
            else:
                self.deltamu = theta[k1+k2+k3:k1+k2+k3+k4]
                self.deltamu_se = np.sqrt(np.diag(hess_inv))[k1+k2+k3:k1+k2+k3+k4]
                self.deltamu_pval =  2*norm.cdf(-abs(theta[k1+k2+k3:k1+k2+k3+k4]/
                                                     np.sqrt(np.diag(hess_inv))[k1+k2+k3:k1+k2+k3+k4]))
                self.deltamu_star = calculate_stars(2*norm.cdf(-abs(theta[k1+k2+k3:k1+k2+k3+k4]/
                                                                    np.sqrt(np.diag(hess_inv))[k1+k2+k3:k1+k2+k3+k4])))
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

    def __repr__(self):
        top_fields = ["model", "lnlikelihood", "status", "message", "iterations" , "func_evals",
                       "score_evals", "theta", "theta_se", "theta_pval", "theta_star","frontier_eq",
                       "beta", "beta_se", "beta_pval", "beta_star"]
        if (self.lnsigmav_matrix is None or np.all(self.lnsigmav_matrix == self.lnsigmav_matrix[0])):
            sigmav_fields = ["sigmav", "sigmav_se"]
        else:
            sigmav_fields = ["lnsigmav_eq", "deltav", "deltav_se", "deltav_pval", "deltav_star"]
        if (self.lnsigmau_matrix is None or np.all(self.lnsigmau_matrix == self.lnsigmau_matrix[0])):
            sigmau_fields = ["sigmau", "sigmau_se"]
            mean_fields = ["mean_eff", "supra_pc_mean_eff"]
        else:
            sigmau_fields = ["lnsigmau_eq", "deltau", "deltau_se", "deltau_pval", "deltau_star"]
            mean_fields = []
        mu_fields = [] # Default empty mu_fields models
        if self.model in ('ntn'):
            if self.mu_matrix is None or np.all(self.mu_matrix == self.mu_matrix[0]):
                mu_fields = ["mu", "mu_se"]
            else:
                mu_fields = ["mu_eq", "deltamu", "deltamu_se", "deltamu_pval", "deltamu_star"]
                mean_fields = []
        elif self.model in ('ng', 'nnak'):
            if self.lnmu_matrix is None or np.all(self.lnmu_matrix == self.lnmu_matrix[0]):
                mu_fields = ["mu", "mu_se"]
            else:
                mu_fields = ["lnmu_eq", "deltamu", "deltamu_se", "deltamu_pval", "deltamu_star"]
                mean_fields = []
        mid_fields_by_model = {
            "nhn": [],
            "nexp": [],
            "nr": [],
            "ntn": mu_fields,
            "ng": mu_fields,
            "nnak": mu_fields}
        bottom_fields = ["eff_bc", "eff_jlms", "eff_mode", "residual",
                         "yhat", "score", "hess_inv"]
        all_fields = (top_fields + sigmav_fields + sigmau_fields + mid_fields_by_model.get(self.model, [])
                      + mean_fields + bottom_fields)
        repr_string = "\n".join(
            f"{field}: {getattr(self, field.split(' ')[0].strip('()'), None)}" for field in all_fields)
        return repr_string

def estimate(data,frontier=None,model='nhn',cost=False,lnsigmav=None,lnsigmau=None,lnmu=None,mu=None,
             startingvalues=None,algorithm='BFGS',tol=1e-4,mpmath=False,approximation=None,points=13,width=2):
    """
    Estimate a Stochastic Frontier Model

    This function estimates a stochastic frontier model using maximum likelihood estimation. 
    The user can specify the model type and customise several options for optimisation.

    Supported models include:
    - Normal-Half Normal (N-HN).
    - Normal-Exponential (N-EXP).
    - Normal-Truncated Normal (N-TN).
    - Normal-Gamma (N-G).
    - Normal-Rayleigh (N-R).
    - Normal-Nakagami (N-NAK).

    Supported approximation methods:
    - Inverse fast Fourier transformation (FFT) of the characteristic function.

    Parameters:
        data (ndarray or DataFrame): A 2D array or pandas DataFrame containing the data.
        frontier (str, optional): A formula specifying the frontier to be estimated using Wilkinson
                        notation, e.g. 'y ~ x1 + x2 + x3'. The dependent variable is placed on the
                        left-hand side, separated from the covariates by '~'. Covariates are added
                        using '+'. Interactions can be specified with '*', and transformations can
                        be applied directly (e.g., 'np.log(x1)'). A constant term is included by
                        default, but may be removed by including '-1' in the formula.
                        Note: If specified, the `data` argument must be a pandas DataFrame. If left
                        blank, it is assumed that the first column of `data` is the dependent
                        variable, and all remaining columns are independent variables (in this
                        case, the user must include a column of ones if an intercept is wanted.)
        model (str, optional): Specifies the distributional assumptions to use. Default is 'nhn'.
                        Valid options: 'nhn', 'ntn', 'nexp', 'ng', 'nr', 'nnak'.
        cost (bool, optional): If True, estimates a cost frontier; otherwise, a production frontier.
                        Default is False.
        lnsigmav, lnsigmau, lnmu, mu (str, optional): Optional formulae for distributional
                        parameters. These allow the user to model the distributional parameters as
                        functions of covariates. For example, specifying `lnsigmav="z1"` means
                        that ln(σ_U) = δ_0 + δ_1*z.
                        - `lnsigmav`: the natural logarithm of σ_V.
                        - `lnsigmau`: the natural logarithm of σ_U.
                        - `lnmu`: the natural logarithm of μ. Only available for the 'ng' and 'nnak'
                        models.
                        - `mu`: μ. Only available for the 'ntn' model.
        startingvalues (array-like, optional): Initial values for the optimisation.
                        The default is to use starting values provided by FronPy's `startvals`
                        function.
        algorithm (str, optional): Optimisation algorithm to use. Default is 'BFGS'.
        tol (float): Tolerance for convergence. Default is 1e-4.
        approximation (str, optional): Specifies a method of approximating the density and
                        log-likelihood functions. The default is to compute the exact
                        expressions. Use 'fft' for the inverse fast Fourier transformation of
                        the characteristic function.
        points (int, optional): An option to specify when using the FFT approaximation method.
                        Specifies the a power of 2 which gives the number of grid points at
                        which to approximate the density. The default is 13, giving
                        2^13=8192 grid points.
        width (float, optional): An option to specify when using the FFT approaximation
                        method. The grid points at which he density is approximated are then
                        distributed across the range [-width*max(|ε|)/g,width*max(|ε|)/g], where g
                        is the number of grid points. The default is 2.

    Returns:
        Frontier: An object of class `Frontier` containing results of the estimation:
            - `model`: The specified model type.
            - `frontier_eq`: The frontier formula.
            - `lnlikelihood`: Log-likelihood value of the fitted model.
            - `k`: Number of frontier parameters in the model.
            - `beta`: Frontier parameter estimates.
            - `beta_se`: Standard errors for the frontier parameter estimates.
            - `beta_pval`: P-values for the frontier parameter estimates.
            - `beta_star`: Significance stars for the frontier parameter estimates.
                        Note: * 0.05 < p =< 0.1, ** 0.01 < p =< 0.05, *** p <= 0.01.
            - `sigmav`: The estimated value of σ_V, the scale parameter of the
                        distribution of V.
            - `sigmav`: The standard error for the estimate of σ_V.
            - `lnsigmav_eq': The formula for lnσ_V.
            - `sigmau`: The estimated value of σ_U, the scale parameter of the
                        distribution of U.
            - `sigmau`: The standard error for the estimate of σ_U.
            - `lnsigmau_eq': The formula for lnσ_U.
            - `mu`: The estimated value of the shape parameter of the distribution of U.
                        Note: only applies to the N-G, N-NAK, and N-TN models.
            - `mu_se`: The standard error for the estimate of μ.
                        Note: only applies to the N-G, N-NAK, and N-TN models.
            - `lnmu_eq': The formula for lnμ.
                        Note: only applies to the N-G and N-NAK models.
            - `lnmu_eq': The formula for μ.
                        Note: only applies to the N-TN model.
            - `theta`: The complete vector of parameter estimates, including both
                        frontier parameters and distributional parameters.
                        Note: distributional parameters are included in the form that
                        they are estimated, e.g. instead of σ_V, ln(σ_V) is
                        included to estimated to ensure that σ_V>0.
            - `theta_pval`: P-values for the complete vector of parameter estimates.
            - `theta_star`: Significance stars for the complete vector of parameter
                        estimates.
                        Note: * 0.05 < p =< 0.1, ** 0.01 < p =< 0.05, *** p <= 0.01. 
            - `mean_eff`: Unconditional mean efficiency, 𝔼[exp(-U)].
            - `supra_pc_mean_eff`: Supra-percentile unconditional mean efficiencies,
                        𝔼[exp(-U)|U<F_U^-1(p)], for various values of p.
            - `score`: Score vector at the solution.
            - `hess_inv`: Inverse of the Hessian matrix.
            - `iterations`: Number of iterations performed.
            - `func_evals`: Number of function evaluations.
            - `score_evals`: Number of score evaluations.
            - `status`: Termination status of the optimiser.
            - `success`: Boolean indicating if optimisation was successful.
            - `message`: Message describing the termination status.
            - `yhat`: Predicted values based on the estimated frontier.
            - `residual`: Estimated residuals.
            - `eff_bc`: Efficiency predictions using the Battese-Coelli (1988)
                        predictor, 𝔼[exp(-U)|E=e].
            - `eff_jlms`: Efficiency predictions using the Jondrow, Lovell, Materov,
                        and Schmidt (1982) predictor, exp(-𝔼[U|E=e]).
            - `eff_mode`: Efficiency predictions using the conditional mode
                        predictor, exp(-Mode(U|E=e)).
        (Note: some elements may not be displayed for certain model specifications,
        but are still accessible.)

    Raises:
        ValueError: If an unsupported model is specified.

    Example:
    --------
    >>> electricity = fronpy.dataset('electricity.csv')
    >>> nhnmodel_electricity = fronpy.estimate(electricity,model='nhn',cost=True)
    Optimization terminated successfully.
         Current function value: -66.864907
         Iterations: 19
         Function evaluations: 208
         Gradient evaluations: 26
    >>> print(nhnmodel_electricity.beta)
    [3.7348835  0.96586287 0.03029115 0.26058867 0.05531295]
    """
    lnsigmav_matrix = lnsigmau_matrix = lnmu_matrix = mu_matrix = np.ones((data.shape[0], 1))
    k2 = k3 = k4 = 1
    if any(arg is not None for arg in [frontier, lnsigmav, lnsigmau, lnmu, mu]):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be a pandas DataFrame when using formulae.")
        if lnsigmav is not None:
            lnsigmav_matrix = Formula(lnsigmav).get_model_matrix(data).to_numpy()
            k2 = lnsigmav_matrix.shape[1]
        if lnsigmau is not None:
            lnsigmau_matrix = Formula(lnsigmau).get_model_matrix(data).to_numpy()
            k3 = lnsigmau_matrix.shape[1]
        if lnmu is not None:
            if model not in ('ng', 'nnak'):
                raise ValueError(f"lnmu option not available for model {model}")
            else:
                lnmu_matrix = Formula(lnmu).get_model_matrix(data).to_numpy()
                k4 = lnmu_matrix.shape[1]
        if mu is not None:
            if model not in ('ntn'):
                raise ValueError(f"mu option not available for model {model}")
            else:
                mu_matrix = Formula(mu).get_model_matrix(data).to_numpy()
                k4 = mu_matrix.shape[1]
        y, X = Formula(frontier).get_model_matrix(data)
        data = np.hstack([y, X])
    else:
        if frontier is None:
            frontier = 'No formula supplied.'
        if isinstance(data, pd.DataFrame):
            data = np.asarray(data)
        else:
            pass
    if model in ('nhn','ntn','nexp','ng','nnak','nr'):
        n = data.shape[0]
        k1 = data.shape[1] - 1
        if startingvalues is None:
            startingvalues = startvals(data,model,lnsigmav_matrix=lnsigmav_matrix,
                                       lnsigmau_matrix=lnsigmau_matrix,
                                       lnmu_matrix=lnmu_matrix,mu_matrix=mu_matrix,
                                       cost=cost)
        result = minimize(minuslnlikelihood,startingvalues,tol=tol*n,
                          method=algorithm,args=(data,model,cost,lnsigmav_matrix,
                                                 lnsigmau_matrix,lnmu_matrix,mu_matrix,
                                                 mpmath,approximation,points,width),
                          options={'disp': 2})
        frontier = Frontier(lnlikelihood = -result.fun,
                            frontier = frontier,
                            lnsigmav_matrix = lnsigmav_matrix,
                            lnsigmav = lnsigmav,
                            lnsigmau_matrix = lnsigmau_matrix,
                            lnsigmau = lnsigmau,
                            lnmu_matrix = lnmu_matrix,
                            lnmu = lnmu,
                            mu_matrix = mu_matrix,
                            mu = mu,
                            k1 = k1,
                            k2 = k2,
                            k3 = k3,
                            k4 = k4,
                            theta = result.x,
                            mean_eff = meanefficiency(result.x,model,None,None,None,0,1,False),
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
                                                                meanefficiency(params=result.x,
                                                                                 model=model,
                                                                                 lnsigmau_matrix=lnsigmau_matrix,
                                                                                 lnmu_matrix=lnmu_matrix,
                                                                                 mu_matrix=mu_matrix,
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
                            eff_bc = efficiency(result.x,data,model,predictor='bc',
                                                lnsigmav_matrix=lnsigmav_matrix,
                                                lnsigmau_matrix=lnsigmau_matrix,
                                                lnmu_matrix=lnmu_matrix,mu_matrix=mu_matrix,
                                                cost=cost),
                            eff_jlms = efficiency(result.x,data,model,predictor='jlms',
                                                  lnsigmav_matrix=lnsigmav_matrix,
                                                  lnsigmau_matrix=lnsigmau_matrix,
                                                  lnmu_matrix=lnmu_matrix,mu_matrix=mu_matrix,
                                                  cost=cost),
                            eff_mode = efficiency(result.x,data,model,predictor='mode',
                                                  lnsigmav_matrix=lnsigmav_matrix,
                                                  lnsigmau_matrix=lnsigmau_matrix,
                                                  lnmu_matrix=lnmu_matrix,mu_matrix=mu_matrix,
                                                  cost=cost),
                            model = model)
        return(frontier)
    else:
        raise ValueError("Unknown model:", model)