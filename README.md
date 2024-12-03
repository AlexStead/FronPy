# FronPy
## Python package for stochastic frontier analysis
This package was built was to facilitate direct maximum likelihood estimation of the normal-gamma and normal-Nakagami stochastic frontier models using closed-form expressions for the log-likelihood functions and efficiency predictors in terms of the parabolic cylinder function, as explored in the paper:

- Stead AD. 2024. Maximum likelihood estimation of normal-gamma and normal-Nakagami stochastic frontier models. _Journal of Productivity Analysis_. DOI: [10.1007/s11123-024-00742-2](https://doi.org/10.1007/s11123-024-00742-2)

The package however also includes options to estimate other stochastic frontier specifications, and may be of more general use to anyone who wishes to use Python for stochastic frontier analysis. All results in the paper were obtained using this package.

The package produces parameter estimates, standard errors, log-likelihoods, efficiency predictors, and more in a convenient format.

 Currently the package is limited to models of the form
$$y_i=\boldsymbol{x'_i\beta}+E_i, \qquad
    E_i=V_i-sU_i, \qquad V_i\sim N\left(0,\sigma_V^2\right),$$
where $U_i$ may be either $N^+\left(0,\sigma_U^2\right)$ (half normal), $N^+\left(\mu,\sigma_U^2\right)$ (truncated normal), $\mathrm{Gamma}\left(\mu,\sigma_U\right)$, or $\mathrm{Exponential}\left(\sigma_U\right)$; N.B. in the latter two cases $\sigma_U$ is a scale parameter. The package may be extended in future in order to accomodate additional specifications.

## Installation
Provided you have the `git` package installed, you can install `FronPy` by entering: 
```python
pip install git+https://github.com/AlexStead/FronPy.git
```

## Getting started
The code block below estimates the normal-gamma stochastic frontier model, and demonstrates the package's basic syntax:
```python
import fronpy
import numpy as np
electricity = fronpy.dataset('electricity.csv')

nexpmodel = fronpy.estimate(electricity,model='nexp',cost=True)
ngmodel = fronpy.estimate(electricity,model='ng',startingvalues=np.append(nexpmodel.theta,0),cost=True)
ngmodel
```
In the final line, `ngmodel` produces rather a lot of output, but is useful to see all of the outputs produced by the `fronpy.estimate`. For example, we can see that `ngmodel.lnlikelihood` would give us the log-likelihood, `ngmodel.beta` would give us the estimated frontier parameters, `ngmodel.eff_bc` would give us $\mathbb{E}[e^{-U_i}|E_i=\varepsilon_i]$ for each observation, and so on. Note that currently, `fronpy.estimate` requires a `numpy.ndarray` with rows representing observations and columns representing variables, where:
- all data are numeric.
- the first column contains the dependent variable.
- columns 2,...,n contain the independent variables, including a constant if desired.
- there are no missing, `NaN`, `inf`, `-inf` or similarly invalid values.
The package then assumes that the frontier is linear in its parameters, e.g. Cobb-Douglas or translog; note that any interactions must be included as columns in the `numpy.ndarray`. The package may be generalised in future to allow for arbitrary functional forms.
