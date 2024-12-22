# FronPy
# Version 1.0.0
## Python package for stochastic frontier analysis
This package was built was to facilitate direct maximum likelihood estimation of the normal-gamma and normal-Nakagami stochastic frontier models using closed-form expressions for the log-likelihood functions and efficiency predictors in terms of the parabolic cylinder function, as explored in the paper:

- Stead AD. 2024. Maximum likelihood estimation of normal-gamma and normal-Nakagami stochastic frontier models. _Journal of Productivity Analysis_. DOI: [10.1007/s11123-024-00742-2](https://doi.org/10.1007/s11123-024-00742-2)

The package however also includes options to estimate other stochastic frontier specifications, and may be of more general use to anyone who wishes to use Python for stochastic frontier analysis. All results in the paper were obtained using this package.

The package produces parameter estimates, standard errors, log-likelihoods, efficiency predictors, and more in a convenient format.

Currently the package is limited to models of the form
$$y_i=\boldsymbol{x'_i\beta}+E_i, \qquad
    E_i=V_i-sU_i, \qquad V_i\sim N\left(0,\sigma_V^2\right),$$
where $U_i$ may be either $N^+\left(0,\sigma_U^2\right)$ (half normal), $N^+\left(\mu,\sigma_U^2\right)$ (truncated normal), $\mathrm{Rayleigh}\left(\sigma_U/2\right)$, $\mathrm{Gamma}\left(\mu,\sigma_U\right)$, or $\mathrm{Exponential}\left(\sigma_U\right)$; N.B. in the latter two cases $\sigma_U$ is a scale parameter.

All distributional parameters are assumed to be scalars by default, but may be modelled as a function of a vector of covariates, $z_i$, e.g. $\ln\sigma_{vi} = \boldsymbol{z'_{vi}\delta_v}$, $\ln\sigma_{ui} = \boldsymbol{z'_{ui}\delta_u}$, $\ln\mu_{i} = \boldsymbol{z'_{\mu i}\delta_\mu}$ (gamma and Nakagami models only), $\mu_{i} = \boldsymbol{z'_{\mu i}\delta_\mu}$ (truncated normal model only). 


The package may be extended in future in order to accomodate additional specifications.

## Installation
Provided you have the `git` package installed, you can install `FronPy` by entering: 
```python
pip install git+https://github.com/AlexStead/FronPy.git
```

## Getting started
The code block below prints the help text explaining the use of the fronpy.dataset() and fronpy.estimate() functions:
```python
import fronpy

help(fronpy.dataset)
help(fronpy.estimate)
```

The code block below loads a built-in dataset, does some basic data transformation, estimates the normal-gamma stochastic frontier model, and demonstrates the package's basic syntax:
```python
import fronpy
import numpy as np
electricity_df = fronpy.dataset('electricity_df.csv',dataframe=True)

electricity_df[f'lnc'] = np.log((electricity_df['cost']/electricity_df['cost'].mean())/(electricity_df['fprice']/electricity_df['fprice'].mean()))
electricity_df[f'q'] = electricity_df['output'] / electricity_df['output'].mean()
electricity_df[f'lnq'] = np.log(electricity_df['output'] / electricity_df['output'].mean())
electricity_df[f'lnw'] = np.log((electricity_df['lprice']/electricity_df['lprice'].mean())/(electricity_df['fprice']/electricity_df['fprice'].mean()))
electricity_df[f'lnr'] = np.log((electricity_df['cprice']/electricity_df['cprice'].mean())/(electricity_df['fprice']/electricity_df['fprice'].mean()))

nexpmodel = fronpy.estimate(electricity,cost=True,model='nexp')
ngmodel = fronpy.estimate(electricity_df,frontier='lnc~np.log(q)+I(np.log(q)**2)+lnw+lnr',cost=True,model='ng',
                          startingvalues=np.append(nexpmodel.theta,[0,0]))
ngmodel
```

In the final line, `ngmodel` produces rather a lot of output, but is useful to see all of the outputs produced by the `fronpy.estimate`. For example, we can see that `ngmodel.lnlikelihood` would give us the log-likelihood, `ngmodel.beta` would give us the estimated frontier parameters, `ngmodel.eff_bc` would give us $\mathbb{E}[e^{-U_i}|E_i=\varepsilon_i]$ for each observation, and so on. Note that currently, `fronpy.estimate` requires a `numpy.ndarray` with rows representing observations and columns representing variables, where:
- all data are numeric.
- the first column contains the dependent variable.
- columns 2,...,n contain the independent variables, including a constant if desired.
- there are no missing, `NaN`, `inf`, `-inf` or similarly invalid values.

Alternatively, the following command may be used to launch a graphical user interface. Note that this is very rudimentary, and needs further development to incorporate all of the options available using the command line interface:
```python
import fronpy
fronpy.launch_gui()
```