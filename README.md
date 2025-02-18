# FronPy
# Version 1.0.2
## Python package for stochastic frontier analysis
This package was built was to facilitate direct maximum likelihood estimation of the normal-gamma and normal-Nakagami stochastic frontier models using closed-form expressions for the log-likelihood functions and efficiency predictors in terms of the parabolic cylinder function, as explored in the paper:

- Stead AD. 2024. Maximum likelihood estimation of normal-gamma and normal-Nakagami stochastic frontier models. _Journal of Productivity Analysis_. DOI: [10.1007/s11123-024-00742-2](https://doi.org/10.1007/s11123-024-00742-2)

The package however also includes options to estimate other stochastic frontier specifications, and may be of more general use to anyone who wishes to use Python for stochastic frontier analysis. All results in the paper were obtained using this package.

The package produces parameter estimates, standard errors, log-likelihoods, efficiency predictors, and more in a convenient format.

Currently the package is limited to models of the form
$$y\_i=\boldsymbol{x'\_i\beta}+E\_i, \qquad
    E\_i=V\_i-sU\_i, \qquad V\_i\sim N\left(0,\sigma\_V^2\right),$$
where $U\_i$ may be either $N^+\left(0,\sigma\_U^2\right)$ (half normal), $N^+\left(\mu,\sigma\_U^2\right)$ (truncated normal), $\mathrm{Rayleigh}\left(\sigma\_U/2\right)$, $\mathrm{Nakagami}\left(\mu,\sigma\_U\right)$ (truncated normal), $\mathrm{Gamma}\left(\mu,\sigma\_U\right)$, and $\mathrm{Exponential}\left(\sigma\_U\right)$; N.B. in the latter three cases $\sigma_U$ is a scale parameter.

All distributional parameters are assumed to be scalars by default, but may be modelled as a function of a vector of covariates, e.g. $\ln\sigma\_{vi}=\boldsymbol{z'\_{vi}\delta\_v}$, or $\ln\sigma\_{ui}=\boldsymbol{z'\_{ui}\delta\_u}$, or $\ln\mu\_{i} = \boldsymbol{z'\_{\mu i}\delta\_\mu}$ (gamma and Nakagami models only), or $\mu\_{i} = \boldsymbol{z'\_{\mu i}\delta\_\mu}$ (truncated normal model only), or combinations of these. 


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

nexpmodel = fronpy.estimate(electricity_df,frontier='lnc~np.log(q)+I(np.log(q)**2)+lnw+lnr',cost=True,model='nexp')
ngmodel = fronpy.estimate(electricity_df,frontier='lnc~np.log(q)+I(np.log(q)**2)+lnw+lnr',cost=True,model='ng',
                          startingvalues=np.append(nexpmodel.theta,[0]))
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
