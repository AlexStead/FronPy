# FronPy
## Python package for stochastic frontier analysis
The main purpose for which this package was built was in order to facilitate maximum likelihood estimation of the normal-gamma stochastic frontier model using a closed-form expression for the log-likelihood function in terms of the parabolic cylinder function. Efficiency predictors for the model are also available in closed-form using the parabolic cylinder function. To our knowledge, this is the first such implementation of the normal-gamma model using direct maximum likelihood estimation; other implementations use various approaches to approximate the log-likelihood - see the working paper cited below. Implementations of the normal-half normal, normal-truncated normal, and normal-exponential models are also included for comparison. The package produces parameter estimtes, standard errors, log-likelihoods, efficiency predictors, and more in a convenient format.

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
electricity = fronpy.utils.dataset('electricity.csv')

nexpmodel = fronpy.estimate(electricity,model='nexp',cost=True)
ngmodel = fronpy.estimate(electricity,model='ng',startingvalues=np.append(nexpmodel.theta,0),cost=True)
ngmodel
```

## Linked publications
- Stead AD. 2023. Maximum likelihood estimation of the normal-gamma stochastic frontier model. Working paper, University of Leeds.
