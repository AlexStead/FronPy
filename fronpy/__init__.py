"""
FronPy: A Python package for frontier analysis

This package was built was to facilitate direct maximum likelihood estimation of the
normal-gamma and normal-Nakagami stochastic frontier models using closed-form
expressions for the log-likelihood functions and efficiency predictors in terms of
the parabolic cylinder function, as explored in the paper:

    Stead AD. 2024. Maximum likelihood estimation of normal-gamma and normal-Nakagami
    stochastic frontier models. Journal of Productivity Analysis.
    https://doi.org/10.1007/s11123-024-00742-2

The package however also includes options to estimate other stochastic frontier
specifications, and is useful for anyone who wishes to use Python for stochastic
frontier analysis. All results in the above paper were obtained using this package.

The package produces parameter estimates, standard errors, log-likelihoods,
efficiency predictors, and more in a convenient format.

Example:
--------
>>> import fronpy
>>> electricity = fronpy.dataset('electricity.csv')
>>> nhnmodel = fronpy.estimate(electricity,model='nhn',cost=True)
Optimization terminated successfully.
         Current function value: -66.864907
         Iterations: 19
         Function evaluations: 208
         Gradient evaluations: 26
"""
__version__ = "1.0.2"
from . import funcs

__all__ = [
    'funcs',
]

from .funcs import estimate
from .funcs import dataset
from .funcs import meanefficiency

def launch_gui():
    from .gui import launch_gui as gui_launch
    gui_launch()