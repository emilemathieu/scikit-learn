"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from ._gaussian_mixture import GaussianMixture
from ._bayesian_mixture import BayesianGaussianMixture
from ._kent_mixture import KentMixture

__all__ = ["GaussianMixture", "BayesianGaussianMixture", "KentMixture"]
