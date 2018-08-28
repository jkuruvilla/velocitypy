# -*- coding: utf-8 -*-

"""
Velocity Statistics in Cosmology
================================
velocitypy is a library to calculate the velocity statistics, i.e.
including one-point velocity dispersion, pairwise velocity quantities,
velocity correlation, etc. It is built using :code: 'numpy' and :code:
'scipy'.
"""

__author__ = "Joseph Kuruvilla"
__email__ = "joseph.k@uni-bonn.de"
__version__ = "0.0.1"

from .realspace import linear

__all__ = ["linear"]
