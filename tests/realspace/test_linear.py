import pytest
import numpy as np
from velocitypy.realspace.linear import density_correlation, radial_mean

wavemode_k = np.arange(0.001, 100, 0.01)
power_spectrum_ones = np.ones(len(wavemode_k))
power_spectrum_zeros = np.zeros(len(wavemode_k))


def test_correlation():
    assert density_correlation(1, wavemode_k, power_spectrum_zeros) == 0
