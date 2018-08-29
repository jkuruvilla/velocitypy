import pytest
import numpy as np
from velocitypy.realspace.linear import density_correlation, radial_mean

wavemode_k = np.arange(0.001, 100, 0.0001)
power_spectrum_ones = np.ones(len(wavemode_k))
power_spectrum_zeros = np.zeros(len(wavemode_k))
radial_bin_small = np.array([1.])
radial_bin_large = np.array([120.])


def test_correlation_zeros():
    """Should return zero"""
    assert density_correlation(radial_bin_small, wavemode_k, power_spectrum_zeros) == 0


def test_correlation_ones_smallseparation():
    result = (
        density_correlation(radial_bin_small, wavemode_k, power_spectrum_ones)
        * 2
        * np.pi ** 2
    )
    assert result == pytest.approx(-86.738, 0.005)


def test_correlation_ones_largeseparation():
    """Highly oscillatory"""
    result = (
        density_correlation(radial_bin_large, wavemode_k, power_spectrum_ones)
        * 2
        * np.pi ** 2
    )
    assert result == pytest.approx(-0.004404, 0.0004)
