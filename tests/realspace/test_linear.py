import pytest
import numpy as np
from velocitypy.realspace.linear import (
    density_correlation,
    radial_mean,
    radial_dispersion,
    transverse_dispersion,
    onepoint_dispersion,
)

low_k, high_k = 0.001, 100
width = 0.0001
wavemode_k = np.arange(low_k, high_k + width, width)
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
    assert result == pytest.approx(-86.738, rel=0.005)


def test_correlation_ones_largeseparation():
    """Highly oscillatory"""
    result = (
        density_correlation(radial_bin_large, wavemode_k, power_spectrum_ones)
        * 2
        * np.pi ** 2
    )
    assert result == pytest.approx(-0.004404, rel=0.005)


def test_radialmean_zeros():
    """Should return zero"""
    assert radial_mean(radial_bin_small, wavemode_k, power_spectrum_zeros, 1.) == 0


def test_radialmean_ones_smallseparation():
    result = (
        radial_mean(radial_bin_small, wavemode_k, power_spectrum_ones, 1.)
        * -1
        * np.pi ** 2
    )
    assert result == pytest.approx(2.0658, rel=0.005)


def test_radialmean_ones_largeseparation():
    result = (
        radial_mean(radial_bin_large, wavemode_k, power_spectrum_ones, 1.)
        * -1
        * np.pi ** 2
    )
    assert result[0] == pytest.approx(0.000167263, abs=5e-4)


def test_radialdispersion_zeros():
    assert (
        radial_dispersion(radial_bin_small, wavemode_k, power_spectrum_zeros, 1.) == 0
    )


def test_radialdispersion_ones_small():
    result = (
        radial_dispersion(radial_bin_small, wavemode_k, power_spectrum_ones, 1.)
        * 2
        * np.pi ** 2
    )
    assert result == pytest.approx(-0.009007, 0.005)


def test_radialdispersion_ones_large():
    result = (
        radial_dispersion(radial_bin_large, wavemode_k, power_spectrum_ones, 1.)
        * 2
        * np.pi ** 2
    )
    assert result == pytest.approx(-0.000333, 0.005)


def test_transversedispersion_zeros():
    assert (
        transverse_dispersion(radial_bin_small, wavemode_k, power_spectrum_zeros, 1.)
        == 0
    )


def test_transversedispersion_ones_small():
    result = (
        transverse_dispersion(radial_bin_small, wavemode_k, power_spectrum_ones, 1.)
        * 2
        * np.pi ** 2
    )
    assert result == pytest.approx(0.785116, 0.005)


def test_transversedispersion_ones_large():
    result = (
        transverse_dispersion(radial_bin_large, wavemode_k, power_spectrum_ones, 1.)
        * 2
        * np.pi ** 2
    )
    assert result == pytest.approx(0.00621181, 0.005)


def test_onepointdispersion_zeros():
    assert onepoint_dispersion(wavemode_k, power_spectrum_zeros, 1) == 0


def test_onepointdispersion_ones():
    result = onepoint_dispersion(wavemode_k, power_spectrum_ones, 1) * 6 * np.pi ** 2
    assert result == pytest.approx(100, 0.005)
