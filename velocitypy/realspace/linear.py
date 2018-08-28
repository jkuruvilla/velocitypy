import numpy as np
from scipy.integrate import simps


def correlation(SpatialR, Wavemode_k, PowerSpectrum):
    """Function to compute the linear correlation function from
    an input linear power spectrum.
    """
    LenR = len(SpatialR)
    Corr_Function = np.empty(LenR)
    product = Wavemode_k ** 2 * PowerSpectrum

    for i in range(LenR):
        integral = (
            product * np.sin(Wavemode_k * SpatialR[i]) / (Wavemode_k * SpatialR[i])
        )
        Corr_Function[i] = simps(integral, Wavemode_k)

    Corr_Function /= 2 * (np.pi ** 2)
    return Corr_Function


def radial_mean(SpatialR, Wavemode_k, PowerSpectrum, OmegaM):
    """Function to compute the linear mean radial pairwise velocity from
    an input linear power spectrum. Mean radial velocity is denoted as <w>

    .. math::
    \langle w(r) \rangle \hat{r} = -f/pi^2 \int_0^{+\infty} dk k P_{lin}(k) j_1(kr)

    where :math: 'j_1(x) = sin(x)/x^2 - cos(x)/x'

    Args:
        SpatialR: Pair separation at which the mean velocity should be
                  computed (in units of h^{-1} Mpc).
        Wavemode_k: Wavemodes at which the power spectrum values are computed.
        PowerSpectrum: Linear power spectrum
        OmegaM: Matter density (needed to compute f=OmegaM**0.545)

    Returns:
        Mean_Velocity: An array containing the computed mean radial pairwise
                       velocity. In units of h^{-1} Mpc.
    """
    SpatialR = np.asarray(SpatialR)
    growth_rate = OmegaM ** 0.545

    LenR = len(SpatialR)
    Mean_Velocity = np.empty(LenR)
    product = Wavemode_k * PowerSpectrum

    for i in range(LenR):
        kr = Wavemode_k * SpatialR[i]
        integral = product * ((np.sin(kr) / kr ** 2) - (np.cos(kr) / kr))
        Mean_Velocity[i] = simps(integral, Wavemode_k)

    Mean_Velocity *= -(growth_rate / np.pi ** 2)
    return Mean_Velocity
