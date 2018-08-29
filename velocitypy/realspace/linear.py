import numpy as np
from scipy.integrate import simps


def density_correlation(SpatialR, Wavemode_k, PowerSpectrum):
    """Function to compute the linear density correlation function from
    an input linear power spectrum.
    """
    SpatialR = np.asanyarray(SpatialR)
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

    $\langle w(r) \rangle \hat{r} = -f/pi^2 \int_0^{+\infty} dk k P_{lin}(k) j_1(kr)$

    where $j_1(x) = sin(x)/x^2 - cos(x)/x$

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
    SpatialR = np.asanyarray(SpatialR)
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


def radial_dispersion(SpatialR, Wavemode_k, PowerSpectrum, OmegaM):
    """Function to compute the linear radial pairwise velocity dispersion from
    an input linear power spectrum. Radial velocity dispersion is denoted as \sigma^2_r

    \sigma^2_r(r) = f^2/(2pi^2)  \int_0^{+\infty} dk P_{lin}(k) (j_0(kr) - 2*j_1(kr)/(kr))
    where, j_0(x) = sin(x)/x
           j_1(x) = sin(x)/x^2 - cos(x)/x

    Args:
        SpatialR: Pair separation at which the radial dispersion should be
                  computed (in units of h^{-1} Mpc).
        Wavemode_k: Wavemodes at which the power spectrum values are computed.
        PowerSpectrum: Linear power spectrum
        OmegaM: Matter density (needed to compute f=OmegaM**0.545)

    Returns:
        Radial_Dispersion: An array containing the computed radial
                           pairwise velocity dispersion. In units of h^{-2} Mpc^2.
    """
    SpatialR = np.asanyarray(SpatialR)
    growth_rate = OmegaM ** 0.545
    LenR = len(SpatialR)
    Radial_Dispersion = np.empty(LenR)

    for i in range(LenR):
        kr = Wavemode_k * SpatialR[i]
        j0 = np.sin(kr) / kr
        j1_by_kr = (np.sin(kr) / kr ** 3) - (np.cos(kr) / kr ** 2)
        integral = PowerSpectrum * (j0 - (2 * j1_by_kr))
        Radial_Dispersion[i] = simps(integral, Wavemode_k)

    Radial_Dispersion *= growth_rate ** 2 / (2 * np.pi ** 2)
    return Radial_Dispersion


def transverse_dispersion(SpatialR, Wavemode_k, PowerSpectrum, OmegaM):
    """Function to compute the linear transversal pairwise velocity dispersion from
    an input linear power spectrum. Transversal dispersion is denoted as \sigma^2_t

    \sigma^2_t(r) = f^2/(2pi^2)  \int_0^{+\infty} dk P_{lin}(k) j_1(kr)/(kr)
    where j_1(x) = sin(x)/x^2 - cos(x)/x

    Args:
        SpatialR: Pair separation at which the radial dispersion should be
                  computed (in units of h^{-1} Mpc).
        Wavemode_k: Wavemodes at which the power spectrum values are computed.
        PowerSpectrum: Linear power spectrum
        OmegaM: Matter density (needed to compute f=OmegaM**0.545)

    Returns:
        Tangential_Dispersion: An array containing the computed tangential
                               pairwise velocity dispersion. In units of h^{-2} Mpc^2.
    """
    SpatialR = np.asanyarray(SpatialR)
    growth_rate = OmegaM ** 0.545
    LenR = len(SpatialR)
    Tangential_Dispersion = np.empty(LenR)

    for i in range(LenR):
        kr = Wavemode_k * SpatialR[i]
        integral = PowerSpectrum * ((np.sin(kr) / kr ** 3) - (np.cos(kr) / kr ** 2))
        Tangential_Dispersion[i] = simps(integral, Wavemode_k)

    Tangential_Dispersion *= growth_rate ** 2 / (2 * np.pi ** 2)
    return Tangential_Dispersion
