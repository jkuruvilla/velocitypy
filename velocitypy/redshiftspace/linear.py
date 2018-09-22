import numpy as np
from scipy.integrate import simps


def radial_mean_kaiser(SpatialR, Wavemode_k, PowerSpectrum, OmegaM, Sampling_Num=1000):
    """Function to compute the linear mean radial pairwise velocity from
    an input linear power spectrum in redshift-space. Mean radial velocity is denoted as <w_s>

    $<w_s(r)> = -f/pi^2  \int_{-1}^{+1} dmu \int_0^{+\infty} dk k P_{lin}(k) (1+f*mu^2)^2 * mu * exp(-i*k*r*mu)$

    Args:
        SpatialR: Pair separation at which the mean velocity should be
                  computed (in units of h^{-1} Mpc).
        Wavemode_k: Wavemodes at which the power spectrum values are computed.
        PowerSpectrum: Linear power spectrum
        OmegaM: Matter density (needed to compute f=OmegaM**0.545)
        Sampling_Num: Number of points to sample the mu (from -1 to +1). Default is 1000.

    Returns:
        Mean_Velocity: An array containing the computed mean radial pairwise
                       velocity. In units of h^{-1} Mpc.
    """
    SpatialR = np.asarray(SpatialR)
    growth_rate = OmegaM ** 0.545
    binwidth = 2. / Sampling_Num
    mu = np.arange(-1, 1 + binwidth, binwidth)

    LenR = len(SpatialR)
    Mean_Velocity = np.empty(LenR)
    product = Wavemode_k * PowerSpectrum

    for i in range(LenR):
        inner_integral = []
        for j in range(len(Wavemode_k)):
            kr = Wavemode_k[j] * SpatialR[i]
            inner_ = mu * (1 + (growth_rate * mu ** 2)) ** 2 * -1 * np.sin(kr * mu)
            inner_integral.append(simps(inner_, mu))
        integral = product * inner_integral
        Mean_Velocity[i] = simps(integral, Wavemode_k)

    Mean_Velocity *= growth_rate / (2 * np.pi ** 2)
    return Mean_Velocity
