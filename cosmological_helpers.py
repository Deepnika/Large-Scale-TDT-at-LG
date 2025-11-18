import os
import re
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import quad
from astropy.constants import M_sun
from scipy.constants import G, parsec


# Simulation Parameters
OMEGA_MATTER_0 = 0.3 
OMEGA_LAMBDA_0 = 1-OMEGA_MATTER_0   # flat universe
OMEGA_0 = OMEGA_MATTER_0 / (OMEGA_MATTER_0 + OMEGA_LAMBDA_0)

INITIAL_REDSHIFT = 127.0
INITIAL_SCALE_FACTOR = 1/(1+INITIAL_REDSHIFT)


# Constants
G_SI = G  # Gravitational constant in SI units (m^3/kg/s^2)
megaparsec = 1e6 * parsec  # Megaparsec in meters
G_gadget = G * 1e10 * M_sun.value / 1e9 / parsec
# solar_mass = 1.989e30  # Solar mass in kg
# H0 = 100   # 


# Define Galactic to Supergalactic rotation matrix
R = np.array([
    [-7.357425748044e-01, 6.772612964139e-01, -6.085819597056e-17],
    [-7.455377836523e-02, -8.099147130698e-02, 9.939225903998e-01],
    [6.731453021092e-01, 7.312711658170e-01, 1.100812622248e-01]
])


# Carrick et. al. Parameters
beta = 0.359
beta_star = 0.431
# vEXT_paper = np.array([89, -131, 17])
# vLG_total_paper = np.array([71, -553, 345])
# BF_paper = np.array([-3, -72, 38])
# vLG_paper = vLG_total_paper - vEXT_paper


# Helper Functions
def critical_density(): 
    """
    Compute the critical density of the Universe at z = 0.
    Returns:
        float: Critical density in units of h² kg/m³.
    Notes:
        The critical density is defined as:
            rho_crit = 3 H0² / (8 π G)
        where:
            - H0 is the Hubble constant in units of 100 km/s/Mpc (h = 1),
            - G is the gravitational constant in SI units,
            - The result is scaled to units of h² kg/m³.
    """
    return (3 * 100**2) / (8 * np.pi * G_SI * (10**6) * parsec**2)


def calculate_rhobar(L, M, N=128):
    """
    Parameters:
    N: int
        Number of grid points along one dimension.
    L: float
        Length of the box in kpc/h.
    M: float
        Mass of each particle in 1e10 Msun/h.

    Returns:
    rho_bar: float
        Mean density in 1e10 Msun h^2 /kpc^3.
    """
    return (M * N**3) / (L**3) 


def E(a):
    """
    Compute the dimensionless Hubble parameter E(a) = H(a)/H0.
    Args:
        a (float): Scale factor (1 / (1 + z))
    Returns:
        float: E(a), the normalized Hubble parameter at scale factor a.
    Notes:
        For a flat ΛCDM universe:
            E(a) = sqrt(Ω_m / a³ + Ω_Λ)
    """
    return np.sqrt(OMEGA_MATTER_0 / a**3 + OMEGA_LAMBDA_0)


def integrand(a):
    """
    Integrand for the linear growth factor D(a).
    Args:
        a (float): Scale factor
    Returns:
        float: Value of the integrand at scale factor a
    Notes:
        The integrand is:
            1 / [a³ * E(a)³]
        used in the growth factor integral for D(a).
    """
    return 1.0 / (a**3 * E(a)**3)


def D(a):
    """
    Compute the linear growth factor D(a), normalized to D(1) = 1.
    Args:
        a (float): Scale factor (1 / (1 + z))
    Returns:
        float: Normalized growth factor D(a)
    Notes:
        The growth factor describes how density perturbations grow over time.
        It is given by:
            D(a) ∝ E(a) ∫₀^a [1 / (a'³ E(a')³)] da'
        normalized so that D(1) = 1.
    """
    integral, _ = quad(integrand, 0, a)
    norm, _ = quad(integrand, 0, 1)
    return E(a) * integral / norm


def f_from_D(a_vals, D_vals):
    loga = np.log(a_vals)
    logD = np.log(D_vals)
    dlogD_dloga = np.gradient(logD, loga)
    return dlogD_dloga


def H(a):
    """
    Returns H(a) in the units h km s^-1 Mpc^-1
    """
    return 100. * E(a)


def omega_matter(a, omega_m0=OMEGA_MATTER_0, omega_lambda0=OMEGA_LAMBDA_0):
    """
    Compute the matter density parameter Omega_m(a) at scale factor a
    in a flat ΛCDM universe with matter density omega_m0 at a=1.

    Parameters:
    - a: scale factor (a = 1 / (1 + z))
    - omega_m0: present-day matter density (default 0.3)

    Returns:
    - Omega_m(a)
    """
    return (omega_m0 * a**-3) / (omega_m0 * a**-3 + omega_lambda0)


def f(a):
    """
    Compute the logarithmic growth rate of structure, f(a) = dlnD/dlna.
    Args:
        a (float): Scale factor
    Returns:
        float: Growth rate f(a)
    Notes:
        For ΛCDM, an accurate approximation is:
            f(a) ≈ [Ω_m(a)]^0.55
        where:
            Ω_m(a) = Ω_m0 / [Ω_m0 + Ω_Λ * a³]
    """
    return omega_matter(a) ** 0.55


def plot_growth_quantities():
    a_vals = np.linspace(0.01, 1.0, 200)
    D_vals = np.array([D(a) for a in a_vals])
    f_numeric = f_from_D(a_vals, D_vals)
    f_approx_vals = f(a_vals)
    omega_vals = omega_matter(a_vals)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    axs[0].plot(a_vals, D_vals, label=r"$D(a)$", color=sns.color_palette("deep")[0])
    axs[0].set_xlabel("Scale Factor $a$")
    axs[0].legend()
    axs[0].set_title("Linear Growth Factor $D(a)$")

    axs[1].plot(a_vals, f_numeric, label=r"$f(a) = \frac{d\ln D}{d\ln a}$", color=sns.color_palette("deep")[2])
    axs[1].plot(a_vals, f_approx_vals, '--', label=r"$f(a) \approx \Omega_m(a)^{0.55}$", color=sns.color_palette("deep")[1])
    axs[1].set_xlabel("Scale Factor $a$")
    axs[1].set_title("Growth Rate $f(a)$")
    axs[1].legend()

    axs[2].plot(a_vals, omega_vals, label=r"$\Omega_m(a)$", color=sns.color_palette("deep")[3])
    axs[2].set_xlabel("Scale Factor $a$")
    axs[2].set_title(r"Matter Density $\Omega_m(a)$")
    axs[2].legend()

    plt.suptitle("Growth Factor and Growth Rate in ΛCDM", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()