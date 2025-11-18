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

# sns.set(style="whitegrid", context="talk")


# Simulation Parameters
OMEGA_MATTER_0 = 0.3 
OMEGA_LAMBDA_0 = 1-OMEGA_MATTER_0   # flat universe
OMEGA_0 = OMEGA_MATTER_0 / (OMEGA_MATTER_0 + OMEGA_LAMBDA_0)
TimeBetSnapshot = 1.112
TimeOfFirstSnapshot = 0.0078125
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
vEXT_paper = np.array([89, -131, 17])
vLG_total_paper = np.array([71, -553, 345])
BF_paper = np.array([-3, -72, 38])
vLG_paper = vLG_total_paper - vEXT_paper


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


def plot_delta(df, b_value):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    sns.histplot(df["Overdensity_mean_removed"], bins=100, ax=axs[0], color="skyblue",
                 label=r"$\delta_g^* - \langle \delta_g^* \rangle$")
    axs[0].set_title("Luminosity-weighted Overdensity")
    axs[0].set_xlabel(r"$\delta_g^* - \langle \delta_g^* \rangle$")
    axs[0].legend()

    sns.histplot(df["Overdensity_mass"], bins=100, ax=axs[1], color="salmon",
                 label=fr"$\delta_m = (\delta_g^* - \langle \delta_g^* \rangle) / b^*$, $b^* = {b_value:.2f}$")
    axs[1].set_title("Mass Overdensity")
    axs[1].set_xlabel(r"$\delta_m$")
    axs[1].legend()

    # plt.suptitle(r"Comparison of Galaxy and Mass Overdensity Distributions in a $200\,h^{-1}\mathrm{Mpc}$ Cube", fontsize=16)
    plt.tight_layout()
    plt.show()
    
def plot_delta_initial(df):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    sns.histplot(df["Overdensity"], bins=100, ax=axs[0], color="skyblue", label=r"$\delta_m$")
    axs[0].set_title("Mass Overdensity (z=0)")
    axs[0].set_xlabel(r"$\delta_m$")
    axs[0].legend()

    sns.histplot(df["Overdensity_Initial"], bins=100, ax=axs[1], color="salmon",label=r"$\delta_i$")
    axs[1].set_title("Initial Mass Overdensity (z=127)")
    axs[1].set_xlabel(r"$\delta_i$")
    axs[1].legend()

    # plt.suptitle(r"Comparison of Mass Overdensity Distributions in a $200\,h^{-1}\mathrm{Mpc}$ Cube at z=0 and z=127", fontsize=16)
    plt.tight_layout()
    plt.show()
    

# Simulation Helper Functions
def mass_each_particle(N=128, L=200000.0):
    """
    Returns mass of each particle in the units: 1e10 solar masses/h

    Parameters:
    N: number of particles
    L: Box Size in kpc/h

    Returns:
    mass: mass of each particle in the units: 1e10 solar masses/h
    """
    return ( (3*OMEGA_0*(100**2)*(L**3)) / (8*np.pi*G_SI*(N**3)*(10**10)*M_sun.value) ) * (10**3 * parsec)


def calculate_masstable(N_PARTICLES_LIST, N=128, L=200000.0):
    """
    Returns mass table for each particle type in the units: 1e10 solar masses/h

    Parameters:
    N_PARTICLES_LIST: list of number of particles for each type.
    grid_spacing: grid spacing in Mpc/h

    Returns:
    masstable: mass table for each particle type in the units: 1e10 solar masses/h
    """
    mass = mass_each_particle(N=N, L=L)
    masstable = np.zeros(6)
    for i in range(len(N_PARTICLES_LIST)):
        if N_PARTICLES_LIST[i] > 0:
            masstable[i] = mass
    return masstable



def fourier_modes(N, L, output_unit="h/kpc"):
    """
    Parameters: 
    N: number of grid points
    L: Units of "kpc/h"
    output_unit: if output_unit is in h/Mpc, converts to it.


    Returns in units "h/kpc" or "h/Mpc"
    """
    kx = np.fft.fftfreq(N) * 2 * np.pi * N / L
    ky = np.fft.fftfreq(N) * 2 * np.pi * N / L
    kz = np.fft.fftfreq(N) * 2 * np.pi * N / L
    k_grid_x, k_grid_y, k_grid_z = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = k_grid_x**2 + k_grid_y**2 + k_grid_z**2
    k2[0, 0, 0] = 1e-20  # avoid div by zero

    if output_unit == "h/Mpc":
        k_grid_x = k_grid_x*1000.
        k_grid_y = k_grid_y*1000.
        k_grid_z = k_grid_z*1000.
        k2 = k2*1e6

    return k_grid_x, k_grid_y, k_grid_z, k2



# Creating Initial Conditions for Simulations

def create_initial_conditions_from_overdensity_field(
    df, overdensity_column=['Overdensity'], N=128, L=200000.0, GS=400000./256., remove_mean=True, remove_luminosity_weight=True, 
    b=f(1.0)/beta_star, plots=False, bins=100, folder_name=None, phi=False, verbose=True
):
    """
    This function creates initial conditions for a cosmological simulation
    based on a peculiar velocity field. It uses the INITIAL_SCALE_FACTOR and other global parameters to create the initial conditions.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing the peculiar velocity data.
        columns should include:
            'Overdensity' or another overdensity column with specified name
            'GX', 'GY', and 'GZ' for the positions in Mpc/h where (0, 0, 0) is the center of the simulation box or the LG.
    -----------
    overdensity_column : str
        Column name in the DataFrame that contain the overdensity.
    N : int
        Number of grid points in each dimension.
    L : float
        Size of the simulation box in kpc/h.
    GS : float
        Spacing between grid points in kpc/h.
    remove_mean : bool
        If true, removes the mean overdensity of the box from all points.
    remove_luminosity_weight : bool
        If true, this will remove the luminosity weighting to convert the luminosity weighted overdensity to mass overdensity.
    b: float
        To go from $$\delta_g^* = b \delta_m$$, the factor b to be divided where $b = f(a)/\beta$ by default.
    plots : bool
        If True, histograms of the displacements and peculiar velocities are plotted.
    bins : int
        Number of bins for the histograms.
    folder_name : str
        Name of the folder where the initial conditions will be saved.
        The initial conditions will be saved in a file named 'initial_conditions.hdf5' in this folder.
    phi : bool
        If True, the potential of the particles is included in the initial conditions.
        The potential is calculated based on the function compute_potential.
        # It is in units of check the units here
    verbose : bool
        If true, will print out the summary
    
    -----------

    Returns:
    df : pandas.DataFrame
        DataFrame with the initial conditions for the simulation.
        The columns 'X', 'Y', and 'Z' are added, representing the positions of the particles today in kpc/h wrt the box.
        The columns 'vx_initial', 'vy_initial', and 'vz_initial' are added, representing the peculiar velocities of the particles in km/s at the initial time.
        The columns 'dx', 'dy', and 'dz' are added, representing the displacements of the particles from the initial grid centers to the initial conditions in kpc/h.
        The columns 'Xi', 'Yi', and 'DZ' are added, representing the positions of the particles at infinite redshift (or the grid centers) in kpc/h wrt the box.
    """
    assert int(np.round(len(df) ** (1/3))) == N, f"Number of particles {len(df)} is not a cube of an integer. Please check the grid size."
    assert D(1.0) == 1.0

    N_PARTICLES_GAS = 0
    N_PARTICLES_HALO = len(df)
    N_PARTICLES_DISK = 0
    N_PARTICLES_BULGE = 0
    N_PARTICLES_STARS = 0
    N_PARTICLES_BNDRY = 0
    N_PARTICLES_LIST = [N_PARTICLES_GAS, N_PARTICLES_HALO, N_PARTICLES_DISK, N_PARTICLES_BULGE, N_PARTICLES_STARS, N_PARTICLES_BNDRY]
    MASSTABALE = calculate_masstable(N_PARTICLES_LIST, N, L)
    M = mass_each_particle(N, L)
    rhobar = calculate_rhobar(L, M) 
    C = - (4 * np.pi * G_gadget * rhobar) / (INITIAL_SCALE_FACTOR)
    velocity_factor = (2 / (3*INITIAL_SCALE_FACTOR*H(INITIAL_SCALE_FACTOR)*f(INITIAL_SCALE_FACTOR)))

    if verbose:
        print("Mass table: ", MASSTABALE)
        print("Number of particles: ", N_PARTICLES_LIST)
        print("Grid Spacing: ", GS, " kpc/h")
        print("Box Size: ", L, " kpc/h")
        print("b = ", b)
        print("D(1/128.) = ", D(INITIAL_SCALE_FACTOR))
        print("Gravitational Constant G: ", G, "m^3/kg/s^2")
        print("Parsec: ", parsec, "m")
        print("Solar Mass M_sun: ", M_sun.value, "kg")
        print("Gravitational Constant G_gadget: ", G_gadget, r"kpc (km)^2/(1e10 M_sun s^2)")
        print("Mass of each particle M: ", M, "1e10 M_sun/h")
        print("Average Density $\\bar{\\rho}$: ", rhobar, "1e10 Msun h^2/kpc^3.")
        print("Poisson Equation Constant C: ", C, "(h^2 km^2)/(kpc^2 s^2)")
        print("H(1/128.) = ", H(INITIAL_SCALE_FACTOR))
        print("f(1/128.) = ", f(INITIAL_SCALE_FACTOR))
        print("Velocity Factor = ", velocity_factor)
    
    if remove_luminosity_weight:
        df["Overdensity_g"] = df[overdensity_column]/b
        df["Overdensity_Initial"] = (D(INITIAL_SCALE_FACTOR) / D(1.0)) * df["Overdensity_g"]
    else: 
        df["Overdensity_Initial"] = (D(INITIAL_SCALE_FACTOR) / D(1.0)) * df[overdensity_column]
    
    if remove_mean:
        df["Overdensity_Initial"] = df["Overdensity_Initial"] - df["Overdensity_Initial"].mean()

    if plots:
        df["Overdensity_Initial"].hist(bins=100, alpha=0.7, figsize=(10, 6), label=r"$\delta_g$ at $z=127$")
        plt.legend()
        plt.show()

    df["Xi"] = (df["GX"] - df["GX"].values[0])*1000 + GS/2
    df["Yi"] = (df["GY"] - df["GY"].values[0])*1000 + GS/2
    df["Zi"] = (df["GZ"] - df["GZ"].values[0])*1000 + GS/2

    delta = df["Overdensity_Initial"].values.reshape(N, N, N)
    delta_k = np.fft.fftn(delta)

    k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L)

    # phi_k = np.zeros((128,128,128),dtype=complex)
    # for i in range(0,128):
    #     for j in range(0,128):
    #         for h in range (0,128):
    #             if k2[i][j][h] == 0:
    #                 phi_k[i][j][h]= 0 +0.j
    #             else:
    #                 phi_k[i][j][h] = (C * delta_k[i][j][h]) / (k2[i][j][h])

    phi_k = C * delta_k / k2
    phi_k[0, 0, 0] = 0

    # gx_k = np.zeros((128,128,128),dtype=complex)
    # gy_k = np.zeros((128,128,128),dtype=complex)
    # gz_k = np.zeros((128,128,128),dtype=complex)

    # for i in range(0,128):
    #     for j in range(0,128):
    #         for h in range (0,128):
    #             if k2[i][j][h] == 0:
    #                 gx_k[i][j][h] = 0 +0.j
    #                 gy_k[i][j][h] = 0 + 0.j
    #                 gz_k[i][j][h] = 0 +0.j
    #             else:
    #                 gx_k[i][j][h] = (-1.0j * C * delta_k[i][j][h] * k_grid_x[i][j][h]) / (k2[i][j][h])
    #                 gy_k[i][j][h] = (-1.0j * C * delta_k[i][j][h] * k_grid_y[i][j][h]) / (k2[i][j][h])
    #                 gz_k[i][j][h] = (-1.0J * C * delta_k[i][j][h] * k_grid_z[i][j][h]) / (k2[i][j][h])

    gx_k = -1.j * k_grid_x * phi_k
    gy_k = -1.j * k_grid_y * phi_k
    gz_k = -1.j * k_grid_z * phi_k
    
    gx = np.fft.ifftn(gx_k).real
    gy = np.fft.ifftn(gy_k).real
    gz = np.fft.ifftn(gz_k).real
    phi = np.fft.ifftn(phi_k).real

    # d = np.zeros((128,128,128,3))
    # for o in range(0,128):
    #     for p in range(0,128):
    #         for u in range (0,128):
    #             d[o][p][u][0] = - gx[o][p][u]/C
    #             d[o][p][u][1] = - gy[o][p][u]/C
    #             d[o][p][u][2] = - gz[o][p][u]/C

    # vpeculiar = np.zeros((128,128,128,3))
    # for i in range(0,128):
    #     for j in range(0,128):
    #         for h in range (0,128):
    #             vpeculiar[i][j][h][0] = velocity_factor * gx[i][j][h]
    #             vpeculiar[i][j][h][1] = velocity_factor * gy[i][j][h]
    #             vpeculiar[i][j][h][2] = velocity_factor * gz[i][j][h]

    # Displacement and peculiar velocity fields
    d = -np.stack([gx, gy, gz], axis=-1) / C  # shape: (N, N, N, 3)
    vpeculiar = velocity_factor * np.stack([gx, gy, gz], axis=-1)

    # df["dx"] = d[:, :, :, 0].flatten()
    # df["dy"] = d[:, :, :, 1].flatten()
    # df["dz"] = d[:, :, :, 2].flatten()

    # df["PVX"] = vpeculiar[:, :, :, 0].flatten() * 10**3
    # df["PVY"] = vpeculiar[:, :, :, 1].flatten() * 10**3
    # df["PVZ"] = vpeculiar[:, :, :, 2].flatten() * 10**3

    df["dx"] = d[..., 0].flatten()
    df["dy"] = d[..., 1].flatten()
    df["dz"] = d[..., 2].flatten()

    df["PVX"] = vpeculiar[..., 0].flatten() * 1e3
    df["PVY"] = vpeculiar[..., 1].flatten() * 1e3
    df["PVZ"] = vpeculiar[..., 2].flatten() * 1e3

    df["VX"] = df["PVX"] / (INITIAL_SCALE_FACTOR**0.5)
    df["VY"] = df["PVY"] / (INITIAL_SCALE_FACTOR**0.5)
    df["VZ"] = df["PVZ"] / (INITIAL_SCALE_FACTOR**0.5)

    df["X"] = df["Xi"] + df["dx"]
    df["Y"] = df["Yi"] + df["dy"]
    df["Z"] = df["Zi"] + df["dz"]

    if plots:
        fig, ax = plt.subplots(3, 3, figsize=(18, 15))
        ax = ax.flatten()
        df["dx"].hist(bins=bins, label="dX (kpc/h)", alpha=0.5, ax=ax[0])
        df["dy"].hist(bins=bins, label="dY (kpc/h)", alpha=0.5, ax=ax[1])
        df["dz"].hist(bins=bins, label="dZ (kpc/h)", alpha=0.5, ax=ax[2])

        df["VX"].hist(bins=bins, label="VX (km/s)", alpha=0.5, ax=ax[3])
        df["VY"].hist(bins=bins, label="VY (km/s)", alpha=0.5, ax=ax[4])
        df["VZ"].hist(bins=bins, label="VZ (km/s)", alpha=0.5, ax=ax[5])

        df["PVX"].hist(bins=bins, label="PVX (km/s)", alpha=0.5, ax=ax[6])
        df["PVY"].hist(bins=bins, label="PVY (km/s)", alpha=0.5, ax=ax[7])
        df["PVZ"].hist(bins=bins, label="PVZ (km/s)", alpha=0.5, ax=ax[8])

        for a in ax:
            a.legend()
            a.set_ylabel("Count")
        
        plt.tight_layout()
        fig.show()

    file_path = f"Simulations/{folder_name}/initial_conditions.hdf5"
    with h5py.File(file_path, 'w') as f:
        header = f.create_group("Header")
        header.attrs['BoxSize'] = L
        header.attrs['Flag_Cooling'] = np.int64(0)
        header.attrs['Flag_Entropy_ICs'] = np.array([0, 0, 0, 0, 0, 0])
        header.attrs['Flag_Feedback'] = np.int64(0)
        header.attrs['Flag_Metals'] = np.int64(0)
        header.attrs['Flag_Sfr'] = np.int64(0)
        header.attrs['Flag_StellarAge'] = np.int64(0)
        header.attrs['HubbleParam'] = 0.73
        header.attrs['MassTable'] = np.array(MASSTABALE)
        header.attrs['NumFilesPerSnapshot'] = np.int64(1)
        header.attrs['NumPart_ThisFile'] = np.array(N_PARTICLES_LIST)
        header.attrs['NumPart_Total'] = np.array(N_PARTICLES_LIST)
        header.attrs['NumPart_Total_HighWord'] = np.array([0, 0, 0, 0, 0, 0])
        header.attrs['Omega0'] = OMEGA_0
        header.attrs['OmegaLambda'] = OMEGA_LAMBDA_0
        header.attrs['Redshift'] = INITIAL_REDSHIFT
        header.attrs['Time'] = INITIAL_SCALE_FACTOR

        parttype1 = f.create_group("PartType1")
        
        coordinates_data = np.array(df[["X", "Y", "Z"]])
        parttype1.create_dataset("Coordinates", dtype=float, shape=(2097152,3), data=coordinates_data)

        velocities_data = np.array(df[["VX", "VY", "VZ"]])
        parttype1.create_dataset("Velocities", dtype=float, shape=(2097152,3), data=velocities_data)

        particle_ids = np.array(df.index)
        parttype1.create_dataset("ParticleIDs", dtype=int, data=particle_ids)

        if phi:
            potential_data = np.array(df[["Phi"]])
            parttype1.create_dataset("Potential", dtype=float, data=potential_data)
        
        f.close()

    return df



# View Initial Conditions and Simulation Snapshots

def check_simulation_results(
    folder_name, snapshotnumber=None, initial_conditions=False, phi=False, N=128, L=200000.0, 
    GS=400000./256., plots=True, bins=200, verbose=True
):
    """
    Check the simulation results for a given folder and snapshot number.
    
    Parameters:
    folder_name: str
        The name of the folder containing the simulation results.
    snapshotnumber: str
        The snapshot number to check in string format. If None, the initial conditions are checked.
    initial_conditions:  bool
        If True, check the initial conditions instead of a snapshot. 
    phi: bool
        If True, include the potential in the output DataFrame. Only works when potential is calculated.
    N: int
        The number of particles in the simulation to the power (1/3).
    L: float
        The size of the simulation box in kpc/h.
    GS: float
        Spacing between grid points in kpc/h.
    plots: bool
        If True, plot histograms of the displacements and velocities.
    bins: int
        The number of bins for the histograms.
    verbose: bool
        If true, will print out the summary

    ----------

    Returns:
    df: pandas.DataFrame
        DataFrame containing the simulation results with the following columns:
        - X: X coordinates
        - Y: Y coordinates
        - Z: Z coordinates
        - VX: X velocities
        - VY: Y velocities
        - VZ: Z velocities
        - PVX: Proper X velocities
        - PVY: Proper Y velocities
        - PVZ: Proper Z velocities
        - dx: Displacement in X
        - dy: Displacement in Y
        - dz: Displacement in Z
        - phi: Potential (if included)
    """
    if snapshotnumber is not None:
        path = f"./Simulations/{folder_name}/Output/snapshot3halosc_{snapshotnumber}.hdf5"
        time = TimeOfFirstSnapshot * TimeBetSnapshot**int(snapshotnumber)
        z = (1/time) -1
    
    elif initial_conditions:
        path = f"./Simulations/{folder_name}/initial_conditions.hdf5"
        z = INITIAL_REDSHIFT
        time = 1/(1+z)
    
    if verbose:
        print("Redshift: ", z)
        print("Scale Factor: ", time)
    
    f = h5py.File(path)
    df = pd.DataFrame(np.array(f['PartType1']["Coordinates"]), columns=["X", "Y", "Z"])
    df["VX"], df["VY"], df["VZ"] = np.array(f['PartType1']["Velocities"])[:, 0], np.array(f['PartType1']["Velocities"])[:, 1], np.array(f['PartType1']["Velocities"])[:, 2]
    df["PVX"] = df["VX"] * time**0.5
    df["PVY"] = df["VY"] * time**0.5
    df["PVZ"] = df["VZ"] * time**0.5
    if phi:
        df["phi"] = np.array(f['PartType1']["Potential"])
    df["ids"] = np.array(f['PartType1']["ParticleIDs"])
    df.sort_values(by=['ids'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["ids"], inplace=True)

    X = Y = Z = np.arange(0, N) * GS + 0.5* GS
    X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")
    df["Xi"] = X.flatten()
    df["Yi"] = Y.flatten()
    df["Zi"] = Z.flatten()

    def periodic_displacement(delta, L):
        return (delta + L / 2) % L - L / 2

    df["dx"] = periodic_displacement(df["X"] - df["Xi"], L)
    df["dy"] = periodic_displacement(df["Y"] - df["Yi"], L)
    df["dz"] = periodic_displacement(df["Z"] - df["Zi"], L)

    if plots:
        fig, ax = plt.subplots(3, 3, figsize=(18, 15), sharey=True)
        ax = ax.flatten()
        df["dx"].hist(bins=bins, label="dX (kpc/h)", alpha=0.5, ax=ax[0])
        df["dy"].hist(bins=bins, label="dY (kpc/h)", alpha=0.5, ax=ax[1])
        df["dz"].hist(bins=bins, label="dZ (kpc/h)", alpha=0.5, ax=ax[2])

        df["VX"].hist(bins=bins, label="PVX Gadget (km/s)", alpha=0.5, ax=ax[3])
        df["VY"].hist(bins=bins, label="PVY Gadget (km/s)", alpha=0.5, ax=ax[4])
        df["VZ"].hist(bins=bins, label="PVZ Gadget (km/s)", alpha=0.5, ax=ax[5])

        df["PVX"].hist(bins=bins, label="PVX (km/s)", alpha=0.5, ax=ax[6])
        df["PVY"].hist(bins=bins, label="PVY (km/s)", alpha=0.5, ax=ax[7])
        df["PVZ"].hist(bins=bins, label="PVZ (km/s)", alpha=0.5, ax=ax[8])

        for i, a in enumerate(ax):
            a.legend()
            if i%3 == 0:
                a.set_ylabel("Count")
        
        plt.tight_layout()
        fig.show()
    
    return df



# Track Particle Evolution with Scale Factor

def track_particle_evolution(folder_name, particle_index=1040319, N=128, L=200000.0, GS=400000./256.):
    """
    Tracks and plots the evolution of a particle's properties across snapshots in a simulation folder.
    
    Parameters:
    folder_name: str
        The name of the folder containing the simulation snapshots.
    particle_index: int
        The index (0 to N^3-1) of the particle to track. 1040319 is the local group particle.
    N: int
        Number of particles per dimension (N^3 total).
    L: float
        Size of the simulation box in kpc/h.
    GS: float
        Grid spacing in kpc/h.
    
    Returns:
    df_all: pandas.DataFrame
        DataFrame with time evolution of the selected particle.
    """
    snapshots = []
    snapshot_folder = f"./Simulations/{folder_name}/Output"
    regex = re.compile(r"snapshot3halosc_(\d+)\.hdf5")
    
    for fname in sorted(os.listdir(snapshot_folder)):
        match = regex.match(fname)
        if match:
            snapshots.append(match.group(1))
    
    all_data = []

    for snap in tqdm(snapshots, desc="Reading snapshots"):
        df = check_simulation_results(folder_name, snapshotnumber=snap, plots=False, N=N, L=L, GS=GS, verbose=False)
        time = TimeOfFirstSnapshot * TimeBetSnapshot**int(snap)
        a = time
        particle = df.iloc[particle_index].copy()
        particle["a"] = a
        particle["snapshot"] = snap
        all_data.append(particle)
    
    df_all = pd.DataFrame(all_data)
    
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ax = ax.flatten()
    
    # ax[0].plot(df_all["a"], np.abs(df_all["dx"]), label="dx")
    # ax[1].plot(df_all["a"], np.abs(df_all["dy"]), label="dy")
    # ax[2].plot(df_all["a"], np.abs(df_all["dz"]), label="dz")
    
    # ax[3].plot(df_all["a"], np.abs(df_all["PVX"]), label="PVX")
    # ax[4].plot(df_all["a"], np.abs(df_all["PVY"]), label="PVY")
    # ax[5].plot(df_all["a"], np.abs(df_all["PVZ"]), label="PVZ")

    ax[0].plot(df_all["a"], df_all["dx"], label="dx")
    ax[1].plot(df_all["a"], df_all["dy"], label="dy")
    ax[2].plot(df_all["a"], df_all["dz"], label="dz")
    
    ax[3].plot(df_all["a"], df_all["PVX"], label="PVX")
    ax[4].plot(df_all["a"], df_all["PVY"], label="PVY")
    ax[5].plot(df_all["a"], df_all["PVZ"], label="PVZ")

    for a in ax:
        a.set_xlabel("Scale factor a")
        a.set_ylabel("Value")
        a.legend()
    
    plt.suptitle(f"Evolution of Particle {particle_index}")
    plt.tight_layout()
    plt.show()
    
    return df_all


def coords_to_try(coords, GS):
    shifts = [-GS / 1000.0, 0, GS / 1000.0]  # convert GS to Mpc/h
    coords_to_try = []

    for dx in shifts:
        for dy in shifts:
            for dz in shifts:
                if dx == dy == dz == 0:
                    continue
                shifted = tuple(np.round(np.array(coords) + np.array([dx, dy, dz]), 5))
                coords_to_try.append(shifted)
    return coords_to_try


def track_particle_evolution_by_coords(folder_name, initial_coords=(0.0, 0.0, 0.0), final_coords=None, plot_nearby=False, N=128, L=200000.0, GS=400000./256.):
    """
    Tracks and plots the evolution of the closest particle to the given (GX, GY, GZ) coordinates across snapshots.
    
    Parameters:
    folder_name : str
        Folder containing the simulation outputs.
    coords : tuple
        Coordinates (GX, GY, GZ) in Mpc/h of the target location.
    N : int
        Grid resolution (N^3 particles).
    L : float
        Simulation box size in kpc/h.
    GS : float
        Grid spacing in kpc/h.

    Returns:
    df_all : pd.DataFrame
        DataFrame with evolution data for the nearest particle.
    """
    snapshots = []
    snapshot_folder = f"./Simulations/{folder_name}/Output"
    regex = re.compile(r"snapshot3halosc_(\d+)\.hdf5")
    for fname in sorted(os.listdir(snapshot_folder)):
        match = regex.match(fname)
        if match:
            snapshots.append(match.group(1))

    df_last = check_simulation_results(folder_name, snapshotnumber="046", plots=False, N=N, L=L, GS=GS, verbose=False)

    if final_coords is None:
        df_last["GX"] = (df_last["Xi"] - (L - GS)/2) / 1000.0
        df_last["GY"] = (df_last["Yi"] - (L - GS)/2) / 1000.0
        df_last["GZ"] = (df_last["Zi"] - (L - GS)/2) / 1000.0
        mask = (
            (df_last["GX"] == initial_coords[0]) &
            (df_last["GY"] == initial_coords[1]) &
            (df_last["GZ"] == initial_coords[2])
        )

        if not mask.any():
            raise ValueError(f"No particle found exactly at {initial_coords} in last snapshot")

        particle_index = df_last[mask].index[0]
        title = f"Tracking particle at index {particle_index} closest to initial coordinates {initial_coords}"

        if plot_nearby:
            coords_to_track = coords_to_try(initial_coords, GS)

    else:
        df_last["GX"] = (df_last["X"] - (L - GS)/2) / 1000.0
        df_last["GY"] = (df_last["Y"] - (L - GS)/2) / 1000.0
        df_last["GZ"] = (df_last["Z"] - (L - GS)/2) / 1000.0
        coords_array = np.array(final_coords)
        df_last["dist"] = np.linalg.norm(df_last[["GX", "GY", "GZ"]].values - coords_array, axis=1)
        particle_index = df_last["dist"].idxmin()
        title = f"Tracking particle at index {particle_index} closest to final coordinates {final_coords}"

        if plot_nearby:
            coords_to_track = coords_to_try(final_coords, GS)

    # Function to track a single particle by index
    def track_index(index):
        tracked = []
        for snap in tqdm(snapshots, desc="Reading snapshots"):
            df = check_simulation_results(folder_name, snapshotnumber=snap, plots=False, N=N, L=L, GS=GS, verbose=False)
            time = TimeOfFirstSnapshot * TimeBetSnapshot**int(snap)
            a = time
            particle = df.iloc[index].copy()
            particle["a"] = a
            particle["snapshot"] = snap
            tracked.append(particle)
        return pd.DataFrame(tracked)
    
    # all_data = []

    # Track main particle
    df_main = track_index(particle_index)

    import matplotlib.pyplot as plt

    # Increase global font size
    plt.rcParams.update({
        'font.size': 14,            # base font size
        'axes.titlesize': 16,       # subplot title
        'axes.labelsize': 16,       # x/y label
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14
    })

    # Plot setup
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ax = ax.flatten()

    # Plot main particle with bold lines
    ax[0].plot(df_main["a"], df_main["dx"], label="dx (kpc/h)", color="blue")
    ax[1].plot(df_main["a"], df_main["dy"], label="dy (kpc/h)", color="blue")
    ax[2].plot(df_main["a"], df_main["dz"], label="dz (kpc/h)", color="blue")
    ax[3].plot(df_main["a"], df_main["PVX"], label="PVX (km/s)", color="blue")
    ax[4].plot(df_main["a"], df_main["PVY"], label="PVY (km/s)", color="blue")
    ax[5].plot(df_main["a"], df_main["PVZ"], label="PVZ (km/s)", color="blue")
    
    if plot_nearby:
        for coord in coords_to_track:
            # mask = (
            #     (df_last["GX"] == coord[0]) &
            #     (df_last["GY"] == coord[1]) &
            #     (df_last["GZ"] == coord[2])
            # )
            # if not mask.any():
            #     continue
            # nearby_index = df_last[mask].index[0]
            if final_coords is None:
                mask = (
                    (df_last["GX"] == coord[0]) &
                    (df_last["GY"] == coord[1]) &
                    (df_last["GZ"] == coord[2])
                )
                if not mask.any():
                    continue
                nearby_index = df_last[mask].index[0]
            else:
                df_last["dist"] = np.linalg.norm(df_last[["GX", "GY", "GZ"]].values - np.array(coord), axis=1)
                nearby_index = df_last["dist"].idxmin()
            df_n = track_index(nearby_index)

            # Plot lighter or dashed lines
            ax[0].plot(df_n["a"], df_n["dx"], linestyle="--", alpha=0.5)
            ax[1].plot(df_n["a"], df_n["dy"], linestyle="--", alpha=0.5)
            ax[2].plot(df_n["a"], df_n["dz"], linestyle="--", alpha=0.5)
            ax[3].plot(df_n["a"], df_n["PVX"], linestyle="--", alpha=0.5)
            ax[4].plot(df_n["a"], df_n["PVY"], linestyle="--", alpha=0.5)
            ax[5].plot(df_n["a"], df_n["PVZ"], linestyle="--", alpha=0.5)

    for a_ in ax:
        a_.set_xlabel("Scale factor")
        a_.legend()

    # plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return df_main



    # for snap in tqdm(snapshots, desc="Reading snapshots"):
    #     df = check_simulation_results(folder_name, snapshotnumber=snap, plots=False, N=N, L=L, GS=GS, verbose=False)

        

    #     time = TimeOfFirstSnapshot * TimeBetSnapshot**int(snap)
    #     a = time
    #     particle = df.iloc[particle_index].copy()
    #     particle["a"] = a
    #     particle["snapshot"] = snap
    #     all_data.append(particle)

    # df_all = pd.DataFrame(all_data)

    # fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    # ax = ax.flatten()

    # ax[0].plot(df_all["a"], df_all["dx"], label="dx")
    # ax[1].plot(df_all["a"], df_all["dy"], label="dy")
    # ax[2].plot(df_all["a"], df_all["dz"], label="dz")
    # ax[3].plot(df_all["a"], df_all["PVX"], label="PVX")
    # ax[4].plot(df_all["a"], df_all["PVY"], label="PVY")
    # ax[5].plot(df_all["a"], df_all["PVZ"], label="PVZ")

    # for a_ in ax:
    #     a_.set_xlabel("Scale factor a")
    #     a_.set_ylabel("Value")
    #     a_.legend()

    # plt.suptitle(title)
    # plt.tight_layout()
    # plt.show()

    # return df_all


# def track_particle_evolution_by_coords(folder_name, initial_coords=(0.0, 0.0, 0.0), final_coords=None,
#                                        plot_nearby=False, N=128, L=200000.0, GS=400000./256.):
#     import matplotlib.pyplot as plt
#     import os, re
#     from tqdm import tqdm

#     # Identify snapshots
#     snapshot_folder = f"./Simulations/{folder_name}/Output"
#     regex = re.compile(r"snapshot3halosc_(\d+)\.hdf5")
#     snapshots = sorted([match.group(1) for fname in os.listdir(snapshot_folder)
#                         if (match := regex.match(fname))])

#     # Helper: find closest index to a coordinate in a given df
#     def get_particle_index(df, coord):
#         df["GX"] = (df["Xi"] - (L - GS)/2) / 1000.0
#         df["GY"] = (df["Yi"] - (L - GS)/2) / 1000.0
#         df["GZ"] = (df["Zi"] - (L - GS)/2) / 1000.0
#         mask = (
#             (df["GX"] == coord[0]) &
#             (df["GY"] == coord[1]) &
#             (df["GZ"] == coord[2])
#         )
#         if not mask.any():
#             raise ValueError(f"No particle found exactly at {coord} in snapshot.")
#         return df[mask].index[0]

#     # Establish base particle index
#     df_last = None
#     if final_coords:
#         df_last = check_simulation_results(folder_name, snapshotnumber=snapshots[-1], plots=False, N=N, L=L, GS=GS, verbose=False)
#         df_last["GX"] = (df_last["X"] - (L - GS)/2) / 1000.0
#         df_last["GY"] = (df_last["Y"] - (L - GS)/2) / 1000.0
#         df_last["GZ"] = (df_last["Z"] - (L - GS)/2) / 1000.0
#         final_array = np.array(final_coords)
#         df_last["dist"] = np.linalg.norm(df_last[["GX", "GY", "GZ"]].values - final_array, axis=1)
#         particle_index = df_last["dist"].idxmin()
#         coord_label = final_coords
#         title = f"Tracking particle closest to final coords {final_coords}"
#     else:
#         df0 = check_simulation_results(folder_name, snapshotnumber=snapshots[0], plots=False, N=N, L=L, GS=GS, verbose=False)
#         particle_index = get_particle_index(df0, initial_coords)
#         coord_label = initial_coords
#         title = f"Tracking particle exactly at initial coords {initial_coords}"

#     # Function to track a single particle by index
#     def track_index(index):
#         tracked = []
#         for snap in snapshots:
#             df = check_simulation_results(folder_name, snapshotnumber=snap, plots=False, N=N, L=L, GS=GS, verbose=False)
#             time = TimeOfFirstSnapshot * TimeBetSnapshot**int(snap)
#             a = time
#             particle = df.iloc[index].copy()
#             particle["a"] = a
#             particle["snapshot"] = snap
#             tracked.append(particle)
#         return pd.DataFrame(tracked)

#     # Track main particle
#     df_main = track_index(particle_index)

#     # Plot setup
#     fig, ax = plt.subplots(2, 3, figsize=(16, 10))
#     ax = ax.flatten()

#     # Plot main particle with bold lines
#     ax[0].plot(df_main["a"], df_main["dx"], label="dx", color="blue")
#     ax[1].plot(df_main["a"], df_main["dy"], label="dy", color="blue")
#     ax[2].plot(df_main["a"], df_main["dz"], label="dz", color="blue")
#     ax[3].plot(df_main["a"], df_main["PVX"], label="PVX", color="blue")
#     ax[4].plot(df_main["a"], df_main["PVY"], label="PVY", color="blue")
#     ax[5].plot(df_main["a"], df_main["PVZ"], label="PVZ", color="blue")

#     # Plot nearby particles if requested
#     if plot_nearby:
#         shifts = [-GS / 1000.0, 0, GS / 1000.0]  # convert GS to Mpc/h
#         coords_to_try = []

#         for dx in shifts:
#             for dy in shifts:
#                 for dz in shifts:
#                     if dx == dy == dz == 0:
#                         continue
#                     shifted = tuple(np.round(np.array(coord_label) + np.array([dx, dy, dz]), 5))
#                     coords_to_try.append(shifted)

#         if final_coords:
#             df_near = df_last.copy()
#             df_near["GX"] = (df_near["X"] - (L - GS)/2) / 1000.0
#             df_near["GY"] = (df_near["Y"] - (L - GS)/2) / 1000.0
#             df_near["GZ"] = (df_near["Z"] - (L - GS)/2) / 1000.0
#             center_coord = final_coords
#         else:
#             df_near = check_simulation_results(folder_name, snapshotnumber=snapshots[0], plots=False, N=N, L=L, GS=GS, verbose=False)
#             df_near["GX"] = (df_near["Xi"] - (L - GS)/2) / 1000.0
#             df_near["GY"] = (df_near["Yi"] - (L - GS)/2) / 1000.0
#             df_near["GZ"] = (df_near["Zi"] - (L - GS)/2) / 1000.0
#             center_coord = initial_coords

#         # df0 = check_simulation_results(folder_name, snapshotnumber=snapshots[0], plots=False, N=N, L=L, GS=GS, verbose=False)
#         # df0["GX"] = (df0["Xi"] - (L - GS)/2) / 1000.0
#         # df0["GY"] = (df0["Yi"] - (L - GS)/2) / 1000.0
#         # df0["GZ"] = (df0["Zi"] - (L - GS)/2) / 1000.0

#         for coord in coords_to_try:
#             mask = (
#                 (df0["GX"] == coord[0]) &
#                 (df0["GY"] == coord[1]) &
#                 (df0["GZ"] == coord[2])
#             )
#             if not mask.any():
#                 continue
#             nearby_index = df0[mask].index[0]
#             df_n = track_index(nearby_index)

#             # Plot lighter or dashed lines
#             ax[0].plot(df_n["a"], df_n["dx"], linestyle="--", alpha=0.5)
#             ax[1].plot(df_n["a"], df_n["dy"], linestyle="--", alpha=0.5)
#             ax[2].plot(df_n["a"], df_n["dz"], linestyle="--", alpha=0.5)
#             ax[3].plot(df_n["a"], df_n["PVX"], linestyle="--", alpha=0.5)
#             ax[4].plot(df_n["a"], df_n["PVY"], linestyle="--", alpha=0.5)
#             ax[5].plot(df_n["a"], df_n["PVZ"], linestyle="--", alpha=0.5)

#     for a_ in ax:
#         a_.set_xlabel("Scale factor a")
#         a_.set_ylabel("Value")
#         a_.legend()

#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()

#     return df_main





# Tidal Tensor Calculations

def calculate_phi_k_from_overdensity_field(df, a=1.0, N=128, L=200000., verbose=True):
    """
    Returns phi_k in the units (Mpc/h)^3 * (km/s)^2
    """
    assert int(np.round(len(df) ** (1/3))) == N, "Data size does not match expected grid size"
    H_a = H(a)
    omega_m = omega_matter(a)
    C = -(3. * H_a**2 * omega_m * a**2) / 2
    # C = (3. * H_a**2 * omega_m * a**2) / 2

    if verbose:
        print(f"Grid Size: {N}, Box Size: {L} kpc/h, Scale Factor: {a}")
        print(f"Omega_m(a): {omega_m:.4f}, H(a): {H_a:.4f}")
        print(f"Constant C: {C}")

    delta = df["Overdensity"].values.reshape(N, N, N)

    delta_k = np.fft.ifftn(delta)

    k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")
    df["k_x"] = k_grid_x.flatten()
    df["k_y"] = k_grid_y.flatten()
    df["k_z"] = k_grid_z.flatten()
    df["k2"] = k2.flatten()

    phi_k = C * delta_k / k2
    df["phi_k"] = phi_k.flatten()
    return df, phi_k

def calculate_smoothed_overdensity_field(df, N=128, smoothing_scale=4.0, L=200000.0):
    delta = df["Overdensity"].values.reshape(N, N, N)
    delta_k = np.fft.ifftn(delta)

    k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")

    smoothing_exponent = np.exp(-k2 * smoothing_scale**2 / 2)

    delta_k = delta_k * smoothing_exponent

    delta_smoothed = np.fft.fftn(delta_k).real
    df["Overdensity"] = delta_smoothed.flatten()
    
    return df



def calculate_phi_k_from_peculiar_velocity_field(df, a, N=128, L=200000.0, verbose=True):
    """
    Calculate the Fourier-space gravitational potential φ(k) from a peculiar velocity field.
    
    Args:
        df: DataFrame with 3D velocity components 'G_PVX', 'G_PVY', 'G_PVZ' in km/s
        a: scale factor
        N: number of grid points per side (default 128)
        L: box size in kpc/h (default 200000.0)
        verbose: whether to print parameters

    Returns:
        phi_k: Fourier-space gravitational potential (N, N, N) in the units (Mpc/h)^3 * (km/s)^2
    """
    assert int(np.round(len(df) ** (1/3))) == N, "Data size does not match expected grid size"
    H_a = H(a)
    omega_m = omega_matter(a)
    f_a = f(a)
    C = (1j * 3. * H_a * omega_m * a) / (2 * f_a)    ### Added an a here from my notes

    if verbose:
        print(f"Grid Size: {N}, Box Size: {L} kpc/h, Scale Factor: {a}")
        print(f"Omega_m(a): {omega_m:.4f}, H(a): {H_a:.4f}, f(a): {f_a:.4f}")
        print(f"Constant C: {C}")

    pvx = df["G_PVX"].values.reshape(N, N, N)
    pvy = df["G_PVY"].values.reshape(N, N, N)
    pvz = df["G_PVZ"].values.reshape(N, N, N)

    pvx_k = np.fft.ifftn(pvx)
    pvy_k = np.fft.ifftn(pvy)
    pvz_k = np.fft.ifftn(pvz)

    k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")
    df["k_x"] = k_grid_x.flatten()
    df["k_y"] = k_grid_y.flatten()
    df["k_z"] = k_grid_z.flatten()
    df["k2"] = k2.flatten()

    phi_k = C * (k_grid_x * pvx_k + k_grid_y * pvy_k + k_grid_z * pvz_k) / k2
    df["phi_k"] = phi_k.flatten()
    return df, phi_k

def calculate_smoothed_phi_k_from_peculiar_velocity_field(df, a, smoothing_scale=4.0, N=128, L=200000.0, verbose=True):
    """
    Calculate the Fourier-space gravitational potential φ(k) from a peculiar velocity field.
    
    Args:
        df: DataFrame with 3D velocity components 'G_PVX', 'G_PVY', 'G_PVZ' in km/s
        a: scale factor
        smoothing_scale: Gaussian smoothing scale in Mpc/h (default 4.0)
        N: number of grid points per side (default 128)
        L: box size in kpc/h (default 200000.0)
        verbose: whether to print parameters

    Returns:
        phi_k: Fourier-space gravitational potential (N, N, N) in the units (Mpc/h)^3 * (km/s)^2
    """
    assert int(np.round(len(df) ** (1/3))) == N, "Data size does not match expected grid size"
    H_a = H(a)
    omega_m = omega_matter(a)
    f_a = f(a)
    C = (1j * 3. * H_a * omega_m) / (2 * f_a)

    if verbose:
        print(f"Grid Size: {N}, Box Size: {L} kpc/h, Scale Factor: {a}")
        print(f"Omega_m(a): {omega_m:.4f}, H(a): {H_a:.4f}, f(a): {f_a:.4f}")
        print(f"Constant C: {C}")

    pvx = df["G_PVX"].values.reshape(N, N, N)
    pvy = df["G_PVY"].values.reshape(N, N, N)
    pvz = df["G_PVZ"].values.reshape(N, N, N)

    pvx_k = np.fft.ifftn(pvx)
    pvy_k = np.fft.ifftn(pvy)
    pvz_k = np.fft.ifftn(pvz)

    k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")
    df["k_x"] = k_grid_x.flatten()
    df["k_y"] = k_grid_y.flatten()
    df["k_z"] = k_grid_z.flatten()
    df["k2"] = k2.flatten()

    smoothing_exponent = np.exp(-k2 * smoothing_scale**2 / 2)

    phi_k = C * (k_grid_x * pvx_k + k_grid_y * pvy_k + k_grid_z * pvz_k) / k2  * smoothing_exponent
    df["phi_k"] = phi_k.flatten()
    
    return df, phi_k



def calculate_smoothed_phi_k_from_phi(df, a, smoothing_scale=4.0, N=128, L=200000.0, verbose=True):
    """
    Calculate the Fourier-space gravitational potential φ(k) from a phi.

    Args:
        df: DataFrame with 3D velocity components 'G_PVX', 'G_PVY', 'G_PVZ' in km/s
        a: scale factor
        smoothing_scale: Gaussian smoothing scale in Mpc/h (default 4.0)
        N: number of grid points per side (default 128)
        L: box size in kpc/h (default 200000.0)
        verbose: whether to print parameters

    Returns:
        phi_k: Fourier-space gravitational potential (N, N, N) in the units (Mpc/h)^3 * (km/s)^2
    """
    
    assert int(np.round(len(df) ** (1/3))) == N, "Data size does not match expected grid size"

    if verbose:
        print(f"Grid Size: {N}, Box Size: {L} kpc/h, Scale Factor: {a}")

    phi = df["phi"].values.reshape(N, N, N)
    phi_k = np.fft.ifftn(phi)

    k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")
    df["k_x"] = k_grid_x.flatten()
    df["k_y"] = k_grid_y.flatten()
    df["k_z"] = k_grid_z.flatten()
    df["k2"] = k2.flatten()

    smoothing_exponent = np.exp(-k2 * smoothing_scale**2 / 2)

    phi_k = phi_k * smoothing_exponent

    phi_gadget_smoothed = np.fft.fftn(phi_k).real
    df["phi_gadget_smoothed"] = phi_gadget_smoothed.flatten()

    # phi_k = phi_k * a

    # phi_new = np.fft.fftn(phi_k).real
    # df["phi_new"] = phi_new.flatten()

    # phi_k = phi_k / a**2

    #Let's use unsmoothed gadget phi to calculate tidal tensor
    # phi_k = phi_k / smoothing_exponent / a
    df["phi_k"] = phi_k.flatten()
    
    return df, phi_k



def calculate_smoothed_phi_k_from_phi_k(df, phi_k, a, smoothing_scale=4.0, N=128, L=200000.0, verbose=True):
    """
    Calculate the Fourier-space gravitational potential φ(k) from a phi.

    Args:
        df: DataFrame with 3D velocity components 'G_PVX', 'G_PVY', 'G_PVZ' in km/s
        a: scale factor
        smoothing_scale: Gaussian smoothing scale in Mpc/h (default 4.0)
        N: number of grid points per side (default 128)
        L: box size in kpc/h (default 200000.0)
        verbose: whether to print parameters

    Returns:
        phi_k: Fourier-space gravitational potential (N, N, N) in the units (Mpc/h)^3 * (km/s)^2
    """
    
    assert int(np.round(len(df) ** (1/3))) == N, "Data size does not match expected grid size"

    if verbose:
        print(f"Grid Size: {N}, Box Size: {L} kpc/h, Scale Factor: {a}")

    k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")
    df["k_x"] = k_grid_x.flatten()
    df["k_y"] = k_grid_y.flatten()
    df["k_z"] = k_grid_z.flatten()
    df["k2"] = k2.flatten()

    smoothing_exponent = np.exp(-k2 * smoothing_scale**2 / 2)

    phi_k = phi_k * smoothing_exponent

    phi_smoothed = np.fft.fftn(phi_k).real
    df["phi_smoothed"] = phi_smoothed.flatten()

    df["phi_k"] = phi_k.flatten()
    
    return df, phi_k





def calculate_peculiar_velocity_from_overdensity(df, a, N=128, L=200000., verbose=True):
    """
    
    """
    assert int(np.round(len(df) ** (1/3))) == N, "Data size does not match expected grid size"
    H_a = H(a)
    omega_m = omega_matter(a)
    f_a = f(a)
    C = (1.0j * 2 * f_a) /(3. * H_a * omega_m * a)

    if verbose:
        print(f"Grid Size: {N}, Box Size: {L} kpc/h, Scale Factor: {a}")
        print(f"Omega_m(a): {omega_m:.4f}, H(a): {H_a:.4f}, f(a): {f_a:.4f}")
        print(f"Constant C: {C}")

    df, phi_k = calculate_phi_k_from_overdensity_field(df, a, N, L)
    k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")

    pvx_k = k_grid_x * phi_k * C
    pvy_k = k_grid_y * phi_k * C
    pvz_k = k_grid_z * phi_k * C

    pvx = np.fft.fftn(pvx_k).real
    pvy = np.fft.fftn(pvy_k).real
    pvz = np.fft.fftn(pvz_k).real

    df["PVX"] = pvx.flatten()
    df["PVY"] = pvy.flatten()
    df["PVZ"] = pvz.flatten()

    return df




def calculate_tidal_tensor_from_phi_k(df, phi_k, a, N=128, L=200000.0, verbose=True):
    """
    
    """
    assert int(np.round(len(df) ** (1/3))) == N
    assert np.allclose(df["phi_k"].values, phi_k.flatten())

    omega_m = omega_matter(a)
    H_a = H(a)

    # C = (megaparsec/1000)**2 / ( (megaparsec/1000)**2  * (2/(3*omega_matter(a))) * (H(a)**2) )
    # C = (2/(3*omega_m)) / (H_a**2 * a**2)  # From  1/s**2 to Dimensionless
    C = -2 / (3 * H_a**2 * omega_m * a**2)   # The (-) comes from kikj

    if verbose:
        print("Grid Size: ", N)
        print("Total Grid Points: ", N**3)
        print("Scale Factor: ", a)
        print("Omega Matter: ", omega_m)
        print("H(a): ", H_a)
        # print("f(a): ", f(a))
        print("Constant C: ", C)

    k_grid_x, k_grid_y, k_grid_z, _ = fourier_modes(N, L, output_unit="h/Mpc")
    T_ij = np.zeros((3, 3) + phi_k.shape, dtype=np.complex128)

    T_ij[0, 0] = k_grid_x * k_grid_x * phi_k  # T_xx
    T_ij[1, 1] = k_grid_y * k_grid_y * phi_k  # T_yy
    T_ij[2, 2] = k_grid_z * k_grid_z * phi_k  # T_zz

    T_ij[0, 1] = k_grid_x * k_grid_y * phi_k  # T_xy
    T_ij[1, 0] = k_grid_y * k_grid_x * phi_k  # T_yx

    T_ij[0, 2] = k_grid_x * k_grid_z * phi_k  # T_xz
    T_ij[2, 0] = k_grid_z * k_grid_x * phi_k  # T_zx

    T_ij[1, 2] = k_grid_y * k_grid_z * phi_k  # T_yz
    T_ij[2, 1] = k_grid_z * k_grid_y * phi_k  # T_zy

    reshaped_array = T_ij.reshape(3, 3, -1)
    # df["T_k"] = [reshaped_array[:, :, i] for i in range(reshaped_array.shape[2])]     # (Mpc/h) (km/s)**2

    T_ij_real = np.zeros_like(T_ij, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            T_ij_real[i, j] = np.real(np.fft.fftn(T_ij[i, j])) * C

    reshaped_array = T_ij_real.reshape(3, 3, -1)
    df["T"] = [reshaped_array[:, :, i] for i in range(reshaped_array.shape[2])]  

    return df, T_ij_real



def calculate_eigenvalues_and_eigenvectors(df, T_ij, N=128, suffix=None):
    eigenvalues = np.zeros((N, N, N) + (3,))
    eigenvectors = np.zeros((N, N, N) + (3, 3))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                tensor = T_ij[:, :, i, j, k]
                vals, vecs = np.linalg.eigh(tensor)  # For symmetric tensors
                eigenvalues[i, j, k] = vals
                eigenvectors[i, j, k] = vecs

    ev_flat = eigenvalues.reshape(-1, 3)
    vecs_flat = eigenvectors.reshape(-1, 3, 3)

    if suffix:
        df[f"eigenvalues_{suffix}"] = list(ev_flat)
        for n in range(3):
            df[f"eigenvector_{n+1}_{suffix}"] = list(vecs_flat[:, :, n])
    else:
        df["eigenvalues"] = list(ev_flat)
        for n in range(3):
            df[f"eigenvector_{n+1}"] = list(vecs_flat[:, :, n])

    # df["eigenvalues_sum"] = df["eigenvalues"].apply(lambda x: sum(x))

    return df, eigenvalues, eigenvectors


def calculate_T_SG_eigenvectors(df, N=128):
    df["T_SG"] = df["T"].apply(lambda x: R @ x @ R.T)
    eigenvectors_SG = np.zeros((len(df),) + (3, 3))

    for i in range(len(df)):
        _, vecs = np.linalg.eigh(df["T_SG"].values[i])
        eigenvectors_SG[i] = vecs

    vecs_flat = eigenvectors_SG.reshape(-1, 3, 3)

    for n in range(3):
        df[f"eigenvector_{n+1}_SG"] = list(vecs_flat[:, :, n])

    return df


def compute_euler_angles(v1, v3):
    """
    Compute Euler angles (alpha, beta, gamma) from principal axes vectors.
    
    Args:
        v1: (3,) array for eigenvector with smallest eigenvalue (expanding)
        v3: (3,) array for eigenvector with largest eigenvalue (contracting)

    Returns:
        alpha, beta, gamma in radians
    """
    v1x, v1y, v1z = v1
    v3x, v3y, v3z = v3

    # --- Alpha ---
    alpha = np.arctan2(v3y, v3x) + np.pi / 2
    if v3z * v3y < 0:
        alpha += np.pi
    alpha = np.mod(alpha, 2 * np.pi)  # Normalize to [0, 2π]

    # --- Beta ---
    beta = np.arctan(np.sqrt(v3x**2 + v3y**2) / abs(v3z))

    # --- Gamma ---
    cross_term = v3x * v1y - v3y * v1x
    denom = np.sqrt(v3x**2 + v3y**2)
    sign_v1z = np.sign(v1z) if v1z != 0 else 1.0
    gamma = np.arccos(np.clip(sign_v1z * (cross_term / denom), -1.0, 1.0))

    return alpha, beta, gamma


def calculate_euler_angles(df, SG=True):
    """
    Calculate Euler angles (alpha, beta, gamma) from eigenvectors.
    
    Args:
        df: DataFrame with eigenvectors
        SG: bool, whether to use SG eigenvectors also (default True)

    Returns:
        DataFrame with Euler angles added
    """
    euler_angles = np.zeros((len(df), 3))  # alpha, beta, gamma

    for i in range(len(df)):
        v1 = df["eigenvector_1"].values[i]
        v3 = df["eigenvector_3"].values[i]
        euler_angles[i] = compute_euler_angles(v1, v3)

    df["alpha"] = euler_angles[:, 0]
    df["beta"] = euler_angles[:, 1]
    df["gamma"] = euler_angles[:, 2]

    if SG:
        euler_angles_SG = np.zeros((len(df), 3))  # alpha, beta, gamma

        for i in range(len(df)):
            v1 = df["eigenvector_1_SG"].values[i]
            v3 = df["eigenvector_3_SG"].values[i]
            euler_angles_SG[i] = compute_euler_angles(v1, v3)

        df["alpha_SG"] = euler_angles_SG[:, 0]
        df["beta_SG"] = euler_angles_SG[:, 1]
        df["gamma_SG"] = euler_angles_SG[:, 2]

    return df


def create_cutoff_df(df, cutoff, save=False, verbose=False):
    """
    Parameters:  
    df: Initial 257 cube grid
    cutoff: Cutoff radius in Mpc/h  (performs a box cut instead of spherical)
    """

    df = df[(np.abs(df["GX"])<=cutoff) & (np.abs(df["GY"])<=cutoff) & (np.abs(df["GZ"])<=cutoff)].copy(deep=True)
    df.reset_index(drop=True, inplace=True)
    if verbose: 
        print("Cutoff: ", cutoff, " Mpc/h")
        print("Mean Overdensity: ", df["Overdensity"].mean())
    
    if save:
        df.to_csv("./Data/{}_cutoff_grid.csv".format(cutoff), index=False, encoding="utf-8")
    return df


def plot_tidal_tensor_atLG_with_distance(df, a=1.0, GS = 400./256., spacing_factor=2, verbose=False):
    spacing=GS*spacing_factor
    distance_grid =  np.arange(4, 150, spacing)
    print(distance_grid)

    results = []

    for distance in tqdm(distance_grid, desc="Considering different volumes.."):
        _df = create_cutoff_df(df, distance, verbose=False)
        _df["Overdensity"] = _df["Overdensity"] - _df["Overdensity"].mean()
        N = int(np.round(len(_df) ** (1 / 3)))
        _df, phi_k = calculate_phi_k_from_overdensity_field(_df, N=N, a=a, verbose=False)
        _df, T_ij = calculate_tidal_tensor_from_phi_k(_df, phi_k, N=N, a=a, verbose=False)
        _df, _, _ = calculate_eigenvalues_and_eigenvectors(_df, T_ij, N=N)
        particle = _df[(_df["GX"] == 0) & (_df["GY"] == 0) & (_df["GZ"] == 0)].copy(deep=True)
        particle["distance"] = distance
        results.append(particle.iloc[0].to_dict())

    df_all = pd.DataFrame(results)
    return df_all