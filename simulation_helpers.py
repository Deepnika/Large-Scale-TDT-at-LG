import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tidal_tensor_calculation import fourier_modes
from cosmological_helpers import M_sun, parsec, f, H, G_gadget, D, beta_star, OMEGA_0, OMEGA_LAMBDA_0, G, G_SI, INITIAL_SCALE_FACTOR, \
    INITIAL_REDSHIFT, calculate_rhobar


# Simulation Constants
TimeBetSnapshot = 1.112
TimeOfFirstSnapshot = 0.0078125


# Initial Conditions Helper Functions
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