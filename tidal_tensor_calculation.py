import numpy as np
from cosmological_helpers import H, omega_matter, R


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



# def calculate_phi_k_from_peculiar_velocity_field(df, a, N=128, L=200000.0, verbose=True):
#     """
#     Calculate the Fourier-space gravitational potential φ(k) from a peculiar velocity field.
    
#     Args:
#         df: DataFrame with 3D velocity components 'G_PVX', 'G_PVY', 'G_PVZ' in km/s
#         a: scale factor
#         N: number of grid points per side (default 128)
#         L: box size in kpc/h (default 200000.0)
#         verbose: whether to print parameters

#     Returns:
#         phi_k: Fourier-space gravitational potential (N, N, N) in the units (Mpc/h)^3 * (km/s)^2
#     """
#     assert int(np.round(len(df) ** (1/3))) == N, "Data size does not match expected grid size"
#     H_a = H(a)
#     omega_m = omega_matter(a)
#     f_a = f(a)
#     C = (1j * 3. * H_a * omega_m * a) / (2 * f_a)    ### Added an a here from my notes

#     if verbose:
#         print(f"Grid Size: {N}, Box Size: {L} kpc/h, Scale Factor: {a}")
#         print(f"Omega_m(a): {omega_m:.4f}, H(a): {H_a:.4f}, f(a): {f_a:.4f}")
#         print(f"Constant C: {C}")

#     pvx = df["G_PVX"].values.reshape(N, N, N)
#     pvy = df["G_PVY"].values.reshape(N, N, N)
#     pvz = df["G_PVZ"].values.reshape(N, N, N)

#     pvx_k = np.fft.ifftn(pvx)
#     pvy_k = np.fft.ifftn(pvy)
#     pvz_k = np.fft.ifftn(pvz)

#     k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")
#     df["k_x"] = k_grid_x.flatten()
#     df["k_y"] = k_grid_y.flatten()
#     df["k_z"] = k_grid_z.flatten()
#     df["k2"] = k2.flatten()

#     phi_k = C * (k_grid_x * pvx_k + k_grid_y * pvy_k + k_grid_z * pvz_k) / k2
#     df["phi_k"] = phi_k.flatten()
#     return df, phi_k



# def calculate_smoothed_phi_k_from_peculiar_velocity_field(df, a, smoothing_scale=4.0, N=128, L=200000.0, verbose=True):
#     """
#     Calculate the Fourier-space gravitational potential φ(k) from a peculiar velocity field.
    
#     Args:
#         df: DataFrame with 3D velocity components 'G_PVX', 'G_PVY', 'G_PVZ' in km/s
#         a: scale factor
#         smoothing_scale: Gaussian smoothing scale in Mpc/h (default 4.0)
#         N: number of grid points per side (default 128)
#         L: box size in kpc/h (default 200000.0)
#         verbose: whether to print parameters

#     Returns:
#         phi_k: Fourier-space gravitational potential (N, N, N) in the units (Mpc/h)^3 * (km/s)^2
#     """
#     assert int(np.round(len(df) ** (1/3))) == N, "Data size does not match expected grid size"
#     H_a = H(a)
#     omega_m = omega_matter(a)
#     f_a = f(a)
#     C = (1j * 3. * H_a * omega_m) / (2 * f_a)

#     if verbose:
#         print(f"Grid Size: {N}, Box Size: {L} kpc/h, Scale Factor: {a}")
#         print(f"Omega_m(a): {omega_m:.4f}, H(a): {H_a:.4f}, f(a): {f_a:.4f}")
#         print(f"Constant C: {C}")

#     pvx = df["G_PVX"].values.reshape(N, N, N)
#     pvy = df["G_PVY"].values.reshape(N, N, N)
#     pvz = df["G_PVZ"].values.reshape(N, N, N)

#     pvx_k = np.fft.ifftn(pvx)
#     pvy_k = np.fft.ifftn(pvy)
#     pvz_k = np.fft.ifftn(pvz)

#     k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")
#     df["k_x"] = k_grid_x.flatten()
#     df["k_y"] = k_grid_y.flatten()
#     df["k_z"] = k_grid_z.flatten()
#     df["k2"] = k2.flatten()

#     smoothing_exponent = np.exp(-k2 * smoothing_scale**2 / 2)

#     phi_k = C * (k_grid_x * pvx_k + k_grid_y * pvy_k + k_grid_z * pvz_k) / k2  * smoothing_exponent
#     df["phi_k"] = phi_k.flatten()
    
#     return df, phi_k




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




# def calculate_peculiar_velocity_from_overdensity(df, a, N=128, L=200000., verbose=True):
#     """
    
#     """
#     assert int(np.round(len(df) ** (1/3))) == N, "Data size does not match expected grid size"
#     H_a = H(a)
#     omega_m = omega_matter(a)
#     f_a = f(a)
#     C = (1.0j * 2 * f_a) /(3. * H_a * omega_m * a)

#     if verbose:
#         print(f"Grid Size: {N}, Box Size: {L} kpc/h, Scale Factor: {a}")
#         print(f"Omega_m(a): {omega_m:.4f}, H(a): {H_a:.4f}, f(a): {f_a:.4f}")
#         print(f"Constant C: {C}")

#     df, phi_k = calculate_phi_k_from_overdensity_field(df, a, N, L)
#     k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")

#     pvx_k = k_grid_x * phi_k * C
#     pvy_k = k_grid_y * phi_k * C
#     pvz_k = k_grid_z * phi_k * C

#     pvx = np.fft.fftn(pvx_k).real
#     pvy = np.fft.fftn(pvy_k).real
#     pvz = np.fft.fftn(pvz_k).real

#     df["PVX"] = pvx.flatten()
#     df["PVY"] = pvy.flatten()
#     df["PVZ"] = pvz.flatten()

#     return df




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