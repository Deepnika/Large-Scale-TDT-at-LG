import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from tidal_tensor_calculation import fourier_modes


def plot_slices(df, center_idx=128, vmax=None, frame='galactic', grid_spacing=400.0 / 256.0):

    mpl.rcParams.update({
        'font.size': 16,           # Base font size
        'axes.titlesize': 14,      # Title font size
        'axes.labelsize': 15,      # Axis label font size
        'xtick.labelsize': 12,     # Tick label font size
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    grid_size = int(np.round(len(df) ** (1 / 3)))
    
    df.sort_values(by=["GX", "GY", "GZ"], inplace=True)
    coords_x = coords_y = coords_z = np.linspace((0 - center_idx) * grid_spacing, (grid_size-1 - center_idx) * grid_spacing, grid_size)
    overdensity = df["Overdensity"].values.reshape(grid_size, grid_size, grid_size)
    xlabel="GX"
    ylabel="GY"
    zlabel="GZ" 
    
    slices = {}
    if grid_size % 2 == 0.:
        slice_indices = [0, grid_size // 2 - 1, grid_size-1]
    elif grid_size % 2 != 0:
        slice_indices = [0, grid_size // 2 , grid_size-1]

    fig, ax = plt.subplots(3, 3, figsize=(18, 18), constrained_layout=True)
    vmin = np.min(overdensity)

    if vmax is None:
        vmax = np.max(overdensity)
    
    plots = []

    for i, slice_idx in enumerate(slice_indices):
        slices[f"X_{i}"] = overdensity[slice_idx, :, :]
        slices[f"Y_{i}"] = overdensity[:, slice_idx, :]
        slices[f"Z_{i}"] = overdensity[:, :, slice_idx]

        plots.append(ax[0][i].pcolormesh(coords_y, coords_z, slices[f"X_{i}"].T, cmap='ocean_r', shading="auto", vmin=vmin, vmax=vmax))
        ax[0][i].set_title(f"{xlabel}: {float((slice_idx - center_idx) * grid_spacing)} Mpc/h")
        ax[0][i].set_xlabel(ylabel)
        ax[0][i].set_ylabel(zlabel)
        ax[0][i].grid(alpha=0.2)

        plots.append(ax[1][i].pcolormesh(coords_x, coords_z, slices[f"Y_{i}"].T, cmap='ocean_r', shading="auto", vmin=vmin, vmax=vmax))
        ax[1][i].set_title(f"{ylabel}: {float((slice_idx - center_idx) * grid_spacing)} Mpc/h")
        ax[1][i].set_xlabel(xlabel)
        ax[1][i].set_ylabel(zlabel)
        ax[1][i].grid(alpha=0.2)


        plots.append(ax[2][i].pcolormesh(coords_x, coords_y, slices[f"Z_{i}"].T, cmap='ocean_r', shading="auto", vmin=vmin, vmax=vmax))
        ax[2][i].set_title(f"{zlabel}: {float((slice_idx - center_idx) * grid_spacing)} Mpc/h")
        ax[2][i].set_xlabel(xlabel)
        ax[2][i].set_ylabel(ylabel)
        ax[2][i].grid(alpha=0.2)

    cbar = fig.colorbar(plots[-1], ax=ax, location="right", shrink=0.99)
    cbar.set_label("Overdensity", fontsize=14)
    plt.show()
    return fig


def create_and_plot_slices_with_smoothing(df, N=128, L=200000.0, plot=True):
    mpl.rcParams.update({
        'font.size': 14,           # Base font size
        'axes.titlesize': 14,      # Title font size
        'axes.labelsize': 14,      # Axis label font size
        'xtick.labelsize': 12,     # Tick label font size
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    delta = df['Overdensity'].values.reshape(N, N, N)
    delta_k = np.fft.ifftn(delta)
    k_grid_x, k_grid_y, k_grid_z, k2 = fourier_modes(N, L, output_unit="h/Mpc")

    smoothing_scales = np.linspace(4.0, 10.0, 6)
    print("Smoothing scales:", smoothing_scales)

    df_dict = {}
    center_idx = N // 2
    grid_spacing = L / N /1000.0
    coords = np.linspace((0 - center_idx) * grid_spacing, (N-1 - center_idx) * grid_spacing, N)

    if plot:
        ncols = 3
        nrows = int(np.ceil(len(smoothing_scales) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), constrained_layout=True)
        axs = axs.flatten()
        vmin, vmax = np.inf, -np.inf
        slices_to_plot = []

    for i, R2 in enumerate(tqdm(smoothing_scales)):
        delta_k_smooth = delta_k * np.exp(-0.5 * (R2**2 - 4.0**2) * k2)
        delta_smooth = np.real(np.fft.fftn(delta_k_smooth))
        _df = df.copy(deep=True)
        _df['Overdensity'] = delta_smooth.flatten()
        _df['Overdensity'] -= _df['Overdensity'].mean()

        df_dict[R2] = _df

        if plot:
            slice_y = delta_smooth[:, center_idx, :]  # GY=0 slice
            slices_to_plot.append((R2, slice_y))
            vmin = min(vmin, np.min(slice_y))
            vmax = max(vmax, np.max(slice_y))

    if plot:
        for i, (R2, slice_y) in enumerate(slices_to_plot):
            im = axs[i].pcolormesh(coords, coords, slice_y.T, shading="auto", cmap='ocean_r', vmin=vmin, vmax=vmax)
            axs[i].set_title(f"GY = 0 | Smoothing = {R2:.1f} h⁻¹ Mpc")
            axs[i].set_xlabel("GX [Mpc/h]")
            axs[i].set_ylabel("GZ [Mpc/h]")
            axs[i].grid(alpha=0.2)
        for j in range(len(slices_to_plot), len(axs)):
            axs[j].axis('off')

        cbar = fig.colorbar(im, ax=axs, location='right', shrink=0.95)
        cbar.set_label("Overdensity", fontsize=14)
        plt.show()

    return df_dict