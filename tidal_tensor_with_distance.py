import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from cosmological_helpers import f, beta_star
from tidal_tensor_calculation import calculate_phi_k_from_overdensity_field, calculate_tidal_tensor_from_phi_k, \
    calculate_eigenvalues_and_eigenvectors, calculate_T_SG_eigenvectors


def create_cutoff_df(df, cutoff, save=False, verbose=False):
    """
    Parameters:  
    df: Initial 257 cube grid
    cutoff: Cutoff radius in Mpc/h  (performs a box cut instead of spherical)
    """

    df = df[(np.abs(df["GX"])<=cutoff) & (np.abs(df["GY"])<=cutoff) & (np.abs(df["GZ"])<=cutoff)].copy(deep=True)
    df.drop(df[(df["GX"] == -cutoff) | (df["GY"] == -cutoff) | (df["GZ"] == -cutoff)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if verbose: 
        print("Cutoff: ", cutoff, " Mpc/h")
        print("Mean Overdensity: ", df["Overdensity"].mean())
    
    if save:
        df.to_csv("./Data/{}_cutoff_grid.csv".format(cutoff), index=False, encoding="utf-8")
    return df


def calculate_tidal_tensor_atLG_with_distance(df, a=1.0, GS = 400./256., spacing_factor=2, verbose=False):
    spacing=GS*spacing_factor
    distance_grid =  np.arange(4, 150, spacing)
    print(distance_grid)

    results = []
    b = f(1.0)/beta_star

    for distance in tqdm(distance_grid, desc="Considering different volumes.."):
        _df = create_cutoff_df(df, distance, verbose=False)
        _df["Overdensity"] = _df["Overdensity"] - _df["Overdensity"].mean()
        _df["Overdensity"] = _df["Overdensity"]/b
        N = int(np.round(len(_df) ** (1 / 3)))
        L = distance*2*1000.0
        _df, phi_k = calculate_phi_k_from_overdensity_field(_df, N=N, L=L, a=a, verbose=False)
        _df, T_ij = calculate_tidal_tensor_from_phi_k(_df, phi_k, N=N, L=L, a=a, verbose=False)
        _df, _, _ = calculate_eigenvalues_and_eigenvectors(_df, T_ij, N=N)
        _df = calculate_T_SG_eigenvectors(_df)
        particle = _df[(_df["GX"] == 0) & (_df["GY"] == 0) & (_df["GZ"] == 0)].copy(deep=True)
        particle["distance"] = distance
        results.append(particle.iloc[0].to_dict())

    df_all = pd.DataFrame(results)
    return df_all


def plot_tidal_tensor_evolution_with_distance(df_all):
    sns.set(style="whitegrid", font_scale=1.4, rc={"figure.dpi": 150})

    # Create 1-row, 2-column layout: left = eigenvalues, right = all alignment angles combined
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    eigen_df = df_all["eigenvalues"].apply(pd.Series)
    eigen_df.columns = ["e1", "e2", "e3"]

    # Left plot: eigenvalue evolution
    palette_eigen = sns.color_palette("colorblind", 3)
    for idx, (label, color) in enumerate(zip(["e1", "e2", "e3"], palette_eigen)):
        axs[0].plot(df_all["distance"], eigen_df[label], marker='o', label=rf"$\lambda_{idx+1}$", color=color, linewidth=2)

    axs[0].set_title("Evolution of Tidal Tensor Eigenvalues", fontsize=16)
    axs[0].set_xlabel("Distance from LG to Box Edge (Mpc/h)", fontsize=14)
    axs[0].set_ylabel("Eigenvalue $\\lambda$", fontsize=14)
    axs[0].legend(fontsize=12, frameon=True)
    axs[0].tick_params(axis='both', which='major', labelsize=12)

    axes = {
        "SGX": np.array([1, 0, 0]),
        "SGY": np.array([0, 1, 0]),
        "SGZ": np.array([0, 0, 1])
    }

    # df = df_all.copy()
    for name, axis in axes.items():
        df_all[f"angle_with_{name}"] = df_all["eigenvector_3_SG"].apply(
            lambda v: np.degrees(np.arccos(np.clip(np.abs(np.dot(v, axis)), -1.0, 1.0)))
        )
    virgo_axis = np.array([-3.72, 16.06, -0.78])
    virgo_axis /= np.linalg.norm(virgo_axis)

    df_all["angle_with_virgo"] = df_all["eigenvector_1_SG"].apply(
        lambda v: np.degrees(np.arccos(np.clip(np.abs(np.dot(v, virgo_axis)), -1.0, 1.0)))
    )

    # Right plot: alignment angles
    palette_align = {"SGX": "goldenrod", "SGY": "orangered", "SGZ": "crimson", "Virgo": "purple"}
    for name in ["SGX", "SGY", "SGZ"]:
        axs[1].plot(df_all["distance"], df_all[f"angle_with_{name}"], marker='o', label=f"$\\vec{{v}}_3 \\cdot$ {name}", color=palette_align[name], linewidth=1)

    axs[1].plot(df_all["distance"], df_all["angle_with_virgo"], marker='o', linestyle='--', label="$\\vec{v}_1 \\cdot$ Virgo", color=palette_align["Virgo"], linewidth=1)

    axs[1].set_title("Alignment of Eigenvectors with SG Axes & Virgo", fontsize=16)
    axs[1].set_xlabel("Distance from LG to Box Edge (Mpc/h)", fontsize=14)
    axs[1].set_ylabel("Angle (degrees)", fontsize=14)
    axs[1].legend(fontsize=12, frameon=True)
    axs[1].tick_params(axis='both', which='major', labelsize=12)

    plt.show()


# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from tidal_tensor_calculation import calculate_phi_k_from_overdensity_field, calculate_tidal_tensor_from_phi_k, calculate_eigenvalues_and_eigenvectors



# def create_cutoff_df(df, cutoff, save=False, verbose=False):
#     """
#     Parameters:  
#     df: Initial 257 cube grid
#     cutoff: Cutoff radius in Mpc/h  (performs a box cut instead of spherical)
#     """

#     df = df[(np.abs(df["GX"])<=cutoff) & (np.abs(df["GY"])<=cutoff) & (np.abs(df["GZ"])<=cutoff)].copy(deep=True)
#     df.reset_index(drop=True, inplace=True)
#     if verbose: 
#         print("Cutoff: ", cutoff, " Mpc/h")
#         print("Mean Overdensity: ", df["Overdensity"].mean())
    
#     if save:
#         df.to_csv("./Data/{}_cutoff_grid.csv".format(cutoff), index=False, encoding="utf-8")
#     return df


# def plot_tidal_tensor_atLG_with_distance(df, a=1.0, GS = 400./256., spacing_factor=2, verbose=False):
#     spacing=GS*spacing_factor
#     distance_grid =  np.arange(4, 150, spacing)
#     print(distance_grid)

#     results = []

#     for distance in tqdm(distance_grid, desc="Considering different volumes.."):
#         _df = create_cutoff_df(df, distance, verbose=False)
#         _df["Overdensity"] = _df["Overdensity"] - _df["Overdensity"].mean()
#         N = int(np.round(len(_df) ** (1 / 3)))
#         _df, phi_k = calculate_phi_k_from_overdensity_field(_df, N=N, a=a, verbose=False)
#         _df, T_ij = calculate_tidal_tensor_from_phi_k(_df, phi_k, N=N, a=a, verbose=False)
#         _df, _, _ = calculate_eigenvalues_and_eigenvectors(_df, T_ij, N=N)
#         particle = _df[(_df["GX"] == 0) & (_df["GY"] == 0) & (_df["GZ"] == 0)].copy(deep=True)
#         particle["distance"] = distance
#         results.append(particle.iloc[0].to_dict())

#     df_all = pd.DataFrame(results)
#     return df_all