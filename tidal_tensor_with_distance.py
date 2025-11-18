import numpy as np
import pandas as pd
from tqdm import tqdm
from tidal_tensor_calculation import calculate_phi_k_from_overdensity_field, calculate_tidal_tensor_from_phi_k, calculate_eigenvalues_and_eigenvectors



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