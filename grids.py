import numpy as np
import pandas as pd


def create_257_cube_grid(
        density_field_filepath='Data/2M++/twompp_density.npy',
        output_file_path='./Data/2M++Grids/257_cube_grid.csv'
    ):
    """
    Creates a 257^3 grid centered on the Local Group (LG) from the provided overdensity field.

    Parameters:
        density_field_filepath (str): 
            Path to the numpy file containing the density field. LG is in central voxel at 128,128,128
            (Default: Carrick et. al. 2015 2M++ density field)

        output_file_path (str):
            Path to save the output CSV file containing the grid data.
            (Default: './Data/2M++Grids/257_cube_grid.csv')
    """
    density_field = np.load(density_field_filepath)

    data = []
    grid_spacing = 400.0 / 256.0

    # X, Y, Z are in Galactic cartesian coordinates with LG at (0, 0, 0) Mpc/h 
    for i in range(density_field.shape[0]):
        for j in range(density_field.shape[1]):
            for k in range(density_field.shape[2]):
                x = (i - 128) * grid_spacing
                y = (j - 128) * grid_spacing
                z = (k - 128) * grid_spacing

                density = density_field[i, j, k]

                data.append([x, y, z, density])

    columns = ["GX", "GY", "GZ", "Overdensity"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file_path, encoding="utf-8", index=False)
    return df


def create_128_cube_grid(
        df_full_file_path='./Data/2M++Grids/257_cube_grid.csv',
        output_file_path='./Data/2M++Grids/128_cube_grid.csv'
    ):
    df = pd.read_csv(df_full_file_path, usecols=["GX", "GY", "GZ", "Overdensity"])

    cutoff = 100
    df_128 = df[(np.abs(df["GX"])<=cutoff) & (np.abs(df["GY"])<=cutoff) & (np.abs(df["GZ"])<=cutoff)].copy(deep=True)
    df_128.drop(df_128[(df_128["GX"] == -100) | (df_128["GY"] == -100) | (df_128["GZ"] == -100)].index, inplace=True)
    df_128.reset_index(drop=True, inplace=True)
    df_128.to_csv(output_file_path, index=False, encoding="utf-8")
    return df_128

