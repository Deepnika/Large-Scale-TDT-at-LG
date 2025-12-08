import numpy as np
import pyvista as pv
from scipy.linalg import eigh
from cosmological_helpers import R


def tensor_to_ellipsoid(position, T, scale=1.0, color='blue'):
    eigvals, eigvecs = eigh(T)  # symmetric, so use eigh
    radii = scale * np.sqrt(1-eigvals/2)
    # radii = scale * np.sqrt(np.abs(1-eigvals))

    # Create a sphere and scale+rotate into ellipsoid
    sphere = pv.Sphere(theta_resolution=20, phi_resolution=20)
    ellipsoid = sphere.scale(radii)

    # Rotation matrix: columns = eigenvectors
    transform = np.eye(4)
    transform[:3, :3] = eigvecs
    transform[:3, 3] = position
    ellipsoid.transform(transform)

    # return ellipsoid

    # Get major axis (direction of largest eigenvalue)
    max_idx = np.argmax(eigvals)
    major_axis_vector = eigvecs[:, max_idx]
    major_axis_start = position - major_axis_vector * radii[max_idx]
    major_axis_end = position + major_axis_vector * radii[max_idx]

    return ellipsoid, major_axis_start, major_axis_end, eigvals



def vizualize_tt_around_LG(df, GS=400./256.):
    # Define 27 positions on cube
    L = 1.0
    offsets = [-L/2, 0, L/2]
    positions = np.array([[x, y, z] for x in offsets for y in offsets for z in offsets] + [[0, 0, 0]])
    # print(positions)

    coords = [-GS, 0, GS]
    tensors = []

    for x in coords:
        for y in coords:
            for z in coords:
                slice = df[(df["GX"] == x) & (df["GY"] == y) & (df["GZ"] == z)]
                tensors.append(slice["T"].values[0])

    # print(tensors)

    # Plotting
    plotter = pv.Plotter()
    for pos, T in zip(positions, tensors):
        # actor = tensor_to_ellipsoid(pos, T, scale=0.3, color='red')
        actor, start, end, eigvals = tensor_to_ellipsoid(pos, T, scale=0.3)

        # structure = classify_structure(eigvals)
        # color = color_map[structure]
        # plotter.add_mesh(actor, color=color, opacity=0.6)

        plotter.add_mesh(actor, opacity=0.6)

        # Add major axis line
        line = pv.Line(start, end)
        plotter.add_mesh(line, color='black', line_width=2)

        # Annotate with eigenvalue sign pattern
        # pattern = ''.join(['+' if v > 0 else '-' for v in eigvals])
        # plotter.add_point_labels([pos], [pattern], font_size=10, point_size=5, shape_opacity=0.3)

    # SGZ = 0 plane ⇒ SGZ unit vector = [0, 0, 1]
    sgz_hat = np.array([0, 0, 1])
    n_galactic = R @ sgz_hat  # Normal in Galactic coordinates

    normal = n_galactic / np.linalg.norm(n_galactic)

    # Define a wide plane centered at origin with the transformed normal
    plane = pv.Plane(center=(0, 0, 0),
                    direction=normal,
                    i_size=1.5, j_size=1.5)  # size can be adjusted

    plotter.add_mesh(plane, color="lightblue", opacity=0.3, style='surface')
    plotter.add_point_labels([(0, 0, 0)], ["Supergalactic Plane"], font_size=10)
    


    # Optional: draw cube for reference
    cube = pv.Cube(center=(0, 0, 0), x_length=L, y_length=L, z_length=L)
    plotter.add_mesh(cube, style='wireframe', color='gray')

    # Supergalactic axes (SGX, SGY, SGZ)
    origin = np.array([0, 0, 0])
    axis_length = 0.75

    axes = {
        'GX': np.array([1, 0, 0]),
        'GY': np.array([0, 1, 0]),
        'GZ': np.array([0, 0, 1])
    }

    # Add arrows for SGX, SGY, SGZ
    for name, direction in axes.items():
        plotter.add_arrows(origin[np.newaxis, :], direction[np.newaxis, :] * axis_length, 
                        mag=1.0, color="red")  # colors: red, green, blue by convention
        plotter.add_point_labels([direction * axis_length], [name], point_size=0, font_size=12)

    # plotter.add_text("Tidal Tensor Ellipsoids in Galactic Coordinates", font_size=12, position='lower_edge')


    plotter.show()