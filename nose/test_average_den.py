import numpy as np
from dlhalos_code import data_processing as dp
import numpy.testing as nt


def r_coords_box(width_box):
    x, y, z = np.unravel_index(np.arange(width_box ** 3), (width_box, width_box, width_box))
    x -= width_box // 2
    y -= width_box // 2
    z -= width_box // 2
    r_coords = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return r_coords


def f_box(r):
    f = np.cos(r)
    return f.reshape((width, width, width))


def f_box_plus_grad(r, y_coord):
    f = np.cos(r) + 0.5*y_coord
    return f.reshape((width, width, width))


def f_shells_box(width, radius_shells, r_coords):
    f_r = np.zeros((len(r_coords),))
    for i in range(len(radius_shells) - 1):
        ind = (r_coords >= radius_shells[i]) & (r_coords < radius_shells[i + 1])
        f_r[ind] = np.mean(f(r_coords[ind]))
    f_r = f_r.reshape((width, width, width))
    return f_r


if __name__ == "__main__":
    width = 51
    number_shells = 25

    # TEST 1

    coords_box = r_coords_box(width)
    f_r_box = f_box(coords_box)

    r_shells = np.unique(coords_box)
    shell_labels = dp.assign_shell_to_pixels(width, number_shells, r_shells)
    f_output = dp.get_spherically_averaged_box(f_r_box, shell_labels)
    nt.assert_allclose(f_r_box, f_output)

    # TEST 2

    x, y, z = np.unravel_index(np.arange(width ** 3), (width, width, width))
    y -= width // 2

    f_r_box_g = f_box_plus_grad(coords_box, y)
    f_output_g = dp.get_spherically_averaged_box(f_r_box_g, shell_labels)
    nt.assert_allclose(f_r_box, f_output_g)
