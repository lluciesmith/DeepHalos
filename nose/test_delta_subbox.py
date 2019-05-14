"""

Here, I am comparing current method to get delta in input subboxes with
old one using a long for loop based on the positions of
particles.

"""


import numpy as np
import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
from mlhalos import parameters
from .. import input_subboxes


def from_ids_to_delta_in_subbox_old(ids, particle_id, delta_all, subbox_shape=9):
    center_subbox = int((subbox_shape - 1) / 2)

    x_pos, y_pos, z_pos = np.where(ids_3d_grid == particle_id)
    one_sided_neighbours = (subbox_shape - 1) / 2

    if any([x_pos[0] + one_sided_neighbours > box_shape, y_pos[0] + one_sided_neighbours > box_shape,
            z_pos[0] + one_sided_neighbours > box_shape, x_pos[0] - one_sided_neighbours < 0,
            y_pos[0] - one_sided_neighbours < 0, z_pos[0] - one_sided_neighbours < 0]):

        print("Going through this horrible loop -- need to come up with something better. "
              "Like shifting particles in the 3D grid")
        grid = np.zeros((subbox_shape, subbox_shape, subbox_shape))

        for particle in ids:
            pos_id = np.where(ids_3d_grid == particle)
            i = pos_id[0] - x_pos
            if i > center_subbox:
                i -= box_shape
            j = pos_id[1] - y_pos
            if j > center_subbox:
                j -= box_shape
            k = pos_id[2] - z_pos
            if k > center_subbox:
                k -= box_shape
            grid[i + center_subbox, j + center_subbox, k + center_subbox] = delta_all[particle]

    else:
        grid = delta_all[ids].reshape((subbox_shape, subbox_shape, subbox_shape))

    center_subbox = int((subbox_shape - 1) / 2)
    assert grid[center_subbox, center_subbox, center_subbox] == delta_all[particle_id], \
        "Particle ID " + str(particle_id) + " is not at the center of the sub-box, something went wrong."
    return grid


def test_delta_in_subbox_around_particle_near_edge_box():
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/mlhalos_files/")
    d = input_subboxes.delta_property(ic.initial_conditions)

    n = 0
    ids = input_subboxes.ids_in_subbox(n, 256, 13, ids_grid=None)
    delta_all = ic.initial_conditions["delta"]

    n_delta_old = from_ids_to_delta_in_subbox_old(ids, n, delta_all, subbox_shape=13)
    n_delta_new = input_subboxes.from_ids_to_delta_in_subbox(ic.initial_conditions, ids, n, delta_all, subbox_shape=13,
                                                             box_shape=256)

    np.testing.assert_allclose(n_delta_old, n_delta_new)