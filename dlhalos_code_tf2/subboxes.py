"""

Here, I am defining subboxes based on the positions of particles in
a 3 unifrom grid. This is better than other method, but a bit slower.

- Define 3d grid and 3d grid coordinates only once to make it faster,
- Implement periodic boundary conditions.

"""


import numpy as np
import sys; sys.path.append("")
import gc
import pynbody


class Subboxes:

    def __init__(self, initial_parameters, snapshot=None, qty="delta", subbox_shape=(9, 9, 9)):

        self.initial_parameters = initial_parameters
        if snapshot is None:
            print("Loading initial conditions")
            self.snapshot = initial_parameters.initial_conditions
        else:
            self.snapshot = snapshot

        self.shape = self.initial_parameters.shape

        self.qty = qty
        if self.qty == "delta":
            self.delta_property(self.snapshot)

        self.ids_3d = self.get_ids_in_3d_grid()
        self.coords_3d = self.get_3d_coordinates(flatten=False)
        self.coords_flatten = self.get_3d_coordinates(flatten=True)

        self.subbox_shape = subbox_shape
        self.half_subbox = int((subbox_shape[0] - 1) / 2)

    def delta_property(self, snapshot):
        rho = snapshot["rho"]
        mean_rho = self.initial_parameters.get_mean_matter_density_in_the_box(snapshot,
                                                                              units=str(snapshot["pos"].units))
        snapshot["delta"] = rho / mean_rho
        return snapshot["delta"]

    def get_ids_in_3d_grid(self):
        ids = self.snapshot["iord"]
        return ids.reshape(self.shape, self.shape, self.shape)

    def get_3d_coordinates(self, flatten=False):
        x = np.arange(self.shape)
        y = np.arange(self.shape)
        z = np.arange(self.shape)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        if flatten is True:
            return xx.flatten(), yy.flatten(), zz.flatten()
        else:
            return xx, yy, zz

    # Get list of particle IDS in subbox around particle_ID

    def boundary_conditions_fix_coords(self, coords, pos):
        print("Fixing some boundary conditions")
        case3 = pos[0] + self.half_subbox > self.shape
        case03 = pos[0] - self.half_subbox < 0

        if case3:
            lz = self.shape - pos
            extra_z = self.half_subbox - lz
            cond = coords <= extra_z

        elif case03:
            extra_z = self.half_subbox - pos
            cond = coords >= self.shape - extra_z

        else:
            raise ValueError("coordinate was not found near the edge of the box")
        return cond

    def conditions_on_coordinate(self, coordinates, position):
        one_sided_neighbours = self.half_subbox

        if any([position[0] + one_sided_neighbours > self.shape, position[0] - one_sided_neighbours < 0]):
            extra_y_cond = self.boundary_conditions_fix_coords(coordinates, position)
            y_cond = ((abs(coordinates - position[0]) <= one_sided_neighbours) | extra_y_cond)
        else:
            y_cond = (abs(coordinates - position[0]) <= one_sided_neighbours)
        return y_cond

    def get_coords_of_subbgrid_around_particle_id(self, particle_id, x_coords, y_coords, z_coords):
        x_pos, y_pos, z_pos = np.where(self.ids_3d == particle_id)

        x_cond = self.conditions_on_coordinate(x_coords, x_pos)
        y_cond = self.conditions_on_coordinate(y_coords, y_pos)
        z_cond = self.conditions_on_coordinate(z_coords, z_pos)

        cond_xyz = x_cond & y_cond & z_cond
        return cond_xyz

    def get_ids_in_subgrid_around_particle_id(self, particle_id, x_coords, y_coords, z_coords):
        cond_xyz = self.get_coords_of_subbgrid_around_particle_id(particle_id, x_coords, y_coords, z_coords)
        n_neighbours_coords = np.where(cond_xyz)
        return self.ids_3d[n_neighbours_coords]

    def sort_ids_correct_order_subbox(self, particle_id, subbox_ids):
        xx, yy, zz = self.coords_flatten
        grid_pos_ids = np.column_stack((xx, yy, zz))

        diff_pos = grid_pos_ids[subbox_ids] - grid_pos_ids[particle_id]
        diff_pos[diff_pos > self.half_subbox] -= self.shape
        diff_pos[diff_pos < -self.half_subbox] += self.shape

        idx = np.lexsort((diff_pos[:, 2], diff_pos[:, 1], diff_pos[:, 0]))
        sorted_ids = subbox_ids[idx]
        return sorted_ids

    def ids_in_subbox(self, particle_id):
        xx, yy, zz = self.coords_3d
        ids_subbox = self.get_ids_in_subgrid_around_particle_id(particle_id, xx, yy, zz)
        ids_sorted_subbox = self.sort_ids_correct_order_subbox(particle_id, ids_subbox)
        return ids_sorted_subbox

    # Get qty in subbox in shape (self.subbox_shape, self.subbox_shape, self.subbox_shape)
    # around particle ID

    def from_ids_to_delta_in_subbox(self, snapshot, particle_id, subbox_sorted_ids):
        qty_all = snapshot[self.qty]
        grid = qty_all[subbox_sorted_ids].reshape(self.subbox_shape)

        center_subbox = self.half_subbox
        assert grid[center_subbox, center_subbox, center_subbox] == qty_all[particle_id], \
            "Particle ID " + str(particle_id) + " is not at the center of the sub-box, something went wrong."
        return grid

    def get_qty_in_subbox(self, particle_id):
        snapshot = self.snapshot
        ids_subbox = self.ids_in_subbox(particle_id)
        qty_ids = self.from_ids_to_delta_in_subbox(snapshot, particle_id, ids_subbox)
        return qty_ids

    # Compute and save subbox

    def compute_and_save_subbox_particle(self, particle_id, path):
        delta_sub = self.get_qty_in_subbox(particle_id)
        np.save(path + "/" + str(particle_id) + "/subbox_51_particle_" + str(particle_id) + ".npy", delta_sub)
        return delta_sub

    # If you don't want to this in the cheap way -- which assumes that the density estimate of a grid point is
    # equivalent to that of the particle at that grid location (this is valid because in the ICs there is basically
    # one particle per grid point) -- then you can actually estimate the density in a 3D box surrounding the particle
    #  using SPH.

    def get_sph_particle(self, particle_id):
        sim = self.snapshot
        qty = self.qty
        resolution = self.subbox_shape[0]
        width = float(sim.properties['boxsize']/self.shape * resolution)

        pynbody.analysis.halo.center(sim[particle_id], vel=False, wrap=True)
        subbox_sph = self.get_sph_on_3dgrid(sim, width=width, resolution=resolution, qty=qty)
        return subbox_sph

    def get_sph_on_3dgrid(self, sim, width=200., resolution=51, qty="delta"):
        x2 = width / 2
        xy_units = sim["pos"].units
        grid_data = pynbody.sph.to_3d_grid(sim, qty=qty, nx=resolution, x2=x2, xy_units=xy_units)
        return grid_data
    # def boundary_condition_on_z(x_condition, y_condition, case3, case03, z_coords, z_pos, one_sided_neighbours, box_shape):
    #     if case3:
    #         lz = box_shape - z_pos
    #         extra_z = one_sided_neighbours - lz
    #         z_cond = z_coords < extra_z
    #         extra_cond = x_condition & y_condition & z_cond
    #
    #     elif case03 is True:
    #         extra_z = one_sided_neighbours - z_pos
    #         z_cond = z_coords > box_shape - extra_z
    #         extra_cond = x_condition & y_condition & z_cond
    #
    #     else:
    #         extra_cond = x_condition & y_condition
    #     return extra_cond
    #
    #
    # def boundary_condition_on_y(x_condition, case2, case02, case3, case03,
    #                             y_coords, z_coords, y_pos, z_pos, one_sided_neighbours, box_shape):
    #     if case2:
    #         ly = box_shape - y_pos
    #         extra_y = one_sided_neighbours - ly
    #         y_cond = y_coords < extra_y
    #
    #         extra_cond = boundary_condition_on_z(x_condition, y_cond, case3, case03, z_coords, z_pos,
    #                                              one_sided_neighbours, box_shape)
    #
    #     elif case02:
    #         print("true")
    #         extra_y = one_sided_neighbours - y_pos
    #         y_cond = y_coords > box_shape - extra_y
    #
    #         extra_cond = boundary_condition_on_z(x_condition, y_cond, case3, case03, z_coords, z_pos,
    #                                              one_sided_neighbours, box_shape)
    #
    #     else:
    #         if case3 is True:
    #             lz = box_shape - z_pos
    #             extra_z = one_sided_neighbours - lz
    #             z_cond = z_coords < extra_z
    #             extra_cond = x_condition & z_cond
    #
    #         elif case03 is True:
    #             extra_z = one_sided_neighbours - z_pos
    #             z_cond = z_coords > box_shape - extra_z
    #             extra_cond = x_condition & z_cond
    #
    #         else:
    #             extra_cond = x_condition
    #     return extra_cond
    #
    #
    # def deal_with_particle_near_boundaries(x_coords, y_coords, z_coords,
    #                                        x_pos, y_pos, z_pos, one_sided_neighbours, box_shape):
    #     case1 = x_pos[0] + one_sided_neighbours > box_shape
    #     case01 = x_pos[0] - one_sided_neighbours < 0
    #     print(case1)
    #     print(case01)
    #
    #     case2 = y_pos[0] + one_sided_neighbours > box_shape
    #     case02 = y_pos[0] - one_sided_neighbours < 0
    #     print(case2)
    #     print(case02)
    #
    #     case3 = z_pos[0] + one_sided_neighbours > box_shape
    #     case03 = z_pos[0] - one_sided_neighbours < 0
    #     print(case3)
    #     print(case03)
    #
    #     if case1:
    #         print(case1)
    #         print("Case1 is true")
    #         lx = box_shape - x_pos
    #         extra_x = one_sided_neighbours - lx
    #         x_cond = x_coords < extra_x
    #
    #         extra_cond = boundary_condition_on_y(x_cond, case2, case02, case3, case03, y_coords, z_coords, y_pos, z_pos,
    #                                              one_sided_neighbours, box_shape)
    #
    #     elif case01:
    #         print(case01)
    #         print("Case 01 is true")
    #         extra_x = one_sided_neighbours - x_pos
    #         x_cond = x_coords > box_shape - extra_x
    #
    #         extra_cond = boundary_condition_on_y(x_cond, case2, case02, case3, case03, y_coords, z_coords, y_pos, z_pos,
    #                                              one_sided_neighbours, box_shape)
    #
    #     else:
    #         print("Neither case1 nor case01 is true")
    #         if case2:
    #             ly = box_shape - y_pos
    #             extra_y = one_sided_neighbours - ly
    #             y_cond = y_coords < extra_y
    #
    #             if case3:
    #                 lz = box_shape - z_pos
    #                 extra_z = one_sided_neighbours - lz
    #                 z_cond = z_coords < extra_z
    #                 extra_cond = y_cond & z_cond
    #
    #             elif case03:
    #                 extra_z = one_sided_neighbours - z_pos
    #                 z_cond = z_coords > box_shape - extra_z
    #                 extra_cond = y_cond & z_cond
    #
    #             else:
    #                 extra_cond = y_cond
    #
    #         elif case02:
    #             extra_y = one_sided_neighbours - y_pos
    #             y_cond = y_coords > box_shape - extra_y
    #
    #             if case3:
    #                 lz = box_shape - z_pos
    #                 extra_z = one_sided_neighbours - lz
    #                 z_cond = z_coords < extra_z
    #                 extra_cond = y_cond & z_cond
    #
    #             elif case03:
    #                 extra_z = one_sided_neighbours - z_pos
    #                 z_cond = z_coords > box_shape - extra_z
    #                 extra_cond = y_cond & z_cond
    #
    #             else:
    #                 extra_cond = y_cond
    #
    #         else:
    #
    #             if case3:
    #                 lz = box_shape - z_pos
    #                 extra_z = one_sided_neighbours - lz
    #                 z_cond = z_coords < extra_z
    #                 extra_cond = z_cond
    #
    #             elif case03:
    #                 extra_z = one_sided_neighbours - z_pos
    #                 z_cond = z_coords > box_shape - extra_z
    #                 extra_cond = z_cond
    #
    #             else:
    #                 raise ValueError("Nothing returned from here -- at least one of these if functions must be right!")
    #
    #     return extra_cond
    #
    #
    # def get_ids_in_subgrid_around_particle_id(particle_id, subbox_shape, ids_in_3d_grid, x_coords, y_coords, z_coords,
    #                                           box_shape, periodic=True):
    #     one_sided_neighbours = (subbox_shape - 1) / 2
    #     x_pos, y_pos, z_pos = np.where(ids_in_3d_grid == particle_id)
    #
    #     cond_xyz = (abs(x_coords - x_pos[0]) <= one_sided_neighbours) & \
    #                (abs(y_coords - y_pos[0]) <= one_sided_neighbours) & \
    #                (abs(z_coords - z_pos[0]) <= one_sided_neighbours)
    #     if periodic is True:
    #         if any([x_pos[0] + one_sided_neighbours > box_shape,
    #                 y_pos[0] + one_sided_neighbours > box_shape,
    #                 z_pos[0] + one_sided_neighbours > box_shape,
    #                 x_pos[0] - one_sided_neighbours < 0,
    #                 y_pos[0] - one_sided_neighbours < 0,
    #                 z_pos[0] - one_sided_neighbours < 0]):
    #             print("Particle near edge of the box, implementing boundary condition fix")
    #             # print(x_pos[0] - one_sided_neighbours < 0)
    #             extra_cond = deal_with_particle_near_boundaries(x_coords, y_coords, z_coords,
    #                                                             x_pos, y_pos, z_pos, one_sided_neighbours, box_shape)
    #             cond_xyz = cond_xyz & extra_cond
    #
    #         else:
    #             pass
    #
    #     n_neighbours_coords = np.where(cond_xyz)
    #     return ids_in_3d_grid[n_neighbours_coords]