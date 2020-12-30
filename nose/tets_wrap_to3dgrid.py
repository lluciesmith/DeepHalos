import pynbody
import numpy as np

######## TEST 1 ###############

f = pynbody.load("Documents/mlhalos_files/Nina-Simulations/double/snapshot_104")
f.physical_units()
pynbody.analysis.halo.center(f[1605031], vel=False, wrap=True)

rho_m = pynbody.analysis.cosmology.rho_M(f, unit=f["rho"].units)
delta = f["rho"]/rho_m
f["delta"] = np.log10(delta)

resolution_grid = 100
cen = resolution_grid/2
rho_grid = pynbody.sph.to_3d_grid(f, qty="delta", nx=resolution_grid, threaded=True)

rho_middle_no_wrap = np.load("rho_middle_centre_1605031.npy")
np.testing.assert_allclose(rho_grid[cen-10:cen+10, cen-10:cen+10, cen-10:cen+10],  rho_middle_no_wrap, rtol=1e-05)

rho_edge = np.load("rho_edge_centre_1605031.npy")
edge = 90
assert not np.allclose(rho_grid[edge-10:edge+10, edge-10:edge+10, edge-10:edge+10],  rho_edge, rtol=1e-05)

rho_edge = np.load("rho_edge_centre_1605031_w_manual_wrapping.npy")


# In order to construct the saved `rho_middle_centre_1605031.npy` array, I ran (using the master branch):
#
# f = pynbody.load("Documents/mlhalos_files/Nina-Simulations/double/snapshot_104")
# f.physical_units()
# pynbody.analysis.halo.center(f[1605031], vel=False, wrap=True)
#
# rho_m = pynbody.analysis.cosmology.rho_M(f, unit=f["rho"].units)
# delta = f["rho"]/rho_m
# f["delta"] = np.log10(delta)
#
# resolution_grid = 100
# cen = resolution_grid/2
# rho_grid = pynbody.sph.to_3d_grid(f, qty="delta", nx=resolution_grid, threaded=True)
# rho_middle_no_wrap = rho_grid[cen-10:cen+10, cen-10:cen+10, cen-10:cen+10]


def get_box_around_particle(input_matrix, p_id, coords, width_box, shape_grid):
    i0 = coords[p_id][0] - width_box //2
    j0 = coords[p_id][1] - width_box // 2
    k0 = coords[p_id][2] - width_box // 2

    box_id = np.zeros((width_box, width_box, width_box))
    for i in range(width_box):
        for j in range(width_box):
            for k in range(width_box):
                box_id[i, j, k] = input_matrix[(i + i0) % shape_grid, (j + j0) % shape_grid, (k + k0) % shape_grid]
    return box_id

# Automatic wrapping -- make sure you are on developing branch

particle_id = 15110404
width = 31
shape_input = 500

f = pynbody.load("Documents/mlhalos_files/Nina-Simulations/double/snapshot_104")
f.physical_units()
rho_m = pynbody.analysis.cosmology.rho_M(f, unit=f["rho"].units)
delta = f["rho"]/rho_m
f["delta"] = np.log10(delta)
rho_grid = pynbody.sph.to_3d_grid(f, qty="delta", nx=shape_input, threaded=True)

boxsize = float(f.properties['boxsize'].in_units(f['pos'].units))
grid_spacing = boxsize/shape_input
grid_coords = (f["pos"]/grid_spacing).astype('int')
grid_coords -= grid_coords.min()

box_particle_auto = get_box_around_particle(rho_grid, particle_id, grid_coords, width, shape_input)
np.save("box_auto_wrapping_15110404.npy", box_particle_auto)

manual_wrap = np.load("box_manual_wrapping_15110404.npy")
np.testing.assert_allclose(manual_wrap, box_particle_auto, rtol=1e-5)

# Manual wrapping -- make sure you are on master branch

particle_id = 15110404
width = 31
shape_input = 500

f = pynbody.load("Documents/mlhalos_files/Nina-Simulations/double/snapshot_104")
f.physical_units()
pynbody.analysis.halo.center(f[particle_id], vel=False, wrap=True)
f.wrap()
rho_grid = pynbody.sph.to_3d_grid(f, qty="rho", nx=shape_input, threaded=True)

boxsize = float(f.properties['boxsize'].in_units(f['pos'].units))
grid_spacing = boxsize/shape_input
grid_coords = (f["pos"]/grid_spacing).astype('int')
grid_coords -= grid_coords.min()

box_particle_manual = get_box_around_particle(rho_grid, particle_id, grid_coords, width, shape_input)
np.save("box_manual_wrapping_15110404.npy", box_particle_manual)

# No wrapping

f = pynbody.load("Documents/mlhalos_files/Nina-Simulations/double/snapshot_104")
f.physical_units()
rho_m = pynbody.analysis.cosmology.rho_M(f, unit=f["rho"].units)
delta = f["rho"]/rho_m
f["delta"] = np.log10(delta)
rho_grid = pynbody.sph.to_3d_grid(f, qty="delta", nx=shape_input, threaded=True)

boxsize = float(f.properties['boxsize'].in_units(f['pos'].units))
grid_spacing = boxsize/shape_input
grid_coords = (f["pos"]/grid_spacing).astype('int')
grid_coords -= grid_coords.min()

box_particle_no_wrap = get_box_around_particle(rho_grid, particle_id, grid_coords, width, shape_input)




