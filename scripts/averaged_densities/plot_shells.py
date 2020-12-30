import matplotlib.pyplot as plt
import numpy as np
import pynbody
import sys; sys.path.append("/Users/lls/Documents/Projects/DeepHalos/")
from dlhalos_code import data_processing as dp
from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {'rescale_mean': 1.005, 'rescale_std': 0.05050, 'shape_sim': 256, 'num_shells': 20}

# Prepare simulation

ics = pynbody.load("/Users/lls/Documents/mlhalos_files/Nina-Simulations/double/ICs_z99_256_L50_gadget3.dat")
ics.physical_units()
f0 = pynbody.load("/Users/lls/Documents/mlhalos_files/Nina-Simulations/double/snapshot_104")
f0.physical_units()
h = f0.halos(make_grp=True)

rho_m = pynbody.analysis.cosmology.rho_M(ics, unit=ics["rho"].units)
den_con = ics["rho"] / rho_m
d_rescaled = ((den_con - params['rescale_mean'])/params['rescale_std']).reshape((params['shape_sim'],
                                                                                 params['shape_sim'], params['shape_sim']))

i, j, k = np.unravel_index(ics["iord"], (params['shape_sim'], params['shape_sim'], params['shape_sim']))
coords = np.column_stack((i, j, k))

# Prepare compute boxes for particle ID

params['particle_id']= 10434650
params['res'] = 75

i0, j0, k0 = coords[params['particle_id']]
out_m = np.zeros((params['res'], params['res'], params['res']))
s = dp.compute_subbox(i0, j0, k0, params['res'], d_rescaled, out_m, 256)
shell_labels = dp.assign_shell_to_pixels(params['res'], params['num_shells'])
s_av = dp.get_spherically_averaged_box(s, shell_labels)

# Plot

pixel_length = 50*1000*0.01/0.701/256
box_length = pixel_length * params['res']

# Do it with pynbody if you want higher resolution

_ = pynbody.analysis.halo.center(ics[params['particle_id']], vel=False, wrap=True)
ics['rescaled'] = d_rescaled.flatten()

im = pynbody.plot.sph.image(ics, qty="rescaled", width=box_length, resolution=500, av_z=False, cmap="Greys",
                            log=False, vmin=-3.2, vmax=3.2)
plt.imsave("/Users/lls/Desktop/figs_paper/sim_figs/fig3_ICs.png", im,
           cmap="binary", origin='lower')

im_av = plt.imshow(s_av[:,:, 37].T[::-1,:], extent=(-box_length/2, box_length/2, -box_length/2, box_length/2),
                   cmap="Greys")
plt.imsave("/Users/lls/Desktop/figs_paper/sim_figs/fig3_ICs_averaged.png", s_av[:,:, 37].T[::-1,:],
           cmap="binary", origin='lower')

# f, ax = plt.subplots()
# im = pynbody.plot.sph.image(ics, qty="rescaled", width=box_length, resolution=500, av_z=False, cmap="RdBu",
#                             ret_im=True, log=False, subplot=ax, vmin=-3, vmax=3)
# ax.set_ylabel(r"$y/\mathrm{kpc}$", fontsize=22)
# ax.set_xlabel(r"$x/\mathrm{kpc}$", fontsize=22)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cb = plt.colorbar(im, cax=cax)
# cb.set_label(label="Density contrast (rescaled)", fontsize=22)
# plt.subplots_adjust(left=0.01, bottom=0.14, right=1.)
# plt.savefig("/Users/lls/Documents/Papers/dlhalos_paper/raw_input.pdf")
#
#
# # Actual resolution
#
# f, ax = plt.subplots()
# im = ax.imshow(s[:, :, 37].T[::-1, :], extent=(-box_length / 2, box_length / 2, -box_length / 2, box_length / 2),
#                cmap='RdBu')
# ax.set_ylabel(r"$y/\mathrm{kpc}$")
# ax.set_xlabel(r"$x/\mathrm{kpc}$")
#
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax, label="Density contrast (rescaled)")
# plt.subplots_adjust(left=0.01, bottom=0.14, right=1.)
#
# # Averaged
#
# f1, ax1 = plt.subplots()
# max_val = s_av[:,:, 37].T[::-1,:].max()
# c1 = ax1.imshow(s_av[:,:, 37].T[::-1,:], extent=(-box_length/2, box_length/2, -box_length/2, box_length/2),
#                     cmap='RdBu', vmin=-max_val, vmax=max_val)
# ax1.set_ylabel(r"$y/\mathrm{kpc}$", fontsize=22)
# ax1.set_xlabel(r"$x/\mathrm{kpc}$", fontsize=22)
#
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cb2 = plt.colorbar(c1, cax=cax)
# cb2.set_label(label="Density contrast (rescaled)", fontsize=22)
# plt.subplots_adjust(left=0.01, bottom=0.14, right=1.)
# plt.savefig("/Users/lls/Documents/Papers/dlhalos_paper/averaged_input.pdf")


# f, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), sharex=True, sharey=True)
#
# for ax in axes[:1]:
#     #ax = plt.gca()
#     im = ax.imshow(s[:,:, 37].T[::-1,:], extent=(-box_length/2, box_length/2, -box_length/2, box_length/2), cmap='RdBu')
#     ax.set_ylabel(r"$y/\mathrm{kpc}$")
#     ax.set_xlabel(r"$x/\mathrm{kpc}$")
#
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(im, cax=cax)
#
# # r_shells = np.linspace(pixel_length, box_length / 2, params['num_shells'], endpoint=True)
# # theta = np.linspace(0, 2*np.pi, 100)
# # for r in r_shells:
# #     x1 = r*np.cos(theta)
# #     x2 = r*np.sin(theta)
# #     axes[1].plot(x1, x2, color="k")
#
# c1 = axes[1].imshow(s_av[:,:, 37].T[::-1,:], extent=(-box_length/2, box_length/2, -box_length/2, box_length/2),
#                     cmap='RdBu')
# axes[1].set_ylabel(r"$y/\mathrm{kpc}$")
# axes[1].set_xlabel(r"$x/\mathrm{kpc}$")
#
# divider = make_axes_locatable(axes[2])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(c1, cax=cax)
#
# for ax in axes[1:]:
#     ax.set_ylabel("")
#
# plt.subplots_adjust(bottom=0.14, left=0.08, hspace=0, wspace=0.2)
# plt.savefig("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/averaged_box.pdf")