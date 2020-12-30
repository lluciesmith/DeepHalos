import pynbody
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    path = '/Users/lls/Documents/deep_halos_files/simulations/standard_reseed4/'

    f0 = pynbody.load(path + "snapshot_099")
    h = f0.halos()

    h_num = 50
    halo_id = h[h_num]
    particle_id = h[h_num]['iord'][0]
    pos0 = f0[particle_id]['pos']

    snapshots = ["snapshot_009", "snapshot_049", "snapshot_069"]
    comoving_width_int = 9.
    comoving_width = str(comoving_width_int) + " Mpc a h**-1"

    for i, snapshot in enumerate(snapshots):
        s = pynbody.load(path + snapshot)
        # _ = pynbody.analysis.halo.center(s[particle_id], vel=False, wrap=True)
        s['pos'] -= pos0
        s.wrap()

        sub_s = s[abs(s["z"]) <= comoving_width_int / 2]
        im = pynbody.plot.sph.image(sub_s, width=comoving_width, resolution=500, av_z=True, cmap="Greys")
        plt.imsave("/Users/lls/Desktop/figs_paper/sim_figs/pos0_z"+ str(int(np.round(s.properties['z']))) +
                   "_halo_" + str(h_num) + ".png", np.log10(im), cmap="Greys", origin='lower')



# import pynbody
# import numpy as np
# import pynbody.sph as sph
# import mayavi
# from mayavi import mlab
# from tvtk.util import ctf as ctfs
# from tvtk.util.ctf import PiecewiseFunction, ColorTransferFunction
# from matplotlib.pyplot import cm
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# from sphviewer.tools import cmaps as sphcmaps
#
#
# def hex_to_rgb(hex):
#     hex = hex.lstrip('#')
#     hlen = len(hex)
#     return tuple(int(hex[i:i + hlen // 3], 16)/255 for i in range(0, hlen, hlen // 3))
#
#
# def volume(sim, qty='rho', width=None, resolution=200,
#            color=None,vmin=None,vmax=None, cmap=None,
#            dynamic_range=4.0,log=True, bgcolor=(1., 1., 1.),
#            create_figure=True, size=(500,500), grid_data=None):
#     """Create a volume rendering of the given simulation using mayavi.
#
#     **Keyword arguments:**
#
#     *qty* (rho): The name of the array to interpolate
#
#     *width* (None): The width of the cube to generate, centered on the origin
#
#     *resolution* (200): The number of elements along each side of the cube
#
#     *color* (white): The color of the volume rendering. The value of each voxel
#        is used to set the opacity.
#
#     *vmin* (None): The value for zero opacity (calculated using dynamic_range if None)
#
#     *vmax* (None): The value for full opacity (calculated from the maximum
#        value in the region if None)
#
#     *dynamic_range*: The dynamic range to use if vmin and vmax are not specified
#
#     *log* (True): log-scale the image before passing to mayavi
#
#     *create_figure* (True): create a new mayavi figure before rendering
#     """
#     if grid_data is None:
#         grid_data = sph.to_3d_grid(sim,qty=qty,nx=resolution, x2=None if width is None else width/2)
#
#     if log:
#         grid_data = np.log10(grid_data)
#         if vmin is None:
#             vmin = grid_data.max()-dynamic_range
#         if vmax is None:
#             vmax = grid_data.max()
#     else:
#         if vmin is None:
#             vmin = np.min(grid_data)
#         if vmax is None:
#             vmax = np.max(grid_data)
#
#     grid_data[grid_data<vmin]=vmin
#     grid_data[grid_data>vmax]=vmax
#
#     otf = PiecewiseFunction()
#     otf.add_point(vmin,0.7)
#     otf.add_point(vmax,1.0)
#
#     if create_figure:
#         fig = mlab.figure(size=size, bgcolor=bgcolor)
#
#     sf = mayavi.tools.pipeline.scalar_field(grid_data)
#     V = mlab.pipeline.volume(sf,color=None,vmin=vmin,vmax=vmax)
#
#     V.trait_get('volume_mapper')['volume_mapper'].blend_mode = 'maximum_intensity'
#     if color == "luisa":
#         ar = ["CBCBCB","C4C4C4","BDBDBD","B6B6B6","AFAFAF","A8A8A8","A1A1A1","9A9A9A","939393","8C8C8C","858585",
#               "7E7E7E","777777","707070","696969","626262","5B5B5B","545454","4D4D4D","464646","3F3F3F","383838",
#               "313131","2A2A2A","232323","1C1C1C","151515","0E0E0E","070707","000000",
#               "090700","110E00","1A1600", "231D00","2B2400","342B00","3D3300","453A00","4E4100","574800", "5F5000",
#               "685700","715E00","796500","826D00","8A7400","937B00","9C8200","A48A00","AD9100","B69800", "BE9F00",
#               "C7A700","D0AE00","D8B500","E1BC00","EAC400","F2CB00","FBD200"]
#         colors = [hex_to_rgb(ari) for ari in ar]
#         cmap_lls = LinearSegmentedColormap.from_list('luisa', colors, N=100)
#         plt.register_cmap(cmap=cmap_lls)
#         cmap = 'luisa'
#
#     elif color == "desert":
#         cmap_jon = sphcmaps.desert(Nbins=100)
#         plt.register_cmap(cmap=cmap_jon)
#         cmap = cmap_jon.name
#     else:
#         cmap = color
#
#     # save the color transfer function of the current volume
#     c = ctfs.save_ctfs(V._volume_property)
#
#     # change the alpha channel as needed
#     c['alpha'][1][1] = 1.
#     c['alpha'][0][1] = 1.
#
#     # change the color points to another color scheme
#     v_values = np.linspace(vmin, vmax, num=100, endpoint=True)
#     rgb = cm.get_cmap(plt.get_cmap(cmap))(np.linspace(0.0, 1.0, 100))[:, :3]
#     c['rgb'] = [[v_values[i], a[0], a[1], a[2]] for i, a in enumerate(rgb)]
#
#     # update the color transfer function
#     ctfs.load_ctfs(c, V._volume_property)
#     V.update_ctf = True
#
#     V._otf = otf
#     V._volume_property.set_scalar_opacity(otf)
#     return V
#
#
# if __name__ == "__main__":
#     path = '/Users/lls/Documents/deep_halos_files/simulations/standard_reseed4/'
#     f0 = pynbody.load(path + "snapshot_099")
#     h = f0.halos()
#
#     halo_id = h[50]
#     particle_id = h[50]['iord'][0]
#
#     bgcolor= hex_to_rgb("CBCBCB")
#
#     # z=0
#
#     c0 = pynbody.analysis.halo.center(f0[particle_id], vel=False, wrap=True)
#
#     comoving_width_int = 0.8
#     dynamic_range = 3.8
#     cmap_b = cm.get_cmap('binary')
#     im0_v = volume(f0, width=comoving_width_int, resolution=100, color=cmap_b, bgcolor=bgcolor,
#                    dynamic_range=dynamic_range)
#     mayavi.mlab.savefig("/Users/lls/Desktop/figs_paper/sim_figs/z0_h50_BW.png")
#
#
#     # other redshifts
#
#     snapshots = ["snapshot_009", "snapshot_049", "snapshot_069"]
#
#     comoving_width_ints = [5.5, 5.5, 5.5]
#     dynamic_ranges = [3., 4.2, 4.]
#
#     for i, snapshot in enumerate(snapshots):
#         comoving_width_int = comoving_width_ints[i]
#         dynamic_range = dynamic_ranges[i]
#
#         s = pynbody.load(path + snapshot)
#         _ = pynbody.analysis.halo.center(s[particle_id], vel=False, wrap=True)
#
#         s_v = volume(s, width=comoving_width_int, resolution=100, color="bla", bgcolor=bgcolor,
#                      dynamic_range=dynamic_range)
#         mayavi.mlab.savefig("/Users/lls/Desktop/figs_paper/sim_figs/z" + str(np.round(s.properties['z'])) + "_h50_BW.png")
#         mayavi.mlab.clf()

# path = '/Users/lls/Documents/deep_halos_files/simulations/standard_reseed4/'
# f0 = pynbody.load(path + "snapshot_099")
# h = f0.halos()
#
# for h_id in [0, 400]:
#     halo_id = h[h_id]
#     particle_id = h[h_id]['iord'][0]
#
#     bgcolor = hex_to_rgb("CBCBCB")
#
#     # z=0
#
#     c0 = pynbody.analysis.halo.center(f0[particle_id], vel=False, wrap=True)
#
#     comoving_width_int = 0.8
#     dynamic_range = 3.8
#     cmap_b = cm.get_cmap('binary')
#     im0_v = volume(f0, width=comoving_width_int, resolution=200, color=cmap_b, bgcolor=bgcolor,
#                    dynamic_range=dynamic_range)
#     mayavi.mlab.savefig("/Users/lls/Desktop/figs_paper/sim_figs/z0_h" + str(h_id) + "_BW.png")



