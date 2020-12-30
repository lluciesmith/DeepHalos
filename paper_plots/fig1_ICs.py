import pynbody
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    path = '/Users/lls/Documents/deep_halos_files/simulations/standard_reseed4/'

    f0 = pynbody.load(path + "snapshot_099")
    h = f0.halos()

    comoving_width_f0_int = 2.5
    comoving_width_f0 = str(comoving_width_f0_int) + " Mpc a h**-1"
    comoving_width_ics_int = 30
    comoving_width_ics = str(comoving_width_ics_int) + " Mpc a h**-1"

    for h_num in [0, 50, 400]:
        halo_id = h[h_num]
        particle_id = h[h_num]['iord'][0]

        # z=0

        _ = pynbody.analysis.halo.center(f0[particle_id], vel=False, wrap=True)
        sub_f = f0[abs(f0["z"]) <= comoving_width_f0_int / 2]
        im = pynbody.plot.sph.image(sub_f, width=comoving_width_f0, resolution=500, av_z=True, cmap="Greys")
        plt.imsave("/Users/lls/Desktop/figs_paper/sim_figs/z0_halo_" + str(h_num) + ".png", np.log10(im),
                    cmap="Greys", origin='lower')

        # ics

        s = pynbody.load(path + "IC.gadget3")
        _ = pynbody.analysis.halo.center(s[particle_id], vel=False, wrap=True)
        s["d"] = s["rho"]/np.mean(s["rho"])

        s_xy = s[abs(s["z"]) <= comoving_width_ics_int / 2]
        im_xy = pynbody.plot.sph.image(s_xy, width=comoving_width_ics, resolution=500, av_z=True, cmap="binary", qty="d")
        plt.imsave("/Users/lls/Desktop/figs_paper/sim_figs/ICs_halo_" + str(h_num) + "_xy_projection.png", im_xy,
                   cmap="binary", origin='lower')
        plt.clf()

        tr = s.rotate_x(90)
        s_xz = s[abs(s["z"]) <= comoving_width_ics_int / 2]
        im_xz = pynbody.plot.sph.image(s_xz, width=comoving_width_ics, resolution=500, av_z=True, cmap="binary",
                                       qty="d")
        plt.imsave("/Users/lls/Desktop/figs_paper/sim_figs/ICs_halo_" + str(h_num) + "_xz_projection.png", im_xz,
                   cmap="binary", origin='lower')
        plt.clf()

        s.rotate_x(270)
        s.rotate_y(90)
        s_zy = s[abs(s["z"]) <= comoving_width_ics_int / 2]
        s_zy = pynbody.plot.sph.image(s_zy, width=comoving_width_ics, resolution=500, av_z=True, cmap="binary", qty="d")
        plt.imsave("/Users/lls/Desktop/figs_paper/sim_figs/ICs_halo_" + str(h_num) + "_zy_projection.png", s_zy,
                   cmap="binary", origin='lower')
        plt.clf()


