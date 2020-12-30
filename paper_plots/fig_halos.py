from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pynbody


def get_colorbar(z=0):
    im = Image.open('/Users/lls/Desktop/PastedGraphic-1.png')
    pix = im.load()
    if z == 0:
        rgb = np.array([(pix[i, 0][0], pix[i, 0][1], pix[i, 0][2]) for i in range(im.size[0])])/255
        cmap_lls = LinearSegmentedColormap.from_list('luisa_z' + str(z), rgb, N=3000)
    else:
        rgb = np.array([(pix[i, 0][0], pix[i, 0][1], pix[i, 0][2]) for i in range(int(im.size[0]/2))]) / 255
        cmap_lls = LinearSegmentedColormap.from_list('luisa_z' + str(z), rgb, N=3000)
    plt.register_cmap(cmap=cmap_lls)
    return cmap_lls

if __name__ == "__main__":
    get_colorbar(z=0)
    cmap_z0 = plt.get_cmap('luisa_z0')

    path = '/Users/lls/Documents/deep_halos_files/simulations/standard_reseed4/'

    f0 = pynbody.load(path + "snapshot_099")
    h = f0.halos()
    particle_id = h[50]['iord'][0]
    pos_h50 = f0[particle_id]['pos']

    pynbody.analysis.halo.center(h[50], vel=False, wrap=True)
    comoving_width = "50 Mpc h**-1"
    comoving_width = "3 Mpc h**-1"
    im2 = pynbody.plot.sph.image(f0, width=comoving_width, resolution=500, av_z=True, cmap=cmap_z0)
    plt.imsave("/Users/lls/Documents/talks/halo_yellow.png", np.log10(im2), cmap=cmap_z0, origin='lower')

    s = pynbody.load(path + "IC.gadget3")
    s['pos'] -= pos_h50
    s.wrap()

    get_colorbar(z=99)
    cmap_z99 = plt.get_cmap('luisa_z99')
    im_ics = pynbody.plot.sph.image(s, width=comoving_width, resolution=500, av_z=True, cmap=cmap_z99)
    plt.imsave("/Users/lls/Documents/talks/ICs_2_yellow.png", im_ics, cmap=cmap_z99, origin='lower')
