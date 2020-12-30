import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def get_light_dark_light_cmap(l0=(1., 1., 1.), l1=(0., 0., 0.),
                              l2=(255/255, 209/255, 0/255), num0 = 200, num1=200):

    c0 = interpolate_colors(l0, l1, num0)
    c1 = interpolate_colors(l1, l2, num1)
    c_all = [col for sublist in [c0, c1] for col in sublist]

    cmap_interp = LinearSegmentedColormap.from_list('luisa', c_all)
    return cmap_interp


def interpolate_colors(color1, color2, num=200):
    r0, g0, b0 = (color1[0]*255, color1[1]*255, color1[2]*255)
    r1, g1, b1 = (color2[0]*255, color2[1]*255, color2[2]*255)

    fraction = np.linspace(0, 1, num, endpoint=True)
    colors = [(((r1 - r0) * fr + r0)/255, ((g1 - g0) * fr + g0)/255, ((b1 - b0) * fr + b0)/255)
              for fr in fraction]
    return colors


def interpolate(color1, color2, num=200):
    colors = interpolate_colors(color1, color2, num)
    cmap_interp = LinearSegmentedColormap.from_list('luisa', colors)
    return cmap_interp


def get_luisa_colormap():
    ar = ["CBCBCB", "C4C4C4", "BDBDBD", "B6B6B6", "AFAFAF", "A8A8A8", "A1A1A1", "9A9A9A", "939393", "8C8C8C", "858585",
          "7E7E7E", "777777", "707070", "696969", "626262", "5B5B5B", "545454", "4D4D4D", "464646", "3F3F3F", "383838",
          "313131", "2A2A2A", "232323", "1C1C1C", "151515", "0E0E0E", "070707", "000000",
          "090700", "110E00", "1A1600", "231D00", "2B2400", "342B00", "3D3300", "453A00", "4E4100", "574800", "5F5000",
          "685700", "715E00", "796500", "826D00", "8A7400", "937B00", "9C8200", "A48A00", "AD9100", "B69800", "BE9F00",
          "C7A700", "D0AE00", "D8B500", "E1BC00", "EAC400", "F2CB00", "FBD200"]
    colors = [hex_to_rgb(ari) for ari in ar]
    cmap_lls = LinearSegmentedColormap.from_list('luisa', colors, N=100)
    return cmap_lls


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i + hlen // 3], 16)/255 for i in range(0, hlen, hlen // 3))