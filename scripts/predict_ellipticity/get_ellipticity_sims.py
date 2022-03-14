import numpy as np
import pynbody
import ellipticity as ge


if __name__ == "__main__":
    sims = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]

    for i in range(len(sims)):
        sim = sims[i]

        path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
        saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

        f = pynbody.load(path_sim + "IC.gadget2")
        f.physical_units()

        scale0 = pynbody.array.SimArray(2., "Mpc h**-1")  # z = 0
        scale0 *= f.properties["a"]
        scale0.sim = f

        scale1 = pynbody.array.SimArray(12., "Mpc h**-1")  # z = 0
        scale1 *= f.properties["a"]
        scale1.sim = f

        for scale in [scale0, scale1]:
            sp0 = ge.ShearProperties(f, scale, number_of_processors=40)
            np.save(saving_path + "reseed" + sim + "_ellipticity_scale_%.2f.npy" % float(scale), sp0.ellipticity)
            np.save(saving_path + "reseed" + sim + "_prolateness_scale_%.2f.npy" % float(scale), sp0.prolateness)
            np.save(saving_path + "reseed" + sim + "_densub_ellipticity_scale_%.2f.npy" % float(scale), sp0.density_subtracted_ellipticity)
            np.save(saving_path + "reseed" + sim + "_densub_prolateness_scale_%.2f.npy" % float(scale), sp0.density_subtracted_prolateness)
