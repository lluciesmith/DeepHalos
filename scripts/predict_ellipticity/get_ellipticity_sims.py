import numpy as np
import pynbody
import ellipticity as ge


if __name__ == "__main__":
    # sims = ["0", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    sims = ["1", "2", "4", "5", "6", "7", "8", "9", "10"]

    for i in range(len(sims)):
        sim = sims[i]
        if sim == "0":
            path_sim = "/share/hypatia/lls/simulations/Nina_sim/ICs_z99_256_L50_gadget3.dat"
            saving_path = "/share/hypatia/lls/deep_halos/training_sim/"
        elif sim == "1":
            path_sim = "/share/hypatia/lls/simulations/reseed50/IC.gadget3"
            saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"
        elif sim == "2":
            path_sim = "/share/hypatia/lls/simulations/reseed50_2/IC_doub_z99_256.gadget3"
            saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"
        elif sim == "4" or sim == "5":
            path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/IC.gadget3"
            saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"
        else:
            path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/IC.gadget2"
            saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

        f = pynbody.load(path_sim)
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
