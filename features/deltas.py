import sys; sys.path.append("/home/lls/mlhalos_code/")
import numpy as np
from mlhalos import parameters
import pynbody

def delta_property(snapshot):
    rho = snapshot["rho"]
    mean_rho = pynbody.analysis.cosmology.rho_M(snapshot, unit=rho.units)
    snapshot["delta"] = rho / mean_rho
    snapshot["delta"].units = "1"
    return snapshot["delta"]


path_sim = "/home/lls/stored_files/Nina-Simulations/double/"
ic1 = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "ICs_z99_256_L50_gadget3.dat",
                                                        final_snapshot=path_sim + "snapshot_104",
                                                        load_final=True)
path_sim = "/share/data1/lls/reseed50/simulation/"
ic_r = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC.gadget3",
                                             final_snapshot=path_sim + "snapshot_099", load_final=True)
path_sim = "/share/hypatia/app/luisa/reseed2/"
ic_r2 = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC_doub_z99_256.gadget3",
                                                        final_snapshot=path_sim + "snapshot_099",
                                                        load_final=True)
path_sim = "/share/hypatia/app/luisa/standard_reseed3/"
ic_r3 = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC.gadget3",
                                                        final_snapshot=path_sim + "snapshot_099",
                                                        load_final=True)
path_sim = "/share/hypatia/app/luisa/standard_reseed4/"
ic_r4 = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC.gadget3",
                                                        final_snapshot=path_sim + "snapshot_099",
                                                        load_final=True)
path_sim = "/share/hypatia/app/luisa/standard_reseed5/"
ic_r5 = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC.gadget3",
                                                        final_snapshot=path_sim + "snapshot_099",
                                                        load_final=True)
ics = [ic1, ic_r, ic_r2, ic_r3, ic_r4, ic_r5]
deltas = []
for ic in ics:
    delta = ic.initial_conditions["rho"].in_units("Msol Mpc**-3")/ic.mean_density
    deltas.append(delta)
np.save("/home/lls/deltas_6_sims.npy", np.array(deltas))


############ redshift z=2.1 ############

d_z2 = []
for path in ["/share/hypatia/app/luisa/reseed2/", "/share/hypatia/app/luisa/standard_reseed3/",
             "/share/hypatia/app/luisa/standard_reseed4/", "/share/hypatia/app/luisa/standard_reseed5/"]:
    s0 = pynbody.load(path + "snapshot_099")
    print(s0.properties["a"])
    s0.physical_units()
    d0 = delta_property(s0)
    d_z2.append(d0)

d_all = np.concatenate(d_z2)
logd = np.log10(d_all)
mean_d = np.mean(d_all)
std_d = np.std(d_all)
print(mean_d)
print(std_d)

