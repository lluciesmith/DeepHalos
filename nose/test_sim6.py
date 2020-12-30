from mlhalos import parameters
import pynbody
import matplotlib.pyplot as plt
import numpy as np


def plot_bridge(initial_condition, final_snapshot, order=True, monotonic=True, halo_id=100, title="Order Bridge"):
    if order is True:
        if monotonic is True:
            b = pynbody.bridge.OrderBridge(initial_condition, final_snapshot)
        else:
            b = pynbody.bridge.OrderBridge(initial_condition, final_snapshot, monotonic=False)
    else:
        b = pynbody.bridge.Bridge(initial_condition, final_snapshot)

    h = final_snapshot.halos()
    hid = h[halo_id]
    hid_ics = b(hid)
    assert np.allclose(h100_ics["iord"], h100["iord"])

    pynbody.analysis.halo.center(hid_ics, vel=False, wrap=True)
    width = float(ics.properties["boxsize"])
    plt.figure()
    pynbody.plot.sph.image(hid_ics, width=width, resolution=500, av_z=True, log=True, title=title)
    plt.subplots_adjust(top=0.9, left=0.05, right=0.9, bottom=0.14)

if __name__ == "__main__":

    # simulation-0

    ic_params = parameters.InitialConditionsParameters(path="/Users/lls/Documents/mlhalos_files/")
    ics = ic_params.initial_conditions
    f = ic_params.final_snapshot
    plot_bridge(ics, f, order=True, monotonic=False, title="Halo-0, sim-0", halo_id=0)

    # simulation-6

    ics = pynbody.load("/Users/lls/Documents/mlhalos_files/reseed6/IC.gadget2")
    f = pynbody.load("/Users/lls/Documents/mlhalos_files/reseed6/snapshot_007")
    ics.physical_units()
    f.physical_units()
    plot_bridge(ics, f, order=True, monotonic=False, title="Halo-0, sim-6", halo_id=0)

    # simulation-7

    ics = pynbody.load("/Users/lls/Documents/mlhalos_files/reseed7/IC.gadget2")
    f = pynbody.load("/Users/lls/Documents/mlhalos_files/reseed7/snapshot_007")
    ics.physical_units()
    f.physical_units()
    plot_bridge(ics, f, order=True, monotonic=False, title="Halo-0, sim-7", halo_id=0)




    # simulation-3

    # ics = pynbody.load("/Users/lls/Documents/mlhalos_files/reseed50/IC.gadget3")
    # f = pynbody.load("/Users/lls/Documents/mlhalos_files/reseed50/snapshot_099")
    # ics.physical_units()
    # f.physical_units()
    #
    # plot_bridge(ics, f, order=False, title="Bridge", halo_id=200)
    # plot_bridge(ics, f, order=True, monotonic=True, title="Order Bridge", halo_id=200)
    # plot_bridge(ics, f, order=True, monotonic=False, title="Order Bridge, monotonic False", halo_id=200)



# Order Bridge

b = pynbody.bridge.Bridge(ic_params.initial_conditions, ic_params.final_snapshot)
h100 = ic_params.halo[100]
h100_ics = b(h100)

pynbody.analysis.halo.center(h100_ics, vel=False, wrap=True)
plt.figure()
pynbody.plot.sph.image(h100_ics, width="200 kpc", resolution=500, av_z=True, log=True, title="Order Bridge")
plt.subplots_adjust()

# Order Bridge, monotonic=False

b = pynbody.bridge.Bridge(ic_params.initial_conditions, ic_params.final_snapshot)
h100 = ic_params.halo[100]
h100_ics = b(h100)

pynbody.analysis.halo.center(h100_ics, vel=False, wrap=True)
plt.figure()
pynbody.plot.sph.image(h100_ics, width="200 kpc", resolution=500, av_z=True, log=True, title="Order Bridge")
plt.subplots_adjust()

# Order Bridge

b = pynbody.bridge.Bridge(ic_params.initial_conditions, ic_params.final_snapshot)
h100 = ic_params.halo[100]
h100_ics = b(h100)

pynbody.analysis.halo.center(h100_ics, vel=False, wrap=True)
plt.figure()
pynbody.plot.sph.image(h100_ics, width="200 kpc", resolution=500, av_z=True, log=True, title="Order Bridge")
plt.subplots_adjust()



