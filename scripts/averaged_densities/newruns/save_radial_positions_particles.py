from pickle import load, dump
import numpy as np
from multiprocessing import Pool
import pynbody


def load_snapshot(sim_num, path, units="Mpc a h**-1"):
    f = pynbody.load(path + 'standard_reseed' + sim_num + '/output/snapshot_007')
    f.physical_units(distance=units)
    h = f.halos()
    return f, h


def find_radius_particles_centred_in_halo(particle_ids, halo_ID, f=None, h=None, sim_num=None, path=None):
    if f is None and h is None:
        f, h = load_snapshot(sim_num, path)

    # centre the snapshot on the "shrinking sphere" radius
    pynbody.analysis.halo.center(f[h[halo_ID].properties['mostboundID']], vel=False)
    f.wrap()
    pynbody.analysis.halo.center(h[halo_ID], vel=False)
    radii = np.array([float(f[f['iord'] == particle]['r']) for particle in particle_ids])

    return radii/float(h[halo_ID].properties['rmean_200'])


def get_radius_of_particles(particles, particles_grp, f=None, h=None, sim_num=None, path=None):
    if f is None and h is None:
        f, h = load_snapshot(sim_num, path)

    assert all(particles_grp > 0)
    radii_particles = np.zeros((len(particles),))

    for halo_id in np.unique(particles_grp):
        print("Halo " + str(halo_id))
        index = particles_grp == halo_id
        particles_halo_id = particles[index]
        r_ = find_radius_particles_centred_in_halo(particles_halo_id, halo_id, f=f, h=h)
        radii_particles[index] = r_

    return radii_particles



def init_worker(halo_cat):
    global shared_halo_cat
    shared_halo_cat = halo_cat


def get_iord_halo(halo_id):
    return h[halo_id]["iord"]


def halo_iord_with_pool(halo_cat, num_halos):
    with Pool(processes=40, initializer=init_worker, initargs=(halo_cat,)) as pool:
        iords = pool.map(get_iord_halo, np.arange(num_halos))
    return iords


def get_particle_grp(snapshot, halo_catalogue):
    num_halos = len(halo_catalogue)
    halos_iord = halo_iord_with_pool(halo_catalogue, num_halos)

    particles_grp = np.ones((len(snapshot),)) * -1
    for i in range(num_halos):
        particles_grp[halos_iord[i]] = i

    particles_grp = particles_grp.astype('int')
    return dict(zip(np.arange(len(snapshot)), particles_grp))


if __name__ == "__main__":
    path_sims = "/share/hypatia/lls/simulations/"
    test_sim = ["6", "22", "23", "24"]

    saving_path = "/share/hypatia/lls/newdlhalos/training_data/"
    test_set = load(open(saving_path + 'test_set.pkl', 'rb'))
    sim_index = np.array([ID[ID.find('sim-') + len('sim-'): ID.find('-id')] for ID in test_set])
    particle_ID = np.array([ID[ID.find('-id-') + + len('-id-'):] for ID in test_set]).astype('int')
    radii_particle = np.zeros((len(particle_ID),))

    for sim in test_sim:
        print("Sim " + sim)
        idx = np.where(sim_index == sim)[0]

        try:
            rparticles = np.load(saving_path + 'radii_testparticles_sim' + sim + '.npy')
        except:
            f, h = load_snapshot(sim, path_sims)
            print("Loaded snapshot")
            try:
                all_pids_grp = load(open(saving_path + 'all_particleids_simulation' + sim + '_grp.pkl', 'rb'))
                print("Loaded grp for sim " + sim)
            except:
                all_pids_grp = get_particle_grp(f, h)
                print("Got particle group number")
                dump(all_pids_grp, open(saving_path + 'all_particleids_simulation' + sim + '_grp.pkl', 'wb'))

            testids_sim = particle_ID[idx]
            rparticles = get_radius_of_particles(testids_sim, np.array([all_pids_grp[pid] for pid in testids_sim]), f=f, h=h)
            np.save(saving_path + 'radii_testparticles_sim' + sim + '.npy', rparticles)

        radii_particle[idx] = rparticles

    dict_radii = dict(zip((test_set, radii_particle)))
    dump(dict_radii, open(saving_path + 'radii_test_set_particles.pkl', 'wb'))
