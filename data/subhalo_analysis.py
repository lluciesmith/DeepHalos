import numpy as np


def make_grp_subhalo_catalogue(snapshot, subhalo_catalogue, halo_ids=None):
    halo_catalogue = snapshot.halos()
    if halo_ids is None:
        halo_ids = np.arange(len(halo_catalogue))

    ids = snapshot['iord']
    if not np.allclose(ids, np.sort(ids)):
        print("Create indices")
        indices = np.argsort(snapshot['iord'])
        print("Done")
    else:
        indices = ids

    subhalo_ids = np.concatenate([halo_catalogue[i].properties['children'][1:] for i in halo_ids])
    print("Done")
    print(len(subhalo_ids))

    # This array assumes the IORD array is ordered

    subh_grp_ = np.ones(len(ids), ) * -1
    subh_grp = assing_subhalo_id(subh_grp_, subhalo_ids, subhalo_catalogue)
    print("Done")

    # This step ensures the order of the array matches that of IORD
    snapshot['subh_grp'] = np.empty(ids.shape)
    snapshot['subh_grp'][indices] = subh_grp.astype("int")
    print("Done")
    return snapshot


def assing_subhalo_id(output_array, subhalo_ids, subhalo_catalogue):
    for subh_id in subhalo_ids:
        output_array[subhalo_catalogue[subh_id]['iord']] = subh_id
    return output_array