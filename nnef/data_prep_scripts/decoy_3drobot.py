import pandas as pd
from Bio.PDB import Selection, PDBParser, Superimposer
# from Bio.PDB.vectors import rotmat, Vector
import numpy as np
import h5py
from tqdm import tqdm
import os
import multiprocessing as mp


"""
PDB file --> beads center DataFrame --> local structure --> rotated local structure 
Functions in this version can handle multiple chains PDB file. 
"""


def extract_beads(pdb_id, decoy, decoy_set='3DRobot_set'):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    p = PDBParser()
    structure = p.get_structure('X', f'data/decoys/{decoy_set}/{pdb_id}/{decoy}.pdb')
    residue_list = Selection.unfold_entities(structure, 'R')

    ca_center_list = []
    cb_center_list = []
    res_name_list = []
    res_num_list = []
    chain_list = []

    for res in residue_list:
        if res.get_resname() not in vocab_aa:
            # raise ValueError('protein has non natural amino acids')
            continue
        chain_list.append(res.parent.id)
        res_name_list.append(vocab_dict[res.get_resname()])
        res_num_list.append(res.id[1])
        ca_center_list.append(res['CA'].get_coord())
        if res.get_resname() != 'GLY':
            cb_center_list.append(res['CB'].get_coord())
        else:
            cb_center_list.append(res['CA'].get_coord())

    ca_center = np.vstack(ca_center_list)
    cb_center = np.vstack(cb_center_list)

    df = pd.DataFrame({'chain_id': chain_list,
                       'group_num': res_num_list,
                       'group_name': res_name_list,
                       'x': ca_center[:, 0],
                       'y': ca_center[:, 1],
                       'z': ca_center[:, 2],
                       'xcb': cb_center[:, 0],
                       'ycb': cb_center[:, 1],
                       'zcb': cb_center[:, 2]})

    df.to_csv(f'data/decoys/{decoy_set}/{pdb_id}/{decoy}_bead.csv', index=False)


def _rotation_matrix(c1, c2):
    z = np.cross(c1, c2)
    x = c1
    y = np.cross(z, x)
    x = x / np.sqrt(np.sum(x ** 2))
    y = y / np.sqrt(np.sum(y ** 2))
    z = z / np.sqrt(np.sum(z ** 2))
    R = np.vstack([x, y, z])
    # Rinv = np.linalg.inv(R.T)
    return R


def extract_one_topk(pdb_id, decoy, k=10,
                     decoy_set='3DRobot_set', profile_set='pdb_profile_validation',
                     reorder=True):
    # df_profile = pd.read_csv(f'data/decoys/3DRobot_set/pdb_profile_training_100/{pdb_id}_profile.csv')
    df_profile = pd.read_csv(f'data/decoys/{decoy_set}/{profile_set}/{pdb_id}_profile.csv')
    df_coords = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/{decoy}_bead.csv')

    # group_num and evo_profile should be un_masked. they trace the original sequence (peptide-bond connection)
    group_num = np.arange(df_profile.shape[0])
    evo_profile = df_profile[[f'aa{i}' for i in range(20)]].values

    if len(group_num) < 20:
        return 0

    idx = df_profile['mask'] == 1
    if (decoy_set == 'casp11') & (decoy != f'{pdb_id}.native'):
        df_coords = df_coords[idx]

    seq = ''.join(df_profile[idx]['seq'])
    seq2 = ''.join(df_coords['group_name'])

    if (df_profile[idx].shape[0] != df_coords.shape[0]) | (seq != seq2):
        print('PDB and profile shape do not match')
        return 0

    group_num = group_num[idx]
    group_name = df_profile['seq'].values[idx]
    group_coords = df_coords[['x', 'y', 'z']].values  # coords may have missing residues.

    df_list = []
    for i, gc in enumerate(group_num):
        if (gc-1 not in group_num) | (gc+1 not in group_num) | (gc-2 not in group_num) | (gc+2 not in group_num):
            continue
        # coords of the previous 2 and next 2 groups in local peptide segment
        cen_i = (group_num == gc)
        pre_i = (group_num == gc-1)
        next_i = (group_num == gc+1)
        pre2_i = (group_num == gc-2)
        next2_i = (group_num == gc+2)

        coords = group_coords - group_coords[cen_i]  # center
        c1 = coords[pre_i]
        c2 = coords[next_i]
        if np.sum(c1**2) == 0:
            break
        if np.sum(c2**2) == 0:
            break
        rotate_mat = _rotation_matrix(c1, c2)

        # get central segment
        ind = (cen_i | pre_i | next_i | pre2_i | next2_i)
        gnum_seg = group_num[ind]
        gname_seg = group_name[ind]
        coords_seg = coords[ind]
        coords_seg = np.squeeze(np.matmul(rotate_mat[None, :, :], coords_seg[:, :, None]))

        # get nearest k residues from other residues
        gnum_others = group_num[~ind]
        gname_others = group_name[~ind]
        coords_others = coords[~ind]

        dist_i = np.sqrt((coords_others**2).sum(axis=1))
        dist_i_arg = np.argsort(dist_i)
        topk_arg = dist_i_arg[:k]

        count_8a = dist_i[dist_i < 8].shape[0]
        count_10a = dist_i[dist_i < 10].shape[0]
        count_12a = dist_i[dist_i < 12].shape[0]

        gnum_topk = gnum_others[topk_arg]
        gname_topk = gname_others[topk_arg]
        coords_topk = coords_others[topk_arg]

        coords_topk = np.squeeze(np.matmul(rotate_mat[None, :, :], coords_topk[:, :, None]))

        # concat central segment and top_k
        gnum = np.append(gnum_seg, gnum_topk)
        gname = np.append(gname_seg, gname_topk)
        coords = np.vstack((coords_seg, coords_topk))

        distance = np.sqrt(np.sum(coords**2, axis=1))

        segment_info = np.ones(gnum.shape[0], dtype=int) * 5
        segment_info[gnum == gc] = 0
        segment_info[gnum == gc-1] = 1
        segment_info[gnum == gc+1] = 2
        segment_info[gnum == gc-2] = 3
        segment_info[gnum == gc+2] = 4

        df_g = pd.DataFrame({'center_num': gc,
                             'group_num': gnum,
                             'group_name': gname,
                             'x': coords[:, 0],
                             'y': coords[:, 1],
                             'z': coords[:, 2],
                             'distance': distance,
                             'segment': segment_info,
                             'count8a': count_8a,
                             'count10a': count_10a,
                             'count12a': count_12a})
        df_g = df_g.sort_values(by=['segment', 'distance'])

        if reorder:
            df_g = df_g.sort_values(by=['segment', 'group_num'])
            gnum = df_g['group_num'].values
            distance = df_g['distance'].values
            # set segment id
            seg = np.ones(15, dtype=np.int)
            seg[5] = 2
            for i in range(6, 15):
                if gnum[i] == gnum[i - 1] + 1:
                    seg[i] = seg[i - 1]
                else:
                    seg[i] = seg[i - 1] + 1
            # calculate mean distance of segment
            seg_dist = np.zeros(15)
            for i in range(5, 15):
                seg_dist[i] = distance[seg == seg[i]].mean()

            df_g['seg'] = seg
            df_g['seg_dist'] = seg_dist
            df_g = df_g.sort_values(by=['segment', 'seg_dist', 'group_num'])

        df_list.append(df_g)

    if len(df_list) > 0:
        df = pd.concat(df_list, ignore_index=True)

        group_profile = evo_profile[df['group_num'].values]
        for i in range(20):
            df[f'aa{i}'] = group_profile[:, i]

        # df.to_csv(f'data/decoys/3DRobot_set/{pdb_id}/{decoy}_local_rot.csv', index=False, float_format='%.3f')

        amino_acids = pd.read_csv('data/amino_acids.csv')
        vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}

        k = 15
        seq = df['group_name'].apply(lambda x: vocab[x])
        seq = seq.values.reshape((-1, k))
        coords = df[['x', 'y', 'z']].values.reshape((-1, k, 3))
        profile = df[[f'aa{i}' for i in range(20)]].values.reshape((-1, k, 20))
        res_counts = df[['count8a', 'count10a', 'count12a']].values.reshape(-1, 15, 3)[:, 0, :]

        if reorder:
            group_num = df['group_num'].values.reshape((-1, k))
            seg = df['seg'].values
            seg = seg.reshape(-1, k)
            start_id = np.zeros_like(seg)
            idx = (seg[:, 1:] - seg[:, :-1] == 0)
            start_id[:, 1:][idx] = 1
            decoy_file_name = f'{decoy}_local_rot_CA.h5'
        else:
            decoy_file_name = f'{decoy}_local_rot.h5'

        with h5py.File(f'data/decoys/{decoy_set}/{pdb_id}/{decoy_file_name}', 'w') as f:
            dset = f.create_dataset("seq", shape=seq.shape, data=seq, dtype='i')
            dset = f.create_dataset("coords", shape=coords.shape, data=coords, dtype='f4')
            dset = f.create_dataset("profile", shape=profile.shape, data=profile, dtype='f4')
            dset = f.create_dataset("res_counts", shape=res_counts.shape, data=res_counts, dtype='i')
            if reorder:
                dset = f.create_dataset("group_num", shape=group_num.shape, data=group_num, dtype='i')
                dset = f.create_dataset("start_id", shape=start_id.shape, data=start_id, dtype='i')
    else:
        print(group_name)

    return 1


def check_3drobot_bead_profile():
    pdb_list = pd.read_csv(f'data/decoys/3DRobot_set/pdb_profile_diff.txt')['pdb'].values
    decoy_set = '3DRobot_set'
    profile_set = 'pdb_profile_training_100'

    # pdb_list = pd.read_csv(f'data/decoys/casp11/pdb_no_need_copy_native.txt')['pdb'].values
    # decoy_set = 'casp11'
    # profile_set = 'pdb_profile'

    pdb_list_good = []
    for pdb_id in tqdm(pdb_list):
        df_profile = pd.read_csv(f'data/decoys/{decoy_set}/{profile_set}/{pdb_id}_profile.csv')
        df = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/list.txt', sep='\s+')
        decoy_list = df['NAME'].values
        # df_coords = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/{pdb_id}.native_bead.csv')
        bad_count = 0
        for decoy in decoy_list:
            df_coords = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/{decoy[:-4]}_bead.csv')
            if df_profile.shape[0] != df_coords.shape[0]:
                bad_count += 1
        if bad_count == 0:
            pdb_list_good.append(pdb_id)
    pd.DataFrame({'pdb': pdb_list_good}).to_csv(f'data/decoys/{decoy_set}/pdb_profile_diff_match.txt', index=False)


def extract_decoy_set_3drobot(pdb_list):
    decoy_set = '3DRobot_set'
    profile_set = 'pdb_profile_training_100'

    # pdb_list = pd.read_csv(f'data/decoys/3DRobot_set/pdb_profile_diff.txt')['pdb'].values
    # pdb_list = ['1HF2A']

    # for pdb_id in pdb_list:
    #     df_profile = pd.read_csv(f'data/decoys/{decoy_set}/{profile_set}/{pdb_id}_profile.csv')
    #     df_coords = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/native_bead.csv')
    #     seq = ''.join(df_profile['seq'])
    #     mask = df_profile['mask']
    #     seq2 = ''.join(df_coords['group_name'])
    #
    #     if (df_profile.shape[0] != df_coords.shape[0]) | (seq != seq2):
    #         # print(f'{pdb_id} PDB and profile shape do not match')
    #         idx = (mask == 1)
    #         seq_mask = ''.join(df_profile[idx]['seq'])
    #         if (df_profile[idx].shape[0] != df_coords.shape[0]) | (seq_mask != seq2):
    #             print(f'{pdb_id} PDB and masked profile shape do not match')

    for pdb_id in pdb_list:
        # ignore 1TJXA 2J1VA, some residues don't have CA.
        # decoy_list = ['native', 'decoy46_42']
        decoy_list = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/list.txt', sep='\s+')['NAME'].values
        for decoy in tqdm(decoy_list):
            decoy = decoy[:-4]
            extract_beads(pdb_id, decoy, decoy_set=decoy_set)
            # result = extract_one_topk(pdb_id, decoy, decoy_set=decoy_set, profile_set=profile_set)
            # if result == 0:
            #     print(pdb_id, decoy)

    # local_rot_list = []
    # for pdb_id in pdb_list:
    #     if os.path.exists(f'data/decoys/{decoy_set}/{pdb_id}/native_local_rot.h5'):
    #         local_rot_list.append(pdb_id)
    # pd.DataFrame({'pdb': local_rot_list}).to_csv(f'data/decoys/{decoy_set}/pdb_local_rot.txt', index=False)

    # train100 = pd.read_csv('pdb_profile_100.txt')['pdb'].values
    # train30 = pd.read_csv('pdb_profile_30.txt')['pdb'].values
    # set_100 = set(train100)
    # set_30 = set(train30)
    # set_diff = set_100 - set_30
    # pd.DataFrame({'pdb': list(set_diff)}).to_csv('pdb_profile_diff.txt', index=False)


def extract_decoy_set_3drobot_all():
    pdb_list = pd.read_csv(f'data/decoys/3DRobot_set/pdb_profile_diff.txt')['pdb'].values
    count = len(pdb_list)
    num_cores = 40
    batch_size = count // num_cores + 1
    idx_list = np.arange(count)

    batch_list = []
    for i in range(0, count, batch_size):
        batch = idx_list[i:i+batch_size]
        batch_list.append(pdb_list[batch])

    # setup the multi-processes
    with mp.Pool(processes=num_cores) as pool:
        pool.map(extract_decoy_set_3drobot, batch_list)


def check_h5_data_3drobot():
    decoy_set = '3DRobot_set'
    pdb_list = pd.read_csv(f'data/decoys/{decoy_set}/pdb_profile_diff.txt')['pdb'].values

    for pdb_id in tqdm(pdb_list):
        if pdb_id == '1WDDS':
            continue
        df = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/list.txt', sep='\s+')
        decoy_list = df['NAME'].values
        native_data_path = f'data/decoys/{decoy_set}/{pdb_id}/native_local_rot.h5'

        native_data = h5py.File(native_data_path, 'r')
        native_shape = native_data['seq'].shape[0]
        for decoy in decoy_list:
            data_path = f'data/decoys/{decoy_set}/{pdb_id}/{decoy[:-4]}_local_rot.h5'
            if not os.path.exists(data_path):
                print(f'{pdb_id} {decoy} no h5')
                continue
            decoy_data = h5py.File(data_path, 'r')
            decoy_data_shape = decoy_data['seq'].shape[0]
            if native_shape != decoy_data_shape:
                print(f'{pdb_id} {decoy} different shape')


def extract_beads_batch_new(pdb_list):
    # use all 200 proteins in the 3DRobot set
    decoy_set = '3DRobot_set'
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    # pdb_list = pd.read_csv(f'data/decoys/3DRobot_set/pdb_all.csv')['pdb'].values
    # pdb_list = pd.read_csv(f'data/decoys/3DRobot_set/pdb_no_missing_residue.csv')['pdb'].values
    # pdb_no_missing_list = []

    for pdb_id in tqdm(pdb_list):
        df_pdb = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/list.txt', sep='\s+')
        decoy_list = df_pdb['NAME'].values
        # decoy_list = ['native.pdb']
        idx = np.zeros(decoy_list.shape[0])
        for i, decoy in enumerate(decoy_list):
            decoy = decoy[:-4]
            try:
                p = PDBParser()
                structure = p.get_structure('X', f'data/decoys/{decoy_set}/{pdb_id}/{decoy}.pdb')
                residue_list = Selection.unfold_entities(structure, 'R')
            except:
                continue

            ca_center_list = []
            cb_center_list = []
            res_name_list = []
            res_num_list = []
            chain_list = []

            for res in residue_list:
                if res.get_resname() not in vocab_aa:
                    # raise ValueError('protein has non natural amino acids')
                    continue
                try:
                    res['CA'].get_coord()
                    if res.get_resname() != 'GLY':
                        res['CB'].get_coord()
                except KeyError:
                    print(pdb_id, res.id[1])
                    continue

                chain_list.append(res.parent.id)
                res_name_list.append(vocab_dict[res.get_resname()])
                res_num_list.append(res.id[1])
                ca_center_list.append(res['CA'].get_coord())
                if res.get_resname() != 'GLY':
                    cb_center_list.append(res['CB'].get_coord())
                else:
                    cb_center_list.append(res['CA'].get_coord())

            ca_center = np.vstack(ca_center_list)
            cb_center = np.vstack(cb_center_list)

            df = pd.DataFrame({'chain_id': chain_list,
                               'group_num': res_num_list,
                               'group_name': res_name_list,
                               'x': ca_center[:, 0],
                               'y': ca_center[:, 1],
                               'z': ca_center[:, 2],
                               'xcb': cb_center[:, 0],
                               'ycb': cb_center[:, 1],
                               'zcb': cb_center[:, 2]})

            df.to_csv(f'data/decoys/{decoy_set}/{pdb_id}/{decoy}_bead.csv', index=False)

            g_num = df['group_num'].values
            num_res_pdb = g_num[-1] - g_num[0] + 1
            if num_res_pdb == g_num.shape[0]:
                idx[i] = 1
                # if decoy == 'native':
                #     pdb_no_missing_list.append(pdb_id)
            else:
                print(pdb_id, 'has missing_residue')

        df_pdb[idx == 1].to_csv(f'data/decoys/{decoy_set}/{pdb_id}/list.csv', index=False)

    # df = pd.DataFrame({'pdb': pdb_no_missing_list})
    # df.to_csv('data/decoys/3DRobot_set/pdb_no_missing_residue.csv', index=False)


def extract_beads_all_new():
    pdb_list = pd.read_csv(f'data/decoys/3DRobot_set/pdb_no_missing_residue.csv')['pdb'].values
    count = len(pdb_list)
    num_cores = 40
    batch_size = count // num_cores + 1
    idx_list = np.arange(count)

    batch_list = []
    for i in range(0, count, batch_size):
        batch = idx_list[i:i+batch_size]
        batch_list.append(pdb_list[batch])

    # setup the multi-processes
    with mp.Pool(processes=num_cores) as pool:
        pool.map(extract_beads_batch_new, batch_list)


def check_NMR():
    from Bio.PDB import PDBList, PDBParser
    import pandas as pd
    import os

    pdb_list = pd.read_csv('pdb_no_missing_residue.csv')['pdb'].values
    method_list = []
    for pdb_id in pdb_list:
        if os.path.exists(f'pdbs/pdb{pdb_id[:4].lower()}.ent'):
            continue
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id[:4], pdir='./pdbs', file_format='pdb')
        if not os.path.exists(f'pdbs/pdb{pdb_id[:4].lower()}.ent'):
            method_list.append('no pdb')
            continue
        p = PDBParser()
        structure = p.get_structure('X', f'./pdbs/pdb{pdb_id[:4].lower()}.ent')
        method_list.append(structure.header['structure_method'])

    df = pd.DataFrame({'pdb': pdb_list, 'method': method_list})
    df.to_csv('pdbs_methods.csv', index=False)


