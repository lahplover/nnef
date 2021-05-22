import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import h5py
import os


###########################################
def match_profile_coords():
    # After applying profile mask, the masked df_profile should match the df_beads on both coordinates and seq.
    amino_acids = pd.read_csv('amino_acids.csv')
    vocab = {y.upper(): x for x, y in zip(amino_acids.AA, amino_acids.AA3C)}

    # profile_dir = 'training_100_profile'
    profile_dir = 'validation_profile'
    bead_dir = 'proteinnet_beads'
    pdb1 = pd.read_csv(f'{profile_dir}/flist.txt')['fname']
    # pdb2 = pdb1.apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
    pdb2 = pdb1.apply(lambda x: x.split('_')[0][3:] + '_' + x.split('_')[2])

    bad = []
    good = []
    for p1, p2 in tqdm(zip(pdb1, pdb2)):
        p2_path = f'{bead_dir}/{p2}_bead.csv'
        if not os.path.exists(p2_path):
            continue
        df1 = pd.read_csv(f'{profile_dir}/{p1}')
        df2 = pd.read_csv(p2_path)
        mask = df1['mask']
        if df1[mask==1].shape[0] == df2.shape[0]:
            df1m = df1[mask==1].copy()
            ca1 = df1m[['x', 'y', 'z']].values
            ca2 = df2[['xca', 'yca', 'zca']].values
            seq1 = ''.join(df1m['group_name'].values)
            seq2 = ''.join(df2['group_name'].apply(lambda x: vocab[x]).values)
            if np.abs(ca1 - ca2).max()>0.002:
                # print(p1)
                bad.append(p1)
            elif seq1 != seq2:
                # print(p1, 'seq diff')
                bad.append(p1)
            else:
                good.append(p1)
                # df1m['xs'] = df2['xs'].values
                # df1m['ys'] = df2['ys'].values
                # df1m['zs'] = df2['zs'].values
                # df1m.to_csv(f'{data_dir}/{p1}_match.csv', index=False, float_format='%.4f')
    df = pd.DataFrame({'pdb': good})
    # df.to_csv('bead_profile_match.csv', index=False)
    df.to_csv('bead_profile_match_validation.csv', index=False)


###########################################
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


def extract_one_topk(pdb_id, df_beads, df_profile, local_rot_dir, k=10, mode='CA'):
    if df_beads.shape[0] < 20:
        return

    idx = (df_profile['mask'] == 1)
    group_num = df_profile['group_num'].values[idx]
    group_name = df_profile['group_name'].values[idx]

    if mode == 'CA':
        group_coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        group_coords = df_beads[['xcb', 'ycb', 'zcb']].values
    elif mode == 'CAS':
        group_coords = (df_beads[['xca', 'yca', 'zca']].values + df_beads[['xs', 'ys', 'zs']].values) / 2
    else:
        raise ValueError('mode should be CA / CB / CAS.')

    df_list = []
    count_res = []
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
        # topk_arg = (dist_i < 8)
        # count_6a = dist_i[dist_i < 6].shape[0]
        count_8a = dist_i[dist_i < 8].shape[0]
        count_10a = dist_i[dist_i < 10].shape[0]
        count_12a = dist_i[dist_i < 12].shape[0]
        # count_res.append(np.array([count_6a, count_8a, count_10a, count_12a]))

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
        # df_g = df_g.sort_values(by=['segment', 'distance'])

        def re_order_df_g(df_g):
            df_g = df_g.sort_values(by=['segment', 'group_num'])
            group_num = df_g['group_num'].values
            distance = df_g['distance'].values
            # set segment id
            seg = np.ones(15, dtype=np.int)
            seg[5] = 2
            for i in range(6, 15):
                if group_num[i] == group_num[i - 1] + 1:
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
            return df_g

        df_g = re_order_df_g(df_g)
        df_list.append(df_g)

    if len(df_list)>0:
        df = pd.concat(df_list, ignore_index=True)

        idx = df['group_num'].values
        for i in range(20):
            aa_i = f'aa{i}'
            df[aa_i] = df_profile[aa_i].values[idx]

        df.to_csv(f'{local_rot_dir}/{pdb_id}_{mode}.csv', index=False, float_format='%.4f')

        # count_res = np.vstack(count_res)
        # df_count = pd.DataFrame({'count_6a': count_res[:, 0],
        #                          'count_8a': count_res[:, 1],
        #                          'count_10a': count_res[:, 2],
        #                          'count_12a': count_res[:, 3]})
        # df_count.to_csv(f'{local_rot_dir}/{pdb_id}_count_res.csv', index=False)


def extract_local_structure():
    dataset = 'training_30'
    # dataset = 'validation'
    mode = 'CAS'

    if dataset == 'training_30':
        # match training_30_protein_id to bead_profile_match.csv
        data_dir = 'local_rot_training_30_v3'
        pdb_list = pd.read_csv('training_30_protein_id2.csv')['pdb_id'].values
        beads_list = pd.read_csv('bead_profile_match.csv')['pdb'].values
    elif dataset == 'validation':
        data_dir = 'local_rot_validation_v3'
        pdb_list = pd.read_csv('validation_protein_id2.csv')['pdb'].values
        beads_list = pd.read_csv('bead_profile_match_validation.csv')['pdb'].values
    else:
        raise ValueError('dataset not found')
    pdb_list = list(set(pdb_list) & set(beads_list))

    def extract_batch(batch):
        for i in tqdm(batch):
            pdb_id = pdb_list[i]
            if dataset == 'training_30':
                pdb_id_bead = pdb_id.split('_')[0] + '_' + pdb_id.split('_')[2]
                profile_dir = 'training_100_profile'
            elif dataset == 'validation':
                pdb_id_bead = pdb_id.split('_')[0][3:] + '_' + pdb_id.split('_')[2]
                profile_dir = 'validation_profile'
            df_beads = pd.read_csv(f'proteinnet_beads/{pdb_id_bead}_bead.csv')
            df_profile = pd.read_csv(f'{profile_dir}/{pdb_id}_profile.csv')
            extract_one_topk(pdb_list[i], df_beads, df_profile, data_dir, mode=mode)

    count = len(pdb_list)
    num_cores = 40
    batch_size = count // num_cores + 1
    idx_list = np.arange(count)

    batch_list = []
    for i in range(0, count, batch_size):
        batch = idx_list[i:i+batch_size]
        batch_list += [batch]

    # setup the multi-processes
    with mp.Pool(processes=num_cores) as pool:
        pool.map(extract_batch, batch_list)


############################################

def save_local_h5():
    input_file = 'training_30'
    # input_file = 'validation'
    # input_file = 'testing'
    mode = 'CAS'

    local_rot_dir = f'local_rot_{input_file}_v3/'
    # flist = pd.read_csv(f'{local_rot_dir}/flist_{mode}.txt')['fname'].values
    # load tht hhsuite-proteinnet-pdb matched pdb list
    flist = pd.read_csv('hh_ca_pdb_list.txt')['pdb_profile']
    flist = flist.apply(lambda x: x + f'_{mode}.csv')

    df_list = []
    for fname in tqdm(flist):
        df = pd.read_csv(f'{local_rot_dir}/{fname}')
        df['pdb'] = fname[:-7]
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)

    # save data to hdf5
    amino_acids = pd.read_csv('amino_acids.csv')
    vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}

    # segment = df['segment'].values
    k = 15
    seq = df['group_name'].apply(lambda x: vocab[x])
    seq = seq.values.reshape((-1, k))
    group_num = df['group_num'].values.reshape((-1, k))
    coords = df[['x', 'y', 'z']].values.reshape((-1, k, 3))
    profile = df[[f'aa{i}' for i in range(20)]].values.reshape((-1, k, 20))
    pdb = df['pdb'].values.reshape((-1, k))[:, 0]

    seg = df['seg'].values
    seg = seg.reshape(-1, k)
    start_id = np.zeros_like(seg)
    idx = (seg[:, 1:] - seg[:, :-1] == 0)
    start_id[:, 1:][idx] = 1
    res_counts = df[['count8a', 'count10a', 'count12a']].values.reshape(-1, k, 3)[:, 0, :]

    distance = df['distance'].values
    dist = distance.reshape(-1, k)
    dist_max = dist.max(axis=-1)

    print(seq.min(), coords.min(), profile.min(), group_num.max(), res_counts.max())
    print(seq.shape, coords.shape, profile.shape, group_num.shape, res_counts.shape)

    # clean using residues distances
    # idx = (dist[:, 1] < 4) & (dist[:, 2] < 4) & (dist[:, 3] < 8) & (dist[:, 4] < 8) & (dist_max < 20)
    idx = (dist_max < 20)

    seq = seq[idx]
    group_num = group_num[idx]
    coords = coords[idx]
    profile = profile[idx]
    start_id = start_id[idx]
    res_counts = res_counts[idx]
    pdb = pdb[idx]

    print(seq.min(), coords.min(), profile.min(), group_num.max(), res_counts.max())
    print(seq.shape, coords.shape, profile.shape, group_num.shape, res_counts.shape)

    # shuffle
    num = seq.shape[0]
    idx = np.arange(num)
    np.random.shuffle(idx)

    seq = seq[idx]
    group_num = group_num[idx]
    coords = coords[idx]
    profile = profile[idx]
    start_id = start_id[idx]
    res_counts = res_counts[idx]
    pdb = pdb[idx]

    df_pdb = pd.DataFrame({'pdb': pdb})
    df_pdb['pdb'] = df_pdb['pdb'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
    df_pdb.to_csv(f'{input_file}_{mode}_pdb.csv', index=False)

    with h5py.File(f'{input_file}_{mode}_v2.h5', 'w') as f:
        dset = f.create_dataset("seq", shape=seq.shape, data=seq, dtype='i1')
        dset = f.create_dataset("group_num", shape=group_num.shape, data=group_num, dtype='i')
        dset = f.create_dataset("start_id", shape=start_id.shape, data=start_id, dtype='i1')
        dset = f.create_dataset("res_counts", shape=res_counts.shape, data=res_counts, dtype='i2')
        dset = f.create_dataset("coords", shape=coords.shape, data=coords, dtype='f4')
        dset = f.create_dataset("profile", shape=profile.shape, data=profile, dtype='f4')

    # if input_file == 'training_30':
    #     N = 100000
    #     df_pdb[N:].to_csv(f'{input_file}_{mode}_pdb_train.csv', index=False)
    #
    #     with h5py.File(f'{input_file}_{mode}_v2_train.h5', 'w') as f:
    #         dset = f.create_dataset("seq", shape=seq[N:].shape, data=seq[N:], dtype='i1')
    #         dset = f.create_dataset("group_num", shape=group_num[N:].shape, data=group_num[N:], dtype='i')
    #         dset = f.create_dataset("start_id", shape=start_id[N:].shape, data=start_id[N:], dtype='i1')
    #         dset = f.create_dataset("res_counts", shape=res_counts[N:].shape, data=res_counts[N:], dtype='i2')
    #         dset = f.create_dataset("coords", shape=coords[N:].shape, data=coords[N:], dtype='f4')
    #         dset = f.create_dataset("profile", shape=profile[N:].shape, data=profile[N:], dtype='f4')
    #
    #     df_pdb[:N].to_csv(f'{input_file}_{mode}_pdb_val.csv', index=False)
    #     with h5py.File(f'training_30_{mode}_v2_val.h5', 'w') as f:
    #         dset = f.create_dataset("seq", shape=(N, k), data=seq[:N], dtype='i1')
    #         dset = f.create_dataset("group_num", shape=(N, k), data=group_num[:N], dtype='i')
    #         dset = f.create_dataset("start_id", shape=(N, k), data=start_id[:N], dtype='i1')
    #         dset = f.create_dataset("res_counts", shape=(N, 3), data=res_counts[:N], dtype='i2')
    #         dset = f.create_dataset("coords", shape=(N, k, 3), data=coords[:N], dtype='f4')
    #         dset = f.create_dataset("profile", shape=(N, k, 20), data=profile[:N], dtype='f4')


def save_small():
    k = 15
    input_file = 'training_30'
    mode = 'CA'
    df_pdb = pd.read_csv(f'{input_file}_{mode}_pdb_small.csv')
    data = h5py.File(f'{input_file}_{mode}_v2.h5', 'r')
    seq = data['seq'][()]
    group_num = data['group_num'][()]
    start_id = data['start_id'][()]
    res_counts = data['res_counts'][()]
    coords = data['coords'][()]
    profile = data['profile'][()]

    df_pdb[:1000].to_csv(f'{input_file}_{mode}_pdb_small.csv', index=False)
    with h5py.File(f'training_30_small_{mode}.h5', 'w') as f:
        dset = f.create_dataset("seq", shape=(1000, k), data=seq[:1000], dtype='i1')
        dset = f.create_dataset("group_num", shape=(1000, k), data=group_num[:1000], dtype='i')
        dset = f.create_dataset("start_id", shape=(1000, k), data=start_id[:1000], dtype='i1')
        dset = f.create_dataset("res_counts", shape=(1000, 3), data=res_counts[:1000], dtype='i2')
        dset = f.create_dataset("coords", shape=(1000, k, 3), data=coords[:1000], dtype='f4')
        dset = f.create_dataset("profile", shape=(1000, k, 20), data=profile[:1000], dtype='f4')



####################################
def small_protein():
    beads_list = pd.read_csv('bead_profile_match.csv')['pdb'].values
    selected = ['1BPI_1_A', '2F4K_1_A', '2F21_d2f21a1', '2HBA_1_A', '2WXC_1_A', '2JOF_1_A', '1FME_1_A', '2P6J_1_A', '2A3D_1_A']
    for pdb in selected:
        if pdb in beads_list:
            print(pdb)
            pdb_bead = pdb.split('_')[0] + '_' + pdb.split('_')[2]
            os.system(f'cp proteinnet_beads/{pdb_bead}_bead.csv small_protein/')
            os.system(f'cp training_100_profile/{pdb}_profile.csv small_protein/')


def full_chain():
    beads_list = pd.read_csv('bead_profile_match.csv')['pdb'].values
    df = pd.read_csv('training_100-validation_protein_id2.csv')
    idx = (df['pdb_res_count'] == df['seq_len'])
    df = df[idx]
    pdb_list = df['pdb_id'].values
    match = np.zeros(pdb_list.shape[0], dtype=int)
    for i, p in tqdm(enumerate(pdb_list)):
        if p in beads_list:
            match[i] = 1
    idx = (match == 1)
    df2 = df[idx]

    df2.to_csv('protein_no_missing_residue_bead_profile_match.csv', index=False)
    sample = df2.sample(n=100)
    sample.to_csv('sample.csv', index=False)
    for pdb in sample['pdb_id']:
        print(pdb)
        pdb_bead = pdb.split('_')[0] + '_' + pdb.split('_')[2]
        os.system(f'cp proteinnet_beads/{pdb_bead}_bead.csv protein_sample/')
        os.system(f'cp training_100_profile/{pdb}_profile.csv protein_sample/')

    # idx = (df2['seq_len']<60) & (df2['profile_info']<0.6)



def reorder_local_struct():
    input_file = 'training_30'
    # input_file = 'validation'
    # input_file = 'testing'
    local_rot_dir = f'local_rot_{input_file}/'
    local_rot_dir_v2 = f'local_rot_{input_file}_v2/'
    if not os.path.exists(f'{local_rot_dir_v2}'):
        os.system(f'mkdir -p {local_rot_dir_v2}')

    def extract_one(fname):
        df_list = []
        df = pd.read_csv(f'{local_rot_dir}/{fname}')
        if (input_file == 'validation') | (input_file == 'testing'):
            df['pdb'] = fname[:-4]

        for gc in df['center_num'].unique():
            df_g = df[df['center_num'] == gc]
            if df_g.shape[0] != 15:
                print(fname)
                continue
            df_g = df_g.sort_values(by=['segment', 'group_num'])

            group_num = df_g['group_num'].values
            distance = df_g['distance'].values
            # set segment id
            seg = np.ones(15, dtype=np.int)
            seg[5] = 2
            for i in range(6, 15):
                if group_num[i] == group_num[i - 1] + 1:
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
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(f'{local_rot_dir_v2}/{fname}', index=False, float_format='%.4f')

    def extract_batch(batch):
        for fname in tqdm(batch):
            extract_one(fname)

    flist = pd.read_csv(f'{local_rot_dir}/flist.txt')['fname'].values
    count = len(flist)
    np.random.shuffle(flist)

    num_cores = 40
    batch_size = count // num_cores + 1

    batch_list = []
    for i in range(0, count, batch_size):
        batch = flist[i:i+batch_size]
        batch_list += [batch]

    with mp.Pool(processes=num_cores) as pool:
        pool.map(extract_batch, batch_list)


def test_reorder_one():
    local_rot_dir = f'local_rot_training_30/'
    fname = '12AS_1_A.csv'
    df_list = []
    df = pd.read_csv(f'{local_rot_dir}/{fname}')

    for gc in df['center_num'].unique():
        df_g = df[df['center_num'] == gc]
        df_g = df_g.sort_values(by=['segment', 'group_num'])

        group_num = df_g['group_num'].values
        distance = df_g['distance'].values
        # set segment id
        seg = np.ones(15, dtype=np.int)
        seg[5] = 2
        for i in range(6, 15):
            if group_num[i] == group_num[i - 1] + 1:
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
    df = pd.concat(df_list, ignore_index=True)
    # df.to_csv(f'{local_rot_dir_v2}/{fname}', index=False, float_format='%.3f')






