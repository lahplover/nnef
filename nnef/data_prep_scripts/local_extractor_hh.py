import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import h5py
import os


def check_hh_pdb():
    # make sure the hhsuite seq and the seq from ATOM records match.
    # make sure the group num is the correct index of hhsuite seq.
    pdb_list = pd.read_csv('hhsuite_beads/hhsuite_pdb_beads_list.txt')['pdb'].values

    amino_acids = pd.read_csv('amino_acids.csv')
    vocab = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    match_pdb_list = []
    bad_pdb_list = []
    for pdb in tqdm(pdb_list):
        df_pdb = pd.read_csv(f'hhsuite_beads/hhsuite/{pdb}_bead.csv')
        df_hh = pd.read_csv(f'~/bio/hhsuite/chimeric/{pdb}.chemeric')
        seq_hh = df_hh['seq'].values[0]
        group_num = df_pdb['group_num'].values

        # in some cases, the chains ids do not match.
        if len(seq_hh) <= group_num.max():
            bad_pdb_list.append(pdb)
            print(len(seq_hh), group_num.max())
            # break
            continue
        # use group num as index to extract residues from hh seq
        seq_hh_pdb = ''.join(np.array(list(seq_hh))[group_num])

        seq_pdb = df_pdb['group_name'].apply(lambda x: vocab[x])
        seq_pdb = ''.join(seq_pdb.values)

        if seq_pdb == seq_hh_pdb:
            match_pdb_list.append(pdb)
            # df_pdb.to_csv(f'hhsuite_beads/hhsuite/{pdb}_bead.csv', index=False)
        else:
            bad_pdb_list.append(pdb)
            # print(seq_pdb)
            # print(seq_hh_pdb)
            # break
    df_match = pd.DataFrame({'pdb': match_pdb_list})
    df_match.to_csv('hhsuite_beads/hhsuite_pdb_beads_list_match.txt', index=False)


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


def extract_one_topk(pdb_id, df_beads, local_rot_dir, k=10, mode='CA'):
    if df_beads.shape[0] < 20:
        return

    group_num = df_beads['group_num'].values
    group_name = df_beads['group_name'].values

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

        # get central segment
        ind = (cen_i | pre_i | next_i | pre2_i | next2_i)
        gnum_seg = group_num[ind]
        gname_seg = group_name[ind]

        if len(gnum_seg) != 5:
            continue

        coords = group_coords - group_coords[cen_i]  # center
        c1 = coords[pre_i]
        c2 = coords[next_i]
        if np.sum(c1**2) == 0:
            break
        if np.sum(c2**2) == 0:
            break
        rotate_mat = _rotation_matrix(c1, c2)

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

        df.to_csv(f'{local_rot_dir}/{pdb_id}_{mode}.csv', index=False, float_format='%.4f')

        # count_res = np.vstack(count_res)
        # df_count = pd.DataFrame({'count_6a': count_res[:, 0],
        #                          'count_8a': count_res[:, 1],
        #                          'count_10a': count_res[:, 2],
        #                          'count_12a': count_res[:, 3]})
        # df_count.to_csv(f'{local_rot_dir}/{pdb_id}_count_res.csv', index=False)


def extract_local_structure():
    mode = 'CB'

    data_dir = 'hhsuite_local_rot'
    pdb_list = pd.read_csv('hhsuite_beads/hhsuite_pdb_beads_list_match.txt')['pdb'].values
    np.random.shuffle(pdb_list)

    def extract_batch(batch):
        for i in tqdm(batch):
            pdb_id = pdb_list[i]
            if os.path.exists(f'hhsuite_local_rot/{pdb_id}_{mode}.csv'):
                continue
            df_beads = pd.read_csv(f'hhsuite_beads/hhsuite/{pdb_id}_bead.csv')
            if type(df_beads['group_num'].values[0]) is str:
                print('group_num is string')
                continue
            extract_one_topk(pdb_list[i], df_beads, data_dir, mode=mode)

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
    mode = 'CB'

    local_rot_dir = f'hhsuite_local_rot/'
    # load tht hhsuite pdb list
    # pdb_list = pd.read_csv('hhsuite_CB_local_rot.txt')['pdb']
    pdb_list = pd.read_csv('hhsuite_CB_pdb_list_cullpdb.csv')['pdb']

    # pdb_list = ['4JZX_B']
    used_pdb_list = []
    with h5py.File(f'hhsuite_{mode}.h5', 'w') as f:
        for pdb_id in tqdm(pdb_list):
            df = pd.read_csv(f'{local_rot_dir}/{pdb_id}_{mode}.csv')

            # save data to hdf5
            amino_acids = pd.read_csv('amino_acids.csv')
            vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

            # segment = df['segment'].values
            k = 15
            seq = df['group_name'].apply(lambda x: vocab[x])
            seq = seq.values.reshape((-1, k))
            group_num = df['group_num'].values.reshape((-1, k))
            coords = df[['x', 'y', 'z']].values.reshape((-1, k, 3))

            seg = df['seg'].values
            seg = seg.reshape(-1, k)
            start_id = np.zeros_like(seg)
            idx = (seg[:, 1:] - seg[:, :-1] == 0)
            start_id[:, 1:][idx] = 1
            res_counts = df[['count8a', 'count10a', 'count12a']].values.reshape(-1, k, 3)[:, 0, :]

            distance = df['distance'].values
            dist = distance.reshape(-1, k)
            dist_max = dist.max(axis=-1)

            # clean using residues distances
            # idx = (dist[:, 1] < 4) & (dist[:, 2] < 4) & (dist[:, 3] < 8) & (dist[:, 4] < 8) & (dist_max < 20)
            idx = (dist_max < 20)

            seq = seq[idx]
            group_num = group_num[idx]
            coords = coords[idx]
            start_id = start_id[idx]
            res_counts = res_counts[idx]

            pdb_grp = f.create_group(pdb_id)
            dset = pdb_grp.create_dataset("seq", shape=seq.shape, data=seq, dtype='i1')
            dset = pdb_grp.create_dataset("group_num", shape=group_num.shape, data=group_num, dtype='i')
            dset = pdb_grp.create_dataset("start_id", shape=start_id.shape, data=start_id, dtype='i1')
            dset = pdb_grp.create_dataset("res_counts", shape=res_counts.shape, data=res_counts, dtype='i2')
            dset = pdb_grp.create_dataset("coords", shape=coords.shape, data=coords, dtype='f4')

            used_pdb_list.append(pdb_id)

    df_pdb = pd.DataFrame({'pdb': used_pdb_list})
    df_pdb.to_csv(f'hhsuite_{mode}_pdb_list.csv', index=False)


def get_weights():
    mode = 'CB'
    df = pd.read_csv(f'hhsuite_{mode}_pdb_list.csv')
    pdb_list = df['pdb'].values
    data_seq = h5py.File('hhsuite_pdb_seq.h5', 'r')
    data_ls = h5py.File(f'../erf/data/hhsuite_{mode}.h5', 'r')
    seq_num = np.zeros(pdb_list.shape[0], dtype=np.int)
    res_num = np.zeros(pdb_list.shape[0], dtype=np.int)

    for i, pdb in tqdm(enumerate(pdb_list)):
        seq_chim_n = data_seq[pdb][()]
        seq_num[i] = seq_chim_n.shape[0]
        ls_group_num = data_ls[pdb]['group_num'][()]
        res_num[i] = ls_group_num.shape[0]

    weight = np.sqrt(seq_num) * res_num / 100.0
    df['weight'] = weight
    df.to_csv(f'hhsuite_{mode}_pdb_list.csv', index=False)


def train_val_partition():
    mode = 'CB'
    df = pd.read_csv(f'hhsuite_{mode}_pdb_list.csv')
    df2 = df.sample(frac=1.0)
    df_train = df2[:-5000]
    df_val = df2[-5000:]
    df_train.to_csv(f'hhsuite_{mode}_pdb_list_train.csv', index=False)
    df_val.to_csv(f'hhsuite_{mode}_pdb_list_val.csv', index=False)


def train_val_partition2():
    mode = 'CB'
    df = pd.read_csv(f'hhsuite_CB_pdb_list_cullpdb.csv')

    df2 = df.sample(frac=1.0)
    df_train = df2[:-2000]
    df_val = df2[-2000:]
    df_train.to_csv(f'hhsuite_{mode}_cullpdb_train.csv', index=False)
    df_val.to_csv(f'hhsuite_{mode}_cullpdb_val.csv', index=False)


if __name__ == '__main__':
    extract_local_structure()


