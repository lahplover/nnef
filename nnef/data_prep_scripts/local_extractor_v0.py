import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os


def save_local_csv():
    # collect rotated local structure into one pandas DataFrame file
    for flag in ['train', 'validation', 'test']:
        df_list = []
        # count = 0
        num = 0
        flist = pd.read_csv(f'cath_Ingraham_{flag}.txt')['fname'].values
        for fname in tqdm(flist):
            if os.path.exists(f'local_rot/{fname}_rot.csv'):
                # count += 1
                df = pd.read_csv(f'local_rot/{fname}_rot.csv')
                df['pdbchain'] = fname
                center_n = df['center_num'].nunique()
                # assert(df.shape[0] == center_n * 10)
                if df.shape[0] != center_n * 10:
                    print(fname)
                    continue
                idx = num + np.arange(center_n)
                num_dict = {x: y for x, y in zip(df['center_num'].unique(), idx)}
                df['num'] = df['center_num'].apply(lambda x: num_dict[x])
                df_list.append(df)
                num += center_n
        # print(count)  # train: 14373, val: 433, test: 871
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(f'local_rot_{flag}.csv', index=False)

    # # collect rotated local structure into one json file
    # for flag in ['train', 'validation', 'test']:
    #     dataset = []
    #     # count = 0
    #     flist = pd.read_csv(f'data/cath_Ingraham_{flag}.txt')['fname'].values
    #     for fname in tqdm(flist):
    #         if os.path.exists(f'data/local_rot/{fname}_rot.csv'):
    #             # count += 1
    #             df = pd.read_csv(f'data/local_rot/{fname}_rot.csv')
    #             center_num = df['center_num'].unique()
    #             for g in center_num:
    #                 df_g = df[df['center_num'] == g]
    #                 local_dict = {'pdbchain': fname,
    #                               'center_num': df_g['center_num'][0],
    #                               'group_num': df_g['group_num'].values,
    #                               'x': df_g['x'].values,
    #                               'y': df_g['y'].values,
    #                               'z': df_g['z'].values}
    #                 dataset.append(local_dict)
    #     outfile = f'dataset_{flag}.json'
    #     with open(outfile, 'w') as f:
    #         for entry in dataset:
    #             f.write(json.dumps(entry) + '\n')


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


def rotate_one(fname):
    df_list = []
    df = pd.read_csv(f'local/{fname}.csv')
    center_num = df['center_num'].unique()
    for g in center_num:
        df_g = df[df['center_num'] == g]
        # # use less than 10 nearest residues
        # df_g = df_g.sort_values(by='distance', ascending=True)
        # df_g.index = np.arange(df_g.shape[0])
        # df_g = df_g[:10]

        df_g = df_g.sort_values(by='group_num')
        n_res = df_g.shape[0]
        idx = np.arange(n_res)
        i = idx[df_g['group_num'].values == g][0]  # locate the central residue
        if (i == 0) | (i == n_res-1):
            # i is the smallest or largest, so can't find previous or next group for the coordinates calculation
            continue

        coords = df_g[['x', 'y', 'z']].values
        coords = coords - coords[i]  # center
        # coords of the previous and next group in local peptide
        c1 = coords[i-1]
        c2 = coords[i+1]

        rotate_mat = _rotation_matrix(c1, c2)

        coords = np.squeeze(np.matmul(rotate_mat[None, :, :], coords[:, :, None]))

        distance = np.sqrt(np.sum(coords**2, axis=1))

        segment_info = np.ones(n_res, dtype=int) * 3
        segment_info[i] = 0
        segment_info[i-1] = 1
        segment_info[i+1] = 2

        df_g = pd.DataFrame({'center_num': df_g['center_num'],
                             'group_num': df_g['group_num'],
                             'group_name': df_g['group_name'],
                             'x': coords[:, 0],
                             'y': coords[:, 1],
                             'z': coords[:, 2],
                             'distance': distance,
                             'segment': segment_info})
        df_g = df_g.sort_values(by='distance')
        df_list.append(df_g)
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(f'local_rot/{fname}_rot.csv', index=False)


def extract_one_topk(fname, k=10):
    df = pd.read_csv(f'cath/{fname}_bead.csv')
    gnum = df['group_num'].values
    gname = df['group_name'].values
    gcoords = df[['x', 'y', 'z']].values

    dist_mat = np.sqrt(((gcoords[None, :, :] - gcoords[:, None, :]) ** 2).sum(axis=2))

    gc_num_list = []
    g_num_list = []
    g_name_list = []
    coords_list = []
    dist_list = []

    for i, gc in enumerate(gnum):
        dist_i = dist_mat[i]
        dist_i_arg = np.argsort(dist_i)
        topk_arg = dist_i_arg[:k]
        gnum_i = gnum[topk_arg]
        gname_i = gname[topk_arg]
        coords_i = gcoords[topk_arg]
        dist_i = dist_i[topk_arg]

        g_num_list.extend(gnum_i)
        gc_num_list.extend([gc] * len(gnum_i))
        g_name_list.extend(gname_i)
        coords_list.append(coords_i)
        dist_list.extend(dist_i)

    coords = np.vstack(coords_list)
    df_chain = pd.DataFrame({'center_num': gc_num_list,
                             'group_num': g_num_list,
                             'group_name': g_name_list,
                             'x': coords[:, 0],
                             'y': coords[:, 1],
                             'z': coords[:, 2],
                             'distance': dist_list})

    df_chain.to_csv(f'local/{fname}.csv', index=False)


def extract_one_dist(fname, distance_cutoff=10):
    df = pd.read_csv(f'cath/{fname}_bead.csv')
    gnum = df['group_num'].values
    gname = df['group_name'].values
    gcoords = df[['x', 'y', 'z']].values

    dist_mat = np.sqrt(((gcoords[None, :, :] - gcoords[:, None, :]) ** 2).sum(axis=2))

    gc_num_list = []
    g_num_list = []
    g_name_list = []
    coords_list = []
    dist_list = []

    dist_ind = (dist_mat < distance_cutoff)
    for i, gc in enumerate(gnum):
        ind = dist_ind[i]
        g_num_i = gnum[ind]
        g_num_list.extend(g_num_i)
        gc_num_list.extend([gc] * len(g_num_i))
        g_name_list.extend(gname[ind])
        coords_list.append(gcoords[ind])
        dist_list.extend(dist_mat[i][ind])
    # print(dist_mat[i][ind])

    coords = np.vstack(coords_list)
    df_chain = pd.DataFrame({'center_num': gc_num_list,
                             'group_num': g_num_list,
                             'group_name': g_name_list,
                             'x': coords[:, 0],
                             'y': coords[:, 1],
                             'z': coords[:, 2],
                             'distance': dist_list})

    df_chain.to_csv(f'local/{fname}.csv', index=False)


def extract_batch(batch):
    for fname in tqdm(batch):
        # extract_one_topk(fname)
        if os.path.exists(f'local_rot/{fname}_rot.csv'):
            continue
        # print(fname)
        rotate_one(fname)


def extract_all():
    flist = pd.read_csv('flist.txt')['fname'].values
    np.random.shuffle(flist)
    num = flist.shape[0]
    num_cores = 40
    batch_size = num // num_cores + 1

    batch_list = []
    for i in range(0, num, batch_size):
        batch = flist[i:i+batch_size]
        batch_list += [batch]

    # setup the multi-processes
    with mp.Pool(processes=num_cores) as pool:
        pool.map(extract_batch, batch_list)


if __name__ == '__main__':
    # fname = '12AS_A'
    # extract_one_topk(fname)
    extract_all()




