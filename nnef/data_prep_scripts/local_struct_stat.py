import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import h5py
import matplotlib.pyplot as pl
import os


def plot_res_env():
    amino_acids = pd.read_csv('../amino_acids.csv')
    vocab = {x-1: y.upper() for x, y in zip(amino_acids.idx, amino_acids.AA3C)}

    data = h5py.File(f'training_30_v3.h5', 'r')
    seq = data['seq'][()][:, 0]
    env = data['res_counts'][()]

    for i in range(20):
        idx = (seq == i)
        fig = pl.figure()
        for j in range(3):
            count = env[:, j][idx]
            pl.hist(count, bins=np.arange(count.max()+1), histtype='step')
        pl.savefig(f'aa_{vocab[i]}.pdf')
        pl.close(fig)

    # pl.figure()
    # pl.plot(phi[:5000] + 180, 180-theta[:5000], 'g.')
    # pl.ylim(0, 180)


def plot_coords_internal_stat(mode='CA'):
    data = h5py.File(f'training_30_{mode}_c-next_sample.h5', 'r')
    c_next = torch.tensor(data['coords_internal'][()])

    r = torch.norm(c_next, dim=-1)
    theta = torch.acos(c_next[:, 0] / r)
    phi = torch.atan2(c_next[:, 2], c_next[:, 1])  # atan2 considers the quadrant,

    r = r.cpu().numpy()
    theta = theta.cpu().numpy()
    phi = phi.cpu().numpy()
    #
    # r = data['r'][()]
    # theta = data['theta'][()]
    # phi = data['phi'][()]

    theta = theta / np.pi * 180.0
    phi = phi / np.pi * 180.0

    pl.figure()
    pl.plot(theta[:5000], phi[:5000], 'g.')
    pl.xlabel('theta')
    pl.ylabel('phi')
    pl.savefig(f'theta_phi_{mode}.pdf')

    pl.figure()
    pl.hist(r, bins=np.arange(10))
    pl.savefig(f'r_hist_{mode}.pdf')

    # pl.figure()
    # pl.plot(phi[:5000] + 180, 180-theta[:5000], 'g.')
    # pl.ylim(0, 180)


def plot_coords_angles(mode='CA'):
    device = torch.device('cuda')
    data = h5py.File(f'training_30_{mode}_v2.h5', 'r')
    coords = data['coords'][()]
    coords = torch.tensor(coords, device=device)

    r = torch.norm(coords, dim=-1)
    theta = torch.acos(coords[:, 1:, 2] / r[:, 1:])  # exclude the origin
    phi = torch.atan2(coords[:, 1:, 1], coords[:, 1:, 0])  # atan2 considers the quadrant,

    theta = theta.cpu().numpy()
    phi = phi.cpu().numpy()

    theta = theta / np.pi * 180.0
    phi = phi / np.pi * 180.0

    pl.figure()
    pl.plot(theta[:500].flatten(), phi[:500].flatten(), 'g.')
    pl.xlabel('theta')
    pl.ylabel('phi')
    pl.savefig(f'training_theta_phi_{mode}.pdf')

    pl.figure()
    pl.plot(theta[:500, :4].flatten(), phi[:500, :4].flatten(), 'g.')
    pl.xlabel('theta')
    pl.ylabel('phi')
    pl.savefig(f'training_theta_phi_{mode}_central.pdf')

    pl.figure()
    pl.plot(theta[:500, 4:].flatten(), phi[:500, 4:].flatten(), 'g.')
    pl.xlabel('theta')
    pl.ylabel('phi')
    pl.savefig(f'training_theta_phi_{mode}_Nk.pdf')

    for i in range(13):
        pl.figure()
        pl.plot(theta[:5000, 1+i], phi[:5000, 1+i], 'g.')
        pl.xlabel('theta')
        pl.ylabel('phi')
        pl.savefig(f'training_theta_phi_{mode}_N{1+i}.pdf')


def coords_internal_stat(mode='CA'):
    device = torch.device('cuda')
    # data = h5py.File('training_30_shuffle2.h5', 'r')
    data = h5py.File(f'training_30_{mode}.h5', 'r')
    coords = data['coords'][()]
    c1 = torch.tensor(coords[:, 1, :], device=device)  # c-1 residue
    c2 = torch.tensor(coords[:, 3, :], device=device)  # c-2 residue
    c_next = torch.tensor(coords[:, 2, :], device=device)  # c+1 residue

    # c1 is minus X-axis, c1 c2 are X-Y plane.
    z = torch.cross(c1, c2, dim=-1)
    x = -c1
    y = torch.cross(z, x, dim=-1)
    x = x / torch.norm(x, dim=-1, keepdim=True)
    y = y / torch.norm(y, dim=-1, keepdim=True)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    R = torch.cat([x, y, z], dim=-1).view(-1, 3, 3)  # (N, 3, 3)

    c_next = torch.matmul(R[:, :, :], c_next[:, :, None]).squeeze()
    # # theta is the angle to X-axis, phi is the angle in Y-Z plane
    # r = torch.norm(c_next, dim=-1)
    # theta = torch.acos(c_next[:, 0] / r)
    # phi = torch.atan2(c_next[:, 2], c_next[:, 1])  # atan2 considers the quadrant,
    #
    # r = r.cpu().numpy()
    # theta = theta.cpu().numpy()
    # phi = phi.cpu().numpy()
    #
    # with h5py.File(f'training_30_shuffle2_stat.h5', 'w') as f:
    #     dset = f.create_dataset("r", shape=r.shape, data=r, dtype='f4')
    #     dset = f.create_dataset("theta", shape=theta.shape, data=theta, dtype='f4')
    #     dset = f.create_dataset("phi", shape=phi.shape, data=phi, dtype='f4')

    # with h5py.File(f'training_30_{mode}_c-next.h5', 'w') as f:
    #     coord_internal = c_next.cpu().numpy()
    #     dset = f.create_dataset("coords_internal", shape=coord_internal.shape, data=coord_internal, dtype='f4')

    with h5py.File(f'training_30_{mode}_c-next_sample.h5', 'w') as f:
        i = torch.randint(0, c_next.shape[0], (100000,))
        coord_internal = c_next[i].cpu().numpy()
        dset = f.create_dataset("coords_internal", shape=coord_internal.shape, data=coord_internal, dtype='f4')


def get_local_stat():
    pdb_list = pd.read_csv('training_30_beads/flist3.txt')['pdb'].values
    df_list = []
    for pdb in pdb_list:
        data_path = f'training_30_beads/{pdb}_count_res.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df_list.append(df)
    df = pd.concat(df_list)
    df.to_csv('training_30_bead_count_res.csv', index=False)

    count6a, count8a, count10a, count12a = df['count_6a'], df['count_8a'], df['count_10a'], df['count_12a']
    print(count6a.max(), count8a.max(), count10a.max(), count12a.max())

    import matplotlib.pyplot as pl
    for p in df.columns:
        print(p, df[p].max())
        pl.figure()
        pl.hist(df[p], bins=np.arange(df[p].max()+1))
        pl.savefig(f'{p}.pdf')


def plot_stat_local_struct():
    input_file = 'training_30'
    # input_file = 'validation'
    # input_file = 'testing'
    local_rot_dir = f'local_rot_{input_file}/'
    local_rot_dir_v2 = f'local_rot_{input_file}_v2/'

    flist = pd.read_csv(f'{local_rot_dir}/flist.txt')['fname'].values

    # concat all dataframes
    df_list = []
    for fname in tqdm(flist):
        df = pd.read_csv(f'{local_rot_dir_v2}/{fname}')
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)

    distance = df['distance'].values
    dist = distance.reshape(-1, 15)
    dist_max = dist.max(axis=-1)

    idx = (dist[:, 1] < 4) & (dist[:, 2] < 4) & (dist[:, 3] < 8) & (dist[:, 4] < 8) & (dist_max < 20)

    seg = df['seg'].values
    seg = seg.reshape(-1, 15)
    seg_num = seg.max(axis=-1)

    pl.figure()
    pl.hist(dist_max, bins=np.arange(12)*10)
    pl.yscale('log')
    pl.savefig('local_rot_training_30_v2_dist_max.pdf')

    pl.figure()
    pl.hist(seg_num, bins=np.arange(11)+1.5)
    pl.savefig('local_rot_training_30_v2_seg_num.pdf')



def count_pep(group_num):
    i = 1
    prev_g = group_num[0]
    if type(prev_g) == str:
        if prev_g[-1].isalpha():
            prev_g = int(prev_g[:-1])
        else:
            prev_g = int(prev_g)

        for g in group_num[1:]:
            if g[-1].isalpha():
                g = int(g[:-1])
            else:
                g = int(g)
            if g - prev_g > 1:
                i += 1
            prev_g = g
    else:
        for g in group_num[1:]:
            if g - prev_g > 1:
                i += 1
            prev_g = g
    return i


def local_stat_v0():
    flist = pd.read_csv('flist.txt')['fname']
    fsample = flist.sample(1000).values

    count = 0
    for f in tqdm(flist):
        df = pd.read_csv(f)
        count += df['center_num'].nunique()  # total number of central residue: 3615824

    num_hist = np.zeros(10, dtype=int)
    for f in tqdm(fsample):
        df = pd.read_csv(f)
        df = df[df['distance'] < 8]
        # count number of residues in each local structure
        num = df['center_num'].value_counts()
        num_hist += np.histogram(num, bins=np.arange(11) * 10)[0]
    print(num_hist)
    # [41803 41304 66611 54287 26288  4867   242    75    12     0]

    n_pep_list = []
    res_pep_ratio = []
    for f in tqdm(fsample):
        df = pd.read_csv(f)
        df = df[df['distance'] < 6]
        # count number of peptide in each local structure
        for g in df['center_num'].unique():
            group_num = df['group_num'][df['center_num'] == g].values
            n_pep = count_pep(group_num)
            n_pep_list.append(n_pep)
            res_pep_ratio.append(group_num.shape[0] * 1.0 / n_pep)

    n_pep = np.array(n_pep_list)
    print(np.histogram(n_pep, bins=np.arange(21)*2))

    res_pep_ratio = np.array(res_pep_ratio)
    print(np.histogram(res_pep_ratio, bins=np.arange(15)))

    # array([ 28355, 130840,  64953,   4615,     37,      0,      0,      0,63.73it/s]
    #             0,      0,      0,      0,      0,      0,      0,      0,
    #             0,      0,      0,      0]),
    # array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
    #        34, 36, 38, 40]))
    # (array([     0, 109824,  85764,  21641,   6994,   3615,    793,    144,
    #            20,      4,      1,      0,      0,      0]),
    #  array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]))






