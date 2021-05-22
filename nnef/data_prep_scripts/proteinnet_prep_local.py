import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import h5py


def read_protein_from_file(file_pointer):
    dict_ = {}
    _mask_dict = {'-': 0, '+': 1}

    while True:
        next_line = file_pointer.readline()
        if next_line == '[ID]\n':
            id_ = file_pointer.readline()[:-1]
            dict_.update({'id': id_})
        elif next_line == '[PRIMARY]\n':
            primary = file_pointer.readline()[:-1]
            dict_.update({'primary': primary})
        elif next_line == '[EVOLUTIONARY]\n':
            evolutionary = []
            for _residue in range(21):
                evolutionary.append(np.array([float(step) for step in file_pointer.readline().split()]))
            evolutionary = np.vstack(evolutionary)
            dict_.update({'evolutionary': evolutionary})
        elif next_line == '[SECONDARY]\n':
            file_pointer.readline()
        elif next_line == '[TERTIARY]\n':
            tertiary = []
            # 3 dimension
            for _axis in range(3):
                tertiary.append(np.array([float(coord) for coord in file_pointer.readline().split()]))
            tertiary = np.vstack(tertiary)
            dict_.update({'tertiary': tertiary})
        elif next_line == '[MASK]\n':
            mask = np.array([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
            dict_.update({'mask': mask})
        elif next_line == '\n':
            return dict_
        elif next_line == '':
            return None


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


def extract_one_topk(protein, local_rot_dir, k=10):
    amino_acids = pd.read_csv('amino_acids.csv')
    vocab = {x: y.upper() for x, y in zip(amino_acids.AA, amino_acids.AA3C)}

    pdb_id = protein['id']

    if pdb_id == '1X18_5_E':
        return

    idx = (protein['mask'] == 1)
    group_num = np.arange(len(protein['mask']))[idx]

    if len(group_num) < 20:
        return

    group_name = np.array([vocab[x] for x in protein['primary']])[idx]
    group_coords = protein['tertiary'].reshape((3, -1, 3))
    group_coords = group_coords[:, :, 1].T[idx] / 100  # pico-meters --> Angstrom

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
                             'segment': segment_info})
        df_g = df_g.sort_values(by=['segment', 'distance'])
        df_list.append(df_g)
    if len(df_list) > 0:
        df = pd.concat(df_list, ignore_index=True)

        evo_profile = protein['evolutionary'].T[:, :20]
        group_profile = evo_profile[df['group_num'].values]
        for i in range(20):
            df[f'aa{i}'] = group_profile[:, i]

        df.to_csv(f'{local_rot_dir}/{pdb_id}.csv', index=False, float_format='%.3f')
    else:
        print(group_name)


# def extract_one_topk_v0(protein, k=10):
#     pdb_id = protein['id']
#
#     if pdb_id == '1X18_5_E':
#         return
#
#     idx = (protein['mask'] == 1)
#     group_num = np.arange(len(protein['mask']))[idx]
#
#     if len(group_num) < 20:
#         return
#
#     group_name = np.array([vocab[x] for x in protein['primary']])[idx]
#     group_coords = protein['tertiary'].reshape((3, -1, 3))
#     group_coords = group_coords[:, :, 1].T[idx] / 100  # pico-meters --> Angstrom
#     dist_mat = np.sqrt(((group_coords[None, :, :] - group_coords[:, None, :]) ** 2).sum(axis=2))
#
#     df_list = []
#     for i, gc in enumerate(group_num):
#
#         dist_i = dist_mat[i]
#         dist_i_arg = np.argsort(dist_i)
#         topk_arg = dist_i_arg[:k]
#
#         gnum = group_num[topk_arg]
#         gname = group_name[topk_arg]
#         coords = group_coords[topk_arg]
#
#         if (gc-1 not in gnum) | (gc+1 not in gnum):
#             continue
#         cen_i = (gnum == gc)
#         pre_i = (gnum == gc-1)
#         next_i = (gnum == gc+1)
#         coords = coords - coords[cen_i]  # center
#         # coords of the previous and next group in local peptide
#         c1 = coords[pre_i]
#         c2 = coords[next_i]
#
#         if np.sum(c1**2) == 0:
#             break
#
#         if np.sum(c2**2) == 0:
#             break
#
#         rotate_mat = _rotation_matrix(c1, c2)
#
#         # if np.sum(np.isnan(rotate_mat)) > 0:
#         #     print(pdb_id, protein)
#         #     import pickle
#         #     pickle.dump(protein, open('nan_protein.pkl', 'wb'))
#         #     # raise ValueError('R has nan')
#         #     continue
#
#         coords = np.squeeze(np.matmul(rotate_mat[None, :, :], coords[:, :, None]))
#
#         distance = np.sqrt(np.sum(coords**2, axis=1))
#
#         segment_info = np.ones(k, dtype=int) * 3
#         segment_info[cen_i] = 0
#         segment_info[pre_i] = 1
#         segment_info[next_i] = 2
#
#         df_g = pd.DataFrame({'center_num': gc,
#                              'group_num': gnum,
#                              'group_name': gname,
#                              'x': coords[:, 0],
#                              'y': coords[:, 1],
#                              'z': coords[:, 2],
#                              # 'distance': distance,
#                              'segment': segment_info})
#         df_g = df_g.sort_values(by='segment')
#         df_list.append(df_g)
#     if len(df_list) > 0:
#         df = pd.concat(df_list, ignore_index=True)
#
#         evo_profile = protein['evolutionary'].T[:, :20]
#         group_profile = evo_profile[df['group_num'].values]
#         for i in range(20):
#             df[f'aa{i}'] = group_profile[:, i]
#
#         df.to_csv(f'local_rot/{pdb_id}.csv', index=False, float_format='%.3f')
#     else:
#         print(group_name)
#

# def write_pdb(protein):
#     amino_acids = pd.read_csv('amino_acids.csv')
#     vocab = {x: y.upper() for x, y in zip(amino_acids.AA, amino_acids.AA3C)}
#
#     pdb_id = protein['id']
#     group_name = [vocab[x] for x in protein['primary']]
#     num = np.arange(len(group_name))
#     mask = protein['mask']
#     coords = protein['tertiary']
#     coords = coords.reshape(3, -1, 3)[:, :, 1] / 100
#     x = coords[0, :]
#     y = coords[1, :]
#     z = coords[2, :]
#     with open(f'{pdb_id}.pdb', 'wt') as mf:
#         for i in range(len(num)):
#             if mask[i] == 1:
#                 mf.write(f'ATOM  {num[i]:5d}   CA {group_name[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')
#
#
# def write_df(protein):
#     pdb_id = protein['id']
#     group_name = np.array([vocab[x] for x in protein['primary']])
#     group_coords = protein['tertiary'].reshape((3, -1, 3))
#     group_coords = group_coords[:, :, 1] / 100  # pico-meters --> Angstrom
#     evo_profile = protein['evolutionary'].T[:, :20]
#
#     df = pd.DataFrame({'pdb_id': pdb_id,
#                        'mask': protein['mask'],
#                        'group_name': group_name,
#                        'x': group_coords[0],
#                        'y': group_coords[1],
#                        'z': group_coords[0],
#                        'evo_profile': evo_profile})
#     df.to_pickle(f'local_rot/{pdb_id}.pkl')
############################################


###########################################


###########################################
def extract_protein_info():
    input_file = 'training_30'
    # input_file = 'validation'
    # input_file = 'testing'
    # input_file = 'training_100'
    print("Processing raw data file", input_file)
    input_file_pointer = open("./" + input_file, "r")

    # write protein ids to text file
    protein_list = []
    seq_len = []
    pdb_res_count = []
    profile_info = []
    while True:
        # while there's more proteins to process
        protein = read_protein_from_file(input_file_pointer)
        if protein is None:
            break
        protein_list.append(protein['id'])
        seq_len.append(len(protein['primary']))
        pdb_res_count.append(protein['mask'].sum())
        profile_info.append(protein['evolutionary'][20].mean())
    df = pd.DataFrame({'pdb': protein_list, 'seq_len': seq_len,
                       'pdb_res_count': pdb_res_count, 'profile_info': profile_info})
    df.to_csv(f'{input_file}_protein_id.csv', index=False)


###########################################
def extract_local_structure():
    # input_file = 'training_30'
    # input_file = 'validation'
    # input_file = 'testing'
    input_file = 'training_100'

    print("Processing raw data file", input_file)

    input_file_pointer = open("./" + input_file, "r")
    local_rot_dir = f'./local_rot_{input_file}/'

    # amino_acids = pd.read_csv('amino_acids.csv')
    # vocab = {x: y.upper() for x, y in zip(amino_acids.AA, amino_acids.AA3C)}

    count = 0
    protein_list = []
    while True:
        # while there's more proteins to process
        protein = read_protein_from_file(input_file_pointer)
        if protein is None:
            break
        protein_list.append(protein)
        count += 1
        # print(count)  # 25299
        # total residues count = 5,274,865
        # break

    def extract_batch(batch):
        for i in tqdm(batch):
            # extract_one_topk(fname)
            # if os.path.exists(f'local_rot/{fname}_rot.csv'):
            #     continue
            # print(fname)
            extract_one_topk(protein_list[i], local_rot_dir)

    count = len(protein_list)
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
def save_local_struct_2_h5():
    input_file = 'training_30'
    # input_file = 'validation'
    # input_file = 'testing'
    local_rot_dir = f'./local_rot_{input_file}/'

    flist = pd.read_csv(f'{local_rot_dir}/flist.txt')['fname'].values
    df_list = []
    for fname in tqdm(flist):
        df = pd.read_csv(f'{local_rot_dir}/{fname}')
        if (input_file == 'validation') | (input_file == 'testing'):
            df['pdb'] = fname[:-4]
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    # save data to DataFrame
    # df['num'] = np.arange(df.shape[0]//10).repeat(10, axis=0)
    # input_file = 'training_30'
    # df.to_csv(f'{input_file}.csv', index=False, float_format='%.3f')

    # save data to hdf5
    amino_acids = pd.read_csv('amino_acids.csv')
    vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    # segment = df['segment'].values
    k = 15
    seq = df['group_name'].apply(lambda x: vocab[x])
    seq = seq.values.reshape((-1, k))
    coords = df[['x', 'y', 'z']].values.reshape((-1, k, 3))
    profile = df[[f'aa{i}' for i in range(20)]].values.reshape((-1, k, 20))

    if (input_file == 'validation') | (input_file == 'testing'):
        pdb = df['pdb'].values.reshape((-1, k))[:, 0]
        pdb_df = pd.DataFrame({'pdb': pdb})
        pdb_df.to_csv(f'{input_file}_pdb.csv', index=False)

    print(seq.min(), coords.min(), profile.min())
    print(seq.shape, coords.shape, profile.shape)

    with h5py.File(f'{input_file}.h5', 'w') as f:
        dset = f.create_dataset("seq", shape=seq.shape, data=seq, dtype='i')
        dset = f.create_dataset("coords", shape=coords.shape, data=coords, dtype='f4')
        dset = f.create_dataset("profile", shape=profile.shape, data=profile, dtype='f4')


def shuffle_h5():
    # shuffle the h5py file
    data = h5py.File('training_30.h5', 'r')
    k = 15

    seq_all = data['seq'][()]
    coords_all = data['coords'][()]
    profile_all = data['profile'][()]
    num = data['seq'].shape[0]

    idx = np.arange(num)
    np.random.shuffle(idx)

    seq = seq_all[idx]
    coords = coords_all[idx]
    profile = profile_all[idx]

    with h5py.File(f'training_30_shuffle.h5', 'w') as f:
        dset = f.create_dataset("seq", shape=seq.shape, data=seq, dtype='i')
        dset = f.create_dataset("coords", shape=coords.shape, data=coords, dtype='f4')
        dset = f.create_dataset("profile", shape=profile.shape, data=profile, dtype='f4')

    with h5py.File(f'training_30_small.h5', 'w') as f:
        dset = f.create_dataset("seq", shape=(1000, k), data=seq[:1000], dtype='i')
        dset = f.create_dataset("coords", shape=(1000, k, 3), data=coords[:1000], dtype='f4')
        dset = f.create_dataset("profile", shape=(1000, k, 20), data=profile[:1000], dtype='f4')


def clean_h5():
    # clean the h5py file, remove local structure where n-2, n+2 residues have distance > 10A.
    data = h5py.File('training_30_shuffle.h5', 'r')
    k = 15

    seq = data['seq'][()]
    coords = data['coords'][()]
    profile = data['profile'][()]

    c1 = coords[:, 3, :]
    c2 = coords[:, 4, :]
    d1 = np.sqrt(np.sum(c1**2, axis=-1))
    d2 = np.sqrt(np.sum(c2**2, axis=-1))

    idx = (d1 < 10) & (d2 < 10)

    seq = seq[idx]
    coords = coords[idx]
    profile = profile[idx]

    with h5py.File(f'training_30_shuffle2.h5', 'w') as f:
        dset = f.create_dataset("seq", shape=seq.shape, data=seq, dtype='i')
        dset = f.create_dataset("coords", shape=coords.shape, data=coords, dtype='f4')
        dset = f.create_dataset("profile", shape=profile.shape, data=profile, dtype='f4')

    with h5py.File(f'training_30_small.h5', 'w') as f:
        dset = f.create_dataset("seq", shape=(1000, k), data=seq[:1000], dtype='i')
        dset = f.create_dataset("coords", shape=(1000, k, 3), data=coords[:1000], dtype='f4')
        dset = f.create_dataset("profile", shape=(1000, k, 20), data=profile[:1000], dtype='f4')


####################################

def extract_profile():
    # input_file = 'training_30'
    input_file = 'training_100'
    input_file_pointer = open("./" + input_file, "r")
    pdb_list = pd.read_csv(f'pdb_profile_{input_file}/pdb_list.txt')['pdb'].values

    while True:
        # while there's more proteins to process
        protein = read_protein_from_file(input_file_pointer)
        if protein is None:
            break
        protein_id = protein['id'].split('_')
        protein_id = protein_id[0] + protein_id[-1]

        if protein_id in pdb_list:
            print(protein_id)
            seq = protein['primary']
            evo_profile = protein['evolutionary'].T[:, :20]
            mask = protein['mask']

            df = pd.DataFrame({'seq': list(seq), 'mask': mask})
            for i in range(20):
                df[f'aa{i}'] = evo_profile[:, i]
            df.to_csv(f'pdb_profile_{input_file}/{protein_id}_profile.csv', index=False, float_format='%.3f')


def extract_profile_val():
    input_file = 'validation'
    input_file_pointer = open("./" + input_file, "r")

    while True:
        # while there's more proteins to process
        protein = read_protein_from_file(input_file_pointer)
        if protein is None:
            break
        protein_id = protein['id'].split('_')
        if len(protein_id) == 3:
            protein_id = protein_id[0][3:] + protein_id[-1]
            print(protein_id)
            seq = protein['primary']
            evo_profile = protein['evolutionary'].T[:, :20]

            df = pd.DataFrame({'seq': list(seq)})
            for i in range(20):
                df[f'aa{i}'] = evo_profile[:, i]
            df.to_csv(f'pdb_profile_{input_file}/{protein_id}_profile.csv', index=False, float_format='%.3f')


def extract_profile_test():
    input_file = 'testing'
    input_file_pointer = open("./" + input_file, "r")

    while True:
        # while there's more proteins to process
        protein = read_protein_from_file(input_file_pointer)
        if protein is None:
            break
        protein_id = protein['id']
        print(protein_id)
        seq = protein['primary']
        evo_profile = protein['evolutionary'].T[:, :20]
        mask = protein['mask']
        df = pd.DataFrame({'seq': list(seq), 'mask': mask})
        for i in range(20):
            df[f'aa{i}'] = evo_profile[:, i]
        df.to_csv(f'pdb_profile_{input_file}/{protein_id}_profile.csv', index=False, float_format='%.3f')


def extract_profile_zdock():
    # input_file = 'training_30'
    input_file = 'training_100'
    input_file_pointer = open("./" + input_file, "r")
    pdb_list = pd.read_csv(f'zdock_pdb_profile/zdock_training_100.txt')['pdb'].values

    while True:
        # while there's more proteins to process
        protein = read_protein_from_file(input_file_pointer)
        if protein is None:
            break
        protein_id = protein['id'].split('_')
        protein_id = protein_id[0] + protein_id[-1]

        if protein_id in pdb_list:
            print(protein_id)
            seq = protein['primary']
            evo_profile = protein['evolutionary'].T[:, :20]
            mask = protein['mask']

            df = pd.DataFrame({'seq': list(seq), 'mask': mask})
            for i in range(20):
                df[f'aa{i}'] = evo_profile[:, i]
            df.to_csv(f'pdb_profile_{input_file}/{protein_id}_profile.csv', index=False, float_format='%.3f')


def extract_one_protein():

    def write_one_protein(protein):
        pdb_id = protein['id']
        mask = protein['mask']
        seq = protein['primary']

        # group_name = np.array([vocab[x] for x in protein['primary']])[idx]
        group_coords = protein['tertiary'].reshape((3, -1, 3))
        group_coords = group_coords[:, :, 1].T / 100  # pico-meters --> Angstrom

        group_profile = protein['evolutionary'].T[:, :20]

        df = pd.DataFrame({'group_name': list(seq),
                           'mask': mask,
                           'x': group_coords[:, 0],
                           'y': group_coords[:, 1],
                           'z': group_coords[:, 2]})

        for i in range(20):
            df[f'aa{i}'] = group_profile[:, i]

        df.to_csv(f'small_protein/{pdb_id}_bead.csv', index=False, float_format='%.3f')
        # extract_one_topk(protein, 'bpti')

    # pdb_id = '1BPI_1_A'
    pdb_selected = ['2F4K_1_A', '2F21_d2f21a1', '2HBA_1_A', '2WXC_1_A', '2JOF_1_A', '1FME_1_A', '2P6J_1_A', '2A3D_1_A']

    input_file = 'training_100'
    input_file_pointer = open("./" + input_file, "r")

    while True:
        # while there's more proteins to process
        protein = read_protein_from_file(input_file_pointer)
        if protein is None:
            break
        # if protein['id'] == pdb_id:
        #     break
        if protein['id'] in pdb_selected:
            print(protein['id'])
            write_one_protein(protein)


def extract_small_proteins():
    df = pd.read_csv('training_100_protein_id.csv')
    df = pd.read_csv('validation_protein_id.csv')

    seq_len = df['seq_len'].values
    pdb_res_count = df['pdb_res_count'].values
    profile_info = df['profile_info'].values
    idx = (seq_len < 40) & (pdb_res_count == seq_len) & (profile_info < 0.6)

    df2 = df[idx]
    pdb = []
    for p in df2['pdb']:
        if p[-4:] == '_1_A':
            print(p)
            pdb.append(p[:-4])

    pdb = np.array([x[:4] for x in pdb])
    pdb_small = ['2JOF', '1FME', '2F4K', '2F21', '2HBA', '2WXC', '1PRB', '2P6J', '1MIO', '2A3D', '1LMB']
    for p in pdb_small:
        print(p, df[pdb == p])

    selected = ['2F4K_1_A', '2F21_d2f21a1', '2HBA_1_A', '2WXC_1_A', '2JOF_1_A', '1FME_1_A', '2P6J_1_A', '2A3D_1_A']


def extract_proteins():

    def write_one_protein(protein, profile_dir):
        pdb_id = protein['id']
        mask = protein['mask']
        seq = protein['primary']

        # group_name = np.array([vocab[x] for x in protein['primary']])[idx]
        group_num = np.arange(len(seq))
        group_coords = protein['tertiary'].reshape((3, -1, 3))
        group_coords = group_coords[:, :, 1].T / 100  # pico-meters --> Angstrom

        group_profile = protein['evolutionary'].T[:, :20]

        df = pd.DataFrame({'group_num': group_num,
                           'group_name': list(seq),
                           'mask': mask,
                           'x': group_coords[:, 0],
                           'y': group_coords[:, 1],
                           'z': group_coords[:, 2]})

        for i in range(20):
            df[f'aa{i}'] = group_profile[:, i]

        df.to_csv(f'{profile_dir}/{pdb_id}_profile.csv', index=False, float_format='%.4f')

    # pdb_list = pd.read_csv('training_100_validation/flist.txt')['pdb']
    # pdb_dict = {x: 1 for x in pdb_list}

    # input_file = 'training_100'
    input_file = 'validation'
    profile_dir = f'{input_file}_profile'

    input_file_pointer = open("./" + input_file, "r")

    count = 0
    while True:
        # while there's more proteins to process
        protein = read_protein_from_file(input_file_pointer)
        if protein is None:
            break
        # if protein['id'] == pdb_id:
        #     break
        pdb_id = protein['id'].split('_')
        if len(pdb_id) == 3:
            write_one_protein(protein, profile_dir)

        # if input_file == 'validation':
        #     pdb_id = pdb_id[0][3:] + '_' + pdb_id[-1]
        # else:
        #     pdb_id = pdb_id[0] + '_' + pdb_id[-1]
        # try:
        #     if pdb_dict[pdb_id] == 1:
        #         write_one_protein(protein)
        #     count += 1
        #     print(count)
        # except KeyError:
        #     pass









