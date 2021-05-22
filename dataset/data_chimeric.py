from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm


class DatasetLocalGenCM(Dataset):
    def __init__(self, data, args):
        self.seq_factor = args.seq_factor  # 0.5
        self.seq_len = args.seq_len + 1
        self.noise_factor = args.noise_factor  # 0.001
        self.seq_type = args.seq_type
        self.residue_type_num = args.residue_type_num
        self.no_homology = args.no_homology

        if self.residue_type_num != 20:
            aa_types = pd.read_csv('data/aa_types.csv')
            assert(self.residue_type_num in [2, 3, 5, 7, 9])
            res_type = aa_types[f'type{self.residue_type_num}']
            self.vocab = {x-1: y for x, y in zip(aa_types.idx, res_type)}

        self.pdb_list = pd.read_csv(data)['pdb'].values

        self.num = self.pdb_list.shape[0]

        # hh_data_pdb = h5py.File('data/hhsuite_CB.h5', 'r', libver='latest', swmr=True)
        # hh_data_seq = h5py.File('data/hhsuite_pdb_seq.h5', 'r', libver='latest', swmr=True)
        hh_data_pdb = h5py.File('data/hhsuite_CB_cullpdb.h5', 'r', libver='latest', swmr=True)
        hh_data_seq = h5py.File('data/hhsuite_pdb_seq_cullpdb.h5', 'r', libver='latest', swmr=True)
        self.group_num_dict = {}
        self.coords_dict = {}
        self.start_id_dict = {}
        self.res_counts_dict = {}
        self.seq_dict = {}
        for pdb in tqdm(self.pdb_list):
            data_pdb = hh_data_pdb[pdb]
            self.group_num_dict[pdb] = data_pdb['group_num'][()]
            self.coords_dict[pdb] = data_pdb['coords'][()]
            self.start_id_dict[pdb] = data_pdb['start_id'][()]
            self.res_counts_dict[pdb] = data_pdb['res_counts'][()]
            self.seq_dict[pdb] = hh_data_seq[pdb][()]

        hh_data_pdb.close()
        hh_data_seq.close()

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        # select a pdb id
        pdb = self.pdb_list[item]
        # load all local structures of the pdb
        # hh_data_pdb = h5py.File('data/hhsuite_CB.h5', 'r', libver='latest', swmr=True)
        # hh_data_seq = h5py.File('data/hhsuite_pdb_seq.h5', 'r', libver='latest', swmr=True)
        # data_pdb = hh_data_pdb[pdb]
        # seq_chim_n = hh_data_seq[pdb][()]

        seq_chim_n = self.seq_dict[pdb]
        # random select one of the N chimeric sequence
        # k = np.random.randint(0, seq_chim_n.shape[0])
        if self.no_homology:
            k = 0
        else:
            k = torch.randint(0, seq_chim_n.shape[0], (1,))[0]
        seq_chim = seq_chim_n[k]

        # random select one of the L local structure
        # group_num_n = data_pdb['group_num']
        # i = np.random.randint(0, group_num_n.shape[0])
        # group_num = group_num_n[i]
        # start_id = data_pdb['start_id'][i]
        # coords = data_pdb['coords'][i]
        # res_counts = data_pdb['res_counts'][i]

        group_num_n = self.group_num_dict[pdb]
        # i = np.random.randint(0, group_num_n.shape[0])
        i = torch.randint(0, group_num_n.shape[0], (1,))[0]
        group_num = group_num_n[i]
        start_id = self.start_id_dict[pdb][i]
        coords = self.coords_dict[pdb][i]
        res_counts = self.res_counts_dict[pdb][i]

        if self.seq_len > group_num.shape[0]:
            raise ValueError('args.seq_len > seq data residue numbers')
        if self.seq_len < group_num.shape[0]:
            coords = coords[:self.seq_len]
            group_num = group_num[:self.seq_len]
            start_id = start_id[:self.seq_len]
            res_counts = res_counts[:self.seq_len]

        # i = 0
        # seq_all = seq_chim[i]
        # seq = np.array([self.vocab[seq_all[x]] for x in group_num])
        # assert (np.sum((seq - seq_native) ** 2) == 0)

        # if group_num.max() >= len(seq_chim):
        #     with open('data/error_group_num.txt', 'at') as ef:
        #         ef.write(f'{pdb}, {i}, {group_num.max()}, {len(seq_chim)}\n')
        #     seq = seq_native
        # else:
        #     try:
        #         seq = np.array([self.vocab[seq_chim[x]] for x in group_num])
        #     except KeyError:
        #         seq = seq_native

        seq = np.array([seq_chim[x] for x in group_num])

        if self.residue_type_num != 20:
            seq = np.array([self.vocab[x] for x in seq])

        seq = torch.tensor(seq, dtype=torch.long)

        # calculate radius and angle features
        coords = torch.tensor(coords, dtype=torch.float)
        r = torch.norm(coords, dim=-1)
        theta = torch.acos(coords[1:, 2] / r[1:])
        phi = torch.atan2(coords[1:, 1], coords[1:, 0])  # atan2 considers the quadrant,
        theta = F.pad(theta, (1, 0), value=0)
        phi = F.pad(phi, (1, 0), value=0)

        coords = torch.stack((r, theta, phi), dim=1)
        # print(coords.shape)

        # stack the featurs
        # x_real = torch.cat((coords, profile), dim=-1)
        # print(x_real.shape)
        start_id = torch.tensor(start_id, dtype=torch.long)
        res_counts = torch.tensor(res_counts, dtype=torch.float)
        # x_real = {'seq': profile, 'coords': coords, 'start_id': start_id}
        # return x_real

        # return item, k, i, seq, coords, start_id, res_counts
        return seq, coords, start_id, res_counts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_factor", type=float, default=0.5)
    parser.add_argument("--noise_factor", type=float, default=0.001)
    parser.add_argument("--seq_len", type=int, default=14)
    parser.add_argument("--dist_mask", action='store_true', default=False)
    parser.add_argument("--dist_cutoff", type=float, default=10)
    parser.add_argument("--seq_type", type=str, default='residue')
    parser.add_argument("--data_flag", type=str, default='training_30_CA_v2')
    parser.add_argument("--residue_type_num", type=int, default=20,
                        help='number of residue types used in the sequence vocabulary')

    args = parser.parse_args()

    train_dataset = DatasetLocalGenCM('data/hhsuite_CB_pdb_list.csv', args)

    pdb_weights = pd.read_csv(f'data/hhsuite_CB_pdb_list.csv')['weight'].values

    from torch.utils.data import DataLoader, WeightedRandomSampler

    datasampler = WeightedRandomSampler(weights=pdb_weights, num_samples=10)
    train_data_loader = DataLoader(train_dataset, batch_size=10, sampler=datasampler, pin_memory=True)





