import numpy as np
import pandas as pd
import torch
from Bio.PDB import Selection, PDBParser


def test_setup(args):
    from physics.protein_os import EnergyFun, ProteinBase
    from model import LocalTransformer

    device = torch.device(args.device)

    model = LocalTransformer(args)
    energy_fn = EnergyFun(model, args)

    model.load_state_dict(torch.load(f'{args.load_exp}/models/model.pt', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    ProteinBase.k = args.seq_len - 4
    ProteinBase.use_graph_net = args.use_graph_net

    return device, model, energy_fn, ProteinBase


def write_pdb(seq, coords, pdb_id, flag, exp_id):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.AA3C)}
    seq = [vocab[x] for x in seq]

    num = np.arange(coords.shape[0])
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    with open(f'data/fold/{exp_id}/{pdb_id}_{flag}.pdb', 'wt') as mf:
        for i in range(len(num)):
            mf.write(f'ATOM  {num[i]:5d}   CA {seq[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')


def write_pdb_sample(seq, coords_sample, pdb_id, flag, exp_id):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    # vocab = {x: y for x, y in zip(amino_acids.AA, amino_acids.AA3C)}
    # seq = [vocab[x] for x in seq]

    with open(f'data/fold/{exp_id}/{pdb_id}_{flag}.pdb', 'wt') as mf:
        for j, coords in enumerate(coords_sample):
            num_steps = (j + 1)
            mf.write('MODEL        '+str(num_steps)+'\n')

            num = np.arange(coords.shape[0])
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            for i in range(len(num)):
                mf.write(f'ATOM  {num[i]:5d}   CA {seq[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')
            mf.write('ENDMDL\n')


def write_pdb_sample2(seq, coords_sample, pdb_id, flag, save_dir):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x: y.upper() for x, y in zip(amino_acids.AA, amino_acids.AA3C)}
    seq = [vocab[x] for x in seq]

    with open(f'{save_dir}/{pdb_id}_{flag}.pdb', 'wt') as mf:
        for j, coords in enumerate(coords_sample):
            num_steps = (j + 1)
            mf.write('MODEL        '+str(num_steps)+'\n')

            num = np.arange(coords.shape[0])
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            for i in range(len(num)):
                mf.write(f'ATOM  {num[i]:5d}   CA {seq[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')
            mf.write('ENDMDL\n')


def transform_profile(seq, profile, noise_factor, seq_factor):
    seq_len = len(seq)
    # add noise to profile
    noise = np.random.rand(seq_len, 20) * noise_factor
    profile += noise

    # profile[range(seq_len), seq] += seq_factor
    # # normalize the profile
    # profile /= profile.sum(axis=1)[:, None]

    df = pd.read_csv('data/aa_freq.csv')
    aa_freq = df['freq'].values / df['freq'].sum()

    profile = profile / (profile + aa_freq)

    return profile


def load_protein_v0(data_path, pdb_id, mode, device, args):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}

    print(pdb_id)
    # pdb_id_bead = pdb_id.split('_')[0] + '_' + pdb_id.split('_')[2]

    # df_beads = pd.read_csv(f'data/fold/protein_sample/{pdb_id_bead}_bead.csv')
    df_beads = pd.read_csv(f'{data_path}/{pdb_id}_bead.csv')
    df_profile = pd.read_csv(f'{data_path}/{pdb_id}_profile.csv')

    seq = df_profile['group_name'].values
    seq_id = df_profile['group_name'].apply(lambda x: vocab[x]).values

    if mode == 'CA':
        coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    elif mode == 'CAS':
        coords = (df_beads[['xca', 'yca', 'zca']].values + df_beads[['xs', 'ys', 'zs']].values) / 2
    else:
        raise ValueError('mode should be CA / CB / CAS.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)

    seq_type = args.seq_type
    if seq_type == 'residue':
        profile = torch.tensor(seq_id, dtype=torch.long, device=device)
    else:
        profile = df_profile[[f'aa{i}' for i in range(20)]].values
        profile = transform_profile(seq_id, profile, args.noise_factor, args.seq_factor)
        profile = torch.tensor(profile, dtype=torch.float, device=device)
    return seq, coords, profile


def load_protein(data_path, pdb_id, mode, device, args):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    print(pdb_id)

    df_beads = pd.read_csv(f'{data_path}/{pdb_id}_bead.csv')

    seq = df_beads['group_name'].values
    seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values

    if mode == 'CA':
        coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    elif mode == 'CAS':
        coords = (df_beads[['xca', 'yca', 'zca']].values + df_beads[['xs', 'ys', 'zs']].values) / 2
    else:
        raise ValueError('mode should be CA / CB / CAS.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)

    profile = torch.tensor(seq_id, dtype=torch.long, device=device)
    return seq, coords, profile


def load_protein_bead(data_path, mode, device):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}
    vocab2 = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    # df_beads = pd.read_csv(f'data/casp14/{stage}/{casp_id}/{pdb_id}_bead.csv')
    df_beads = pd.read_csv(data_path)

    seq = df_beads['group_name'].values
    if len(seq[0]) == 1:
        seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values
    else:
        seq_id = df_beads['group_name'].apply(lambda x: vocab2[x]).values

    if mode == 'CA':
        coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    else:
        raise ValueError('mode should be CA / CB.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)
    profile = torch.tensor(seq_id, dtype=torch.long, device=device)

    return seq, coords, profile


def load_protein_decoy(pdb_id, decoy_id, mode, device, args):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}

    decoy_set = args.decoy_set
    profile_set = 'pdb_profile_training_100'

    # print(pdb_id)

    df_beads = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/{decoy_id}_bead.csv')

    seq_type = args.seq_type
    if seq_type != 'residue':
        df_profile = pd.read_csv(f'data/decoys/{decoy_set}/{profile_set}/{pdb_id}_profile.csv')

    seq = df_beads['group_name'].values
    seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values

    if mode == 'CA':
        coords = df_beads[['x', 'y', 'z']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    elif mode == 'CAS':
        coords = (df_beads[['xca', 'yca', 'zca']].values + df_beads[['xs', 'ys', 'zs']].values) / 2
    else:
        raise ValueError('mode should be CA / CB / CAS.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)

    if seq_type == 'residue':
        profile = torch.tensor(seq_id, dtype=torch.long, device=device)
    else:
        profile = df_profile[[f'aa{i}' for i in range(20)]].values
        profile = transform_profile(seq_id, profile, args.noise_factor, args.seq_factor)
        profile = torch.tensor(profile, dtype=torch.float, device=device)
    return seq, coords, profile


def extract_beads(pdb_path):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    p = PDBParser()
    structure = p.get_structure('X', pdb_path)
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

        try:
            res['CA'].get_coord()
            if res.get_resname() != 'GLY':
                res['CB'].get_coord()
        except KeyError:
            print(f'{pdb_path}, {res} missing CA / CB atoms')
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

    df.to_csv(f'{pdb_path}_bead.csv', index=False)
    return df


def load_protein_pdb(data_path, mode, device):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}
    vocab2 = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    # df_beads = pd.read_csv(f'data/casp14/{stage}/{casp_id}/{pdb_id}_bead.csv')
    df_beads = extract_beads(data_path)

    seq = df_beads['group_name'].values
    if len(seq[0]) == 1:
        seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values
    else:
        seq_id = df_beads['group_name'].apply(lambda x: vocab2[x]).values

    if mode == 'CA':
        coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    else:
        raise ValueError('mode should be CA / CB.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)
    profile = torch.tensor(seq_id, dtype=torch.long, device=device)

    return seq, coords, profile

