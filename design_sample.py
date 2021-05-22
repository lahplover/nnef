import numpy as np
import pandas as pd
import torch
from physics.protein_os import Protein
import options
from physics.anneal import AnnealSeq
import os
import h5py
from tqdm import tqdm
from utils import test_setup


"""
do mutations & design for a sample of protein backbones. 
"""

#################################################
parser = options.get_fold_parser()
args = options.parse_args_and_arch(parser)

device, model, energy_fn, ProteinBase = test_setup(args)
torch.set_grad_enabled(False)


#################################################
def load_protein(root_dir, pdb_id, mode, device, args):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    df_beads = pd.read_csv(f'{root_dir}/{pdb_id}_bead.csv')

    seq = df_beads['group_name'].values
    seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values

    if mode == 'CA':
        coords = df_beads[['xca', 'yca', 'zca']].values
    elif mode == 'CB':
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
    else:
        raise ValueError('mode should be CA / CB.')

    coords = torch.tensor(coords, dtype=torch.float, device=device)
    profile = torch.tensor(seq_id, dtype=torch.long, device=device)

    return seq, coords, profile


# root_dir = 'data/design/cullpdb_val_sample'
root_dir = 'data/design/cullpdb_val_deep'
# root_dir = 'data/design/ref'
protein_sample = pd.read_csv(f'{root_dir}/sample.csv')
pdb_selected = protein_sample['pdb'].values
np.random.shuffle(pdb_selected)

design_engine = args.fold_engine
mode = args.mode
exp_id = args.load_exp[-5:]
save_dir = args.save_dir
if not os.path.exists(f'{root_dir}/{save_dir}'):
    os.mkdir(f'{root_dir}/{save_dir}')

for pdb_id in tqdm(pdb_selected):
    if os.path.exists(f'{root_dir}/{save_dir}/{pdb_id}_profile.h5'):
        continue

    seq, coords_native, profile = load_protein(root_dir, pdb_id, mode, device, args)

    # skip long sequences
    # if len(seq) > 400:
    #     continue

    protein_native = Protein(seq, coords_native, profile)
    energy_native = protein_native.get_energy(energy_fn).item()
    print('energy_native:', energy_native)
    residue_energy = protein_native.get_residue_energy(energy_fn)
    print(profile)
    print(residue_energy)

    protein = Protein(seq, coords_native.clone(), profile.clone())
    if args.random_init:
        protein.profile = torch.randint(0, 20, profile.size(), device=profile.device)

    energy_init = protein.get_energy(energy_fn).item()
    print('energy_init:', energy_init)

    if design_engine != 'mutation':
        # simulated annealing
        torch.set_grad_enabled(False)
        anneal_steps = int(args.L * (seq.shape[0] / 50.0))
        annealer = AnnealSeq(energy_fn, protein, seq_move_type=args.seq_move_type,
                             T_max=args.T_max, T_min=args.T_min, L=anneal_steps)
        annealer.run()
        profile_best = annealer.x_best
        energy_best = annealer.energy_best
        sample = annealer.sample
        sample_energy = annealer.sample_energy
        profile_native = protein_native.profile
        recovery = (profile_native.cpu().numpy() == profile_best.cpu().numpy())
        print(pdb_id, recovery.sum(), float(recovery.sum()) / protein.profile.size(0))

        # save sampled structures
        sample_profile = [profile_native.cpu(), profile_best.cpu()] + sample
        sample_profile = torch.stack(sample_profile, dim=0).numpy()
        with h5py.File(f'{root_dir}/{save_dir}/{pdb_id}_profile.h5', 'w') as f:
            profile_dtype = 'f4' if args.seq_type == 'profile' else 'i1'
            dset = f.create_dataset("profile", shape=sample_profile.shape, data=sample_profile, dtype=profile_dtype)

        sample_energy = [energy_native, energy_best] + sample_energy
        pd.DataFrame({'sample_energy': sample_energy}).to_csv(f'{root_dir}/{save_dir}/{pdb_id}_energy.csv', index=False)
    else:
        torch.set_grad_enabled(False)
        n = protein.profile.size(0)
        sample_energy = np.zeros((n, 20))
        for i in tqdm(range(n)):
            res_i = protein.profile[i].item()
            for j in range(20):
                protein.profile[i] = j
                sample_energy[i, j] = protein.get_energy(energy_fn).item()
            protein.profile[i] = res_i
        # energy_best = sample_energy.min()
        assert(torch.sum(protein_native.profile == protein.profile) == n)

        profile = protein.profile.cpu().numpy()
        profile_min = np.argmin(sample_energy, axis=1)
        recovery = (profile == profile_min)
        print(pdb_id, recovery.sum(), float(recovery.sum()) / n, recovery)

        # df_profile = pd.read_csv(f'{root_dir}/{pdb_id}_profile.csv')
        # f_profile = df_profile[[f'aa{i}' for i in range(20)]].values

        # kT = 0.01
        # energy_min = np.min(sample_energy, axis=1)
        # delta_energy = sample_energy - energy_min[:, None]
        # p = np.exp(-delta_energy / kT)
        # weighted_p = np.sum(p * f_profile, axis=1) / np.sum(p, axis=1)
        # weighted_recovery = weighted_p.mean()
        # print(np.sum(p, axis=1), weighted_recovery)

        with h5py.File(f'{root_dir}/{save_dir}/{pdb_id}_profile.h5', 'w') as f:
            dset = f.create_dataset("wt_residue_energy", shape=residue_energy.shape, data=residue_energy, dtype='f4')
            dset = f.create_dataset("mutant_energy", shape=sample_energy.shape, data=sample_energy, dtype='f4')
            dset = f.create_dataset("seq", shape=profile.shape, data=profile, dtype='f4')
            # dset = f.create_dataset("profile", shape=f_profile.shape, data=f_profile, dtype='f4')






