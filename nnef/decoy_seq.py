import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import options
from physics.protein_os import Protein
from utils import test_setup

parser = options.get_decoy_parser()
args = options.parse_args_and_arch(parser)

#################################################
# device = torch.device(args.device)
#
# model = LocalTransformer(args)
# model.load_state_dict(torch.load(f'{args.load_exp}/models/model.pt', map_location=torch.device('cpu')))
# model.to(device)
# model.eval()
#
# energy_fn = EnergyFun(model, args)
#
# ProteinBase.k = args.seq_len - 4

device, model, energy_fn, ProteinBase = test_setup(args)

torch.set_grad_enabled(False)


#################################################
def load_protein(pdb_id, mode, device, args):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    df_beads = pd.read_csv(f'../hhsuite/hhsuite_beads/hhsuite/{pdb_id}_bead.csv')

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


decoy_flag = args.decoy_set
decoy_loss_dir = args.decoy_loss_dir

if not os.path.exists(f'data/decoys/decoys_seq/{decoy_loss_dir}'):
    os.system(f'mkdir -p data/decoys/decoys_seq/{decoy_loss_dir}')

protein_sample = pd.read_csv('data/decoys/decoys_seq/hhsuite_CB_cullpdb_val_no_missing_residue_sample.csv')
pdb_list = protein_sample['pdb'].values

amino_acids = pd.read_csv('data/amino_acids.csv')
vocab = {x: y - 1 for x, y in zip(amino_acids.AA, amino_acids.idx)}

loss_native = []
for pdb_id in tqdm(pdb_list):
    seq, coords, profile_native = load_protein(pdb_id, args.mode, device, args)
    protein_native = Protein(seq, coords, profile_native)
    energy_native = protein_native.get_energy(energy_fn).item()
    loss_native.append(energy_native)

    if decoy_flag == 'polyAA':
        fake_seq = []
        for i in range(20):
            fake_seq.append(amino_acids.AA.values[i] * len(seq))
    else:
        fake_seq = pd.read_csv(f'data/decoys/decoys_seq/{pdb_id}_seq_{decoy_flag}.csv')['seq'].values

    loss_all = []
    if decoy_flag == 'type2LD':
        fake_seq = fake_seq[0:1]
    for s in fake_seq:
        seq_id = [vocab[x] for x in s]
        profile = torch.tensor(seq_id, dtype=torch.long, device=device)
        protein = Protein(seq, coords, profile)
        energy = protein.get_energy(energy_fn).item()
        loss_all.append(energy)

    loss_all = np.array(loss_all)
    print(pdb_id, energy_native, loss_all.mean(), loss_all.var())
    if decoy_flag == 'polyAA':
        df = pd.DataFrame({'AA': amino_acids.AA, 'loss': loss_all})
    else:
        df = pd.DataFrame({'loss': loss_all})
    df.to_csv(f'data/decoys/decoys_seq/{decoy_loss_dir}/{pdb_id}_{decoy_flag}_loss.csv', index=False)

df = pd.DataFrame({'pdb': pdb_list, 'loss_native': loss_native})
df.to_csv(f'data/decoys/decoys_seq/{decoy_loss_dir}/hhsuite_CB_cullpdb_val_no_missing_residue_sample_loss.csv', index=False)

