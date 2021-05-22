import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import options
from physics.protein_os import Protein
from physics.grad_minimizer import GradMinimizerCartesian
from utils import test_setup
import h5py
import mdtraj as md
import matplotlib.pyplot as pl


parser = options.get_decoy_parser()
args = options.parse_args_and_arch(parser)

#################################################
device, model, energy_fn, ProteinBase = test_setup(args)
if not args.relax:
    torch.set_grad_enabled(False)


#################################################
def load_protein(data_path, pdb_id, device):
    # load coords as torch.double dtype
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    print(pdb_id)
    df_beads = pd.read_csv(f'{data_path}/{pdb_id}_bead.csv')
    seq = df_beads['group_name'].values
    seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values

    coords = df_beads[['xcb', 'ycb', 'zcb']].values
    coords = torch.tensor(coords, dtype=torch.float, device=device)
    profile = torch.tensor(seq_id, dtype=torch.long, device=device)
    return seq, coords, profile


decoy_set = args.decoy_set
decoy_loss_dir = args.decoy_loss_dir
root_dir = f'./data/fold/cullpdb_val_deep/'

if not os.path.exists(f'{root_dir}/{decoy_loss_dir}'):
    os.system(f'mkdir -p {root_dir}/{decoy_loss_dir}')

pdb_selected = pd.read_csv(f'{root_dir}/sample.csv')['pdb'].values

for pdb_id in tqdm(pdb_selected):
    if os.path.exists(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_{decoy_set}_loss.csv'):
        continue
    seq, coords_native, profile = load_protein(root_dir, pdb_id, device)
    protein = Protein(seq, coords_native, profile)
    energy_native = protein.get_energy(energy_fn).item()

    # load decoy coordinates
    decoy_file = h5py.File(f'{root_dir}/{pdb_id}_decoys_{decoy_set}.h5', 'r')
    coords_decoy = decoy_file['coords'][()]
    coords_decoy = torch.tensor(coords_decoy, dtype=torch.float, device=device)

    loss_all = [energy_native]
    loss_relax = [energy_native]
    for i in tqdm(range(coords_decoy.shape[0])):
        protein.coords = coords_decoy[i]
        energy = protein.get_energy(energy_fn).item()
        loss_all.append(energy)

        if args.relax:
            minimizer = GradMinimizerCartesian(energy_fn, protein, num_steps=args.relax_steps)
            minimizer.run()
            energy_relax = minimizer.energy_best
            print('energy relaxed:', energy_relax)
            loss_relax.append(energy_relax)
        else:
            loss_relax.append(0)

    # compute rmsd
    coords_decoy = torch.cat((coords_native[None, :, :], coords_decoy), dim=0).detach().cpu().numpy()
    t = md.Trajectory(xyz=coords_decoy, topology=None)
    t = t.superpose(t, frame=0)
    sample_rmsd = md.rmsd(t, t, frame=0)

    df = pd.DataFrame({'loss': loss_all, 'loss_relax': loss_relax,
                       'rmsd': sample_rmsd})
    df.to_csv(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_{decoy_set}_loss.csv', index=False, float_format='%.4f')


for pdb_id in tqdm(pdb_selected):
    df = pd.read_csv(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_{decoy_set}_loss.csv')
    fig = pl.figure()
    pl.plot(df['rmsd'], df['loss'], 'b.')
    pl.xlabel(r'RMSD [$\AA$]')
    pl.ylabel('energy score')
    pl.title(f'{pdb_id} {decoy_set}')
    pl.savefig(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_{decoy_set}_loss.pdf')
    pl.close(fig)



