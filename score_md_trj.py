import numpy as np
import pandas as pd
import torch
from physics.protein_os import Protein
import options
import os
from tqdm import tqdm
import mdtraj as md
from utils import test_setup, load_protein_bead


#################################################
parser = options.get_decoy_parser()
args = options.parse_args_and_arch(parser)

device, model, energy_fn, ProteinBase = test_setup(args)

torch.set_grad_enabled(False)


#################################################
amino_acids = pd.read_csv('data/amino_acids.csv')
vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}


def get_cb_index(topology):
    ca_gly = topology.select('(name == CA) and (resname == GLY)')
    cb = topology.select('name == CB')
    beads_idx = np.append(ca_gly, cb)
    beads_idx = np.sort(beads_idx)
    print(beads_idx)
    return beads_idx


# md_data_list = ['BPTI', 'Fip35', 'val_deep']
md_data_list = ['val_deep']

if 'Fip35' in md_data_list:
    root_dir = '/home/hyang/bio/erf/data/decoys/msm'
    trj_dir1 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_1-protein/ww_1-protein/'
    trj_dir2 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_2-protein/ww_2-protein/'

    seq_native, coords_native, profile_native = load_protein_bead(f'{root_dir}/fip35_bead.csv', 'CB', device)
    protein_native = Protein(seq_native, coords_native, profile_native)
    energy_native = protein_native.get_energy(energy_fn).item()
    print('native', energy_native)

    for trj_dir in [trj_dir1, trj_dir2]:
        structure = md.load(f'{trj_dir}/ww-protein.pdb')
        top = structure.topology
        df = pd.read_csv(f'{trj_dir}/ww-protein-beads.csv')
        cb_idx = df['beads_cb_index'].values
        seq = df['group_name'].values
        seq_id = df['group_name'].apply(lambda x: vocab[x]).values
        profile = torch.tensor(seq_id, dtype=torch.long, device=device)

        score_list = []
        flist = pd.read_csv(f'{trj_dir}/flist.txt')['fname']
        for k, fname in enumerate(flist):
            trj = md.load(f'{trj_dir}/{fname}', top=top)
            coords_all = trj.xyz * 10
            coords_cb_all = coords_all[:, cb_idx, :]
            for i in tqdm(range(coords_cb_all.shape[0])):
                coords = torch.tensor(coords_cb_all[i], dtype=torch.float, device=device)
                protein = Protein(seq, coords, profile)
                energy = protein.get_energy(energy_fn).item()
                score_list.append(energy)
            df_i = pd.DataFrame({'energy': score_list})
            df_i.to_csv(f'{trj_dir}/energy_{k}.csv', index=False)
        df = pd.DataFrame({'energy': score_list})
        df.to_csv(f'{trj_dir}/energy.csv', index=False)

if 'BPTI' in md_data_list:
    root_dir = '/home/hyang/bio/erf/data/decoys/md'
    trj_dir = f'/home/hyang/bio/bg/bpti/DESRES-Trajectory-bpti-protein-short/bpti-protein-short/'

    seq_native, coords_native, profile_native = load_protein_bead(f'{root_dir}/1BPI_A_bead.csv', 'CB', device)
    protein_native = Protein(seq_native, coords_native, profile_native)
    energy_native = protein_native.get_energy(energy_fn).item()
    print('native', energy_native)

    structure = md.load(f'{trj_dir}/bpti.pdb')
    top = structure.topology
    cb_idx = get_cb_index(top)
    # stack all coordinates
    coords_cb_all = []
    flist = pd.read_csv(f'{trj_dir}/flist.txt')['fname']
    for k, fname in enumerate(flist):
        trj = md.load(f'{trj_dir}/{fname}', top=top)
        coords_k = trj.xyz * 10
        coords_cb_k = coords_k[:, cb_idx, :]
        coords_cb_all.append(coords_cb_k)
    coords_cb_all = np.vstack(coords_cb_all)
    # score all frames
    score_list = []
    for i in tqdm(range(coords_cb_all.shape[0])):
        coords = torch.tensor(coords_cb_all[i], dtype=torch.float, device=device)
        protein = Protein(seq_native, coords, profile_native)
        energy = protein.get_energy(energy_fn).item()
        score_list.append(energy)

    t = md.Trajectory(xyz=coords_cb_all, topology=None)
    t = t.superpose(t, frame=0)
    rmsd = md.rmsd(t, t, frame=0)
    rmsf = md.rmsf(t, t, frame=0)
    df = pd.DataFrame({'energy': score_list, 'rmsd': rmsd})
    df.to_csv(f'{root_dir}/BPTI/BPTI_energy_rmsd.csv', index=False)
    df = pd.DataFrame({'rmsf': rmsf})
    df.to_csv(f'{root_dir}/BPTI/BPTI_rmsf.csv', index=False)


if 'val_deep' in md_data_list:
    root_dir = '/home/hyang/bio/erf/data/decoys/md'
    trj_dir = f'/home/hyang/bio/openmm/data'

    pdb_id_list = pd.read_csv(f'{trj_dir}/list', header=None, names=['pdb'])['pdb'].values
    # pdb_id_list = ['3KXT']
    for pdb_id in tqdm(pdb_id_list):
        pdb_path = f'{root_dir}/{pdb_id}_A_bead.csv'
        seq_native, coords_native, profile_native = load_protein_bead(pdb_path,'CB', device)
        protein = Protein(seq_native, coords_native, profile_native)
        energy_native = protein.get_energy(energy_fn).item()
        print('native', energy_native)

        for flag in ['T300', 'T500']:
            trj = md.load(f'{trj_dir}/{pdb_id}/production2_{flag}.dcd',
                          top=f'{trj_dir}/{pdb_id}/production_{flag}.pdb')
            trj_pro = trj.remove_solvent()
            top = trj_pro.topology
            cb_idx = get_cb_index(top)
            coords_all = trj_pro.xyz * 10
            coords_cb_all = coords_all[:, cb_idx, :]
            # score all frames
            score_list = [energy_native]
            for i in tqdm(range(coords_cb_all.shape[0])):
                coords = torch.tensor(coords_cb_all[i], dtype=torch.float, device=device)
                protein = Protein(seq_native, coords, profile_native)
                energy = protein.get_energy(energy_fn).item()
                score_list.append(energy)

            t = md.Trajectory(xyz=coords_cb_all, topology=None)
            t = t.superpose(t, frame=0)
            rmsd = md.rmsd(t, t, frame=0)
            rmsd = np.append(np.array([0]), rmsd)
            df = pd.DataFrame({'energy': score_list, 'rmsd': rmsd})
            df.to_csv(f'{root_dir}/cullpdb_val_deep/{pdb_id}_{flag}_energy_rmsd.csv', index=False)










