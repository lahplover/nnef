import torch
import pandas as pd
import numpy as np
from physics.protein_os import Protein
import matplotlib.pyplot as pl
import mdtraj as md
import h5py
from tqdm import tqdm

"""
prepare the Normal Modes dataset.
"""


###################################################
def load_protein(data_path, pdb_id, device):
    # load coords as torch.double dtype
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x.upper(): y - 1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    print(pdb_id)
    df_beads = pd.read_csv(f'{data_path}/{pdb_id}_bead.csv')
    seq = df_beads['group_name'].values
    seq_id = df_beads['group_name'].apply(lambda x: vocab[x]).values

    coords = df_beads[['xcb', 'ycb', 'zcb']].values
    coords = torch.tensor(coords, dtype=torch.double, device=device)
    profile = torch.tensor(seq_id, dtype=torch.long, device=device)
    return seq, coords, profile


def excite_mode_cart(coords, dxyz, num=7, scale=1.0):
    coords_sample = []
    for i in range(num):
        coords_i = coords + dxyz * scale * (i - num // 2)
        coords_sample.append(coords_i)
    return coords_sample


def excite_mode_int(protein, dphi, num=7, scale=1.0):
    coords_sample = []
    coords_native = protein.coords
    r, theta, phi = protein.coords_int[:, 0], protein.coords_int[:, 1], protein.coords_int[:, 2]
    for i in range(num):
        phi_i = phi + dphi * scale * (i - num // 2)
        coords_int_i = torch.stack((r, theta, phi_i), dim=1)
        coords_i = protein.internal_to_cartesian(coords_native[0:3], coords_int_i)
        coords_sample.append(coords_i)
    return coords_sample


def mix_modes_cart(coords, modes_vec, num=7, scale=1.0):
    # coords (L, 3)
    # modes_vec is a tensors with N low freq modes. (N, L*3)
    # In each mode dimension, use num points.
    # generate a grid. in total we have num**N decoys.
    N, d = modes_vec.size()
    a = torch.arange(num, device=coords.device) - num // 2
    a_mesh = torch.meshgrid([a] * N)
    a_0 = a_mesh[0].flatten()[:, None]  # (num**N, 1)
    dxyz = a_0 * modes_vec[0]   # (num**N, L*3)
    for i in range(1, N):
        a_i = a_mesh[i].flatten()[:, None]
        dxyz = dxyz + a_i * modes_vec[i]
    dxyz = dxyz.reshape((-1, d // 3, 3))  # (num**N, L, 3)
    coords_all = coords[None, :, :] + dxyz * scale
    return coords_all


def mix_modes_int(protein, modes_vec, num=7, scale=1.0):
    # phi (L,)
    # modes_vec is a tensors with N low freq modes. (N, L)
    # In each mode dimension, use num points.
    # generate a grid. in total we have num**N decoys.
    coords_native = protein.coords
    r, theta, phi = protein.coords_int[:, 0], protein.coords_int[:, 1], protein.coords_int[:, 2]
    N, d = modes_vec.size()
    a = torch.arange(num, device=phi.device) - num // 2
    a_mesh = torch.meshgrid([a] * N)
    a_0 = a_mesh[0].flatten()[:, None]  # (num**N, 1)
    dphi = a_0 * modes_vec[0]   # (num**N, L)
    for i in range(1, N):
        a_i = a_mesh[i].flatten()[:, None]
        dphi = dphi + a_i * modes_vec[i]
    phi_all = phi[None, :] + dphi * scale

    coords_sample = []
    for i in tqdm(range(num**N)):
        phi_i = phi_all[i]
        coords_int_i = torch.stack((r, theta, phi_i), dim=1)
        coords_i = protein.internal_to_cartesian(coords_native[0:3], coords_int_i)
        coords_sample.append(coords_i)
    coords_all = torch.stack(coords_sample, 0)
    return coords_all


def align_rmsd(coords_mode, ref_frame=0):
    # align and compute RMSD,
    sample_xyz = torch.stack(coords_mode, 0).cpu().detach().numpy()
    t = md.Trajectory(xyz=sample_xyz, topology=None)
    t = t.superpose(t, frame=ref_frame)
    sample_rmsd = md.rmsd(t, t, frame=ref_frame)  # computation will change sample_xyz;
    return sample_rmsd, t.xyz


def write_pdb_sample(coords_sample, seq, pdb_id, data_path, nm='mode1'):
    with open(f'{data_path}/{pdb_id}_{nm}.pdb', 'wt') as mf:
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


def plot_hessian(hessian, pdb_id, data_path, flag='cart'):
    # plot hessian matrix
    hessian_np = hessian.detach().cpu().numpy()
    pl.figure()
    pl.imshow(hessian_np)
    pl.savefig(f'{data_path}/{pdb_id}_hessian_{flag}.pdf')
    pl.close()


def calc_modes_cart(pdb_id, protein, data_path='.', debug=False):
    # compute cartesian normal modes
    coords_native = protein.coords
    distmap_native = protein.get_distmap(coords_native)
    distmap_cutoff_mask = (distmap_native < 10)

    def energy_cart(coords):
        k = 1.0
        distmap = protein.get_distmap(coords)
        dist = (distmap[distmap_cutoff_mask] - distmap_native[distmap_cutoff_mask])**2
        energy = 0.5 * torch.sum(k * dist)
        return energy

    coords = coords_native.clone()
    coords.requires_grad_()
    hessian = torch.autograd.functional.hessian(energy_cart, coords)
    N, d, _, _ = hessian.size()
    hessian = hessian.reshape(N*d, N*d)
    if debug:
        plot_hessian(hessian, pdb_id, data_path, flag='cart')
    e, v = torch.symeig(hessian, eigenvectors=True)
    return e, v


def calc_modes_int(pdb_id, protein, data_path='.', debug=False):
    coords_native = protein.coords
    distmap_native = protein.get_distmap(coords_native)
    distmap_cutoff_mask = (distmap_native < 10)
    coords_int_native = protein.cartesian_to_internal(coords_native)
    protein.coords_int = coords_int_native
    r, theta, phi = coords_int_native[:, 0], coords_int_native[:, 1], coords_int_native[:, 2]

    def energy_int(phi):
        k = 1.0
        coords_int = torch.stack((r, theta, phi), dim=1)
        coords = protein.internal_to_cartesian(coords_native[0:3], coords_int)
        distmap = protein.get_distmap(coords)
        dist = (distmap[distmap_cutoff_mask] - distmap_native[distmap_cutoff_mask])**2
        energy = 0.5 * torch.sum(k * dist)
        return energy

    # compute torsional normal modes
    # theta.requires_grad_()
    phi.requires_grad_()
    hessian_int = torch.autograd.functional.hessian(energy_int, phi)
    if debug:
        plot_hessian(hessian_int, pdb_id, data_path, flag='int')
    e_int, v_int = torch.symeig(hessian_int, eigenvectors=True)
    return e_int, v_int


###################################################
def nma_test():
    # data_path = f'data/fold/cullpdb_val_deep'
    data_path = f'data/normal_modes'

    protein_sample = pd.read_csv(f'{data_path}/sample.csv')

    pdb_selected = protein_sample['pdb'].values

    for pdb_id in pdb_selected:
        seq, coords_native, profile = load_protein(data_path, pdb_id, device)

        protein = Protein(seq, coords_native.clone(), profile)
        e, v = calc_modes_cart(pdb_id, protein, data_path)

        protein = Protein(seq, coords_native.clone(), profile)
        e_int, v_int = calc_modes_int(pdb_id, protein, data_path)

        # make movies of the normal modes in cartesian space
        num = 7
        n_modes = 6
        for i in range(n_modes):
            mode1 = v[:, 6+i]
            # mode1 = v[:, -i]
            dxyz = mode1.reshape((-1, 3))
            dxyz = 0.5 * dxyz * np.sqrt(dxyz.shape[0])  # make the norm of each mode to 0.5 A
            coords_mode = excite_mode_cart(coords_native, dxyz, num=num, scale=1)
            write_pdb_sample(coords_mode, seq, pdb_id, data_path, nm=f'mode{i}')
            print((torch.sum(dxyz**2)/dxyz.shape[0])**0.5)

            sample_rmsd, coords_align = align_rmsd(coords_mode, num // 2)
            print(sample_rmsd)
            write_pdb_sample(coords_align, seq, pdb_id, data_path, nm=f'mode{i}_align')

        # make movies of the normal modes in torsional space
        num = 7
        n_modes = 6
        for i in range(n_modes):
            dphi = v_int[:, 6+i]
            # dphi = v_int[:, -i]
            coords_mode = excite_mode_int(protein, dphi, num=num, scale=0.5)
            write_pdb_sample(coords_mode, seq, pdb_id, data_path, nm=f'mode_int{i}')

            sample_rmsd, coords_align = align_rmsd(coords_mode, num // 2)
            print(sample_rmsd)
            write_pdb_sample(coords_align, seq, pdb_id, data_path, nm=f'mode_int{i}_align')

        # make decoys by linear combinations of low freq modes in cartesian space
        num = 7
        n_modes = 4
        modes_vec = v[:, 6:6+n_modes].transpose(0, 1)
        coords_all = mix_modes_cart(coords_native, modes_vec, num=num, scale=4.0)
        with h5py.File(f'{data_path}/{pdb_id}_decoys_cart.h5', 'w') as f:
            dset = f.create_dataset("coords", shape=coords_all.shape, data=coords_all.detach().cpu().numpy(), dtype='f4')
        # write_pdb_sample(coords_all, seq, pdb_id, data_path, nm=f'mode_mix')

        # make decoys by linear combinations of low freq modes in torsional space
        num = 7
        n_modes = 4
        modes_vec = v_int[:, 6:6+n_modes].transpose(0, 1)
        coords_all = mix_modes_int(protein, modes_vec, num=num, scale=0.5)
        with h5py.File(f'{data_path}/{pdb_id}_decoys_int.h5', 'w') as f:
            dset = f.create_dataset("coords", shape=coords_all.shape, data=coords_all.detach().cpu().numpy(), dtype='f4')
        # write_pdb_sample(coords_all, seq, pdb_id, data_path, nm=f'mode_int_mix')

        # project the vector native-unfold onto normal modes vectors
        # num = 21
        # for i in range(v_int.shape[1]):
        #     dphi = v_int[:, i]
        #     dphi_norm = dphi / torch.norm(dphi)
        #     print(torch.dot(v_int[:, 9], dphi))
        #     a = torch.dot(phi, dphi)
        #     if a > 5.0:
        #         print(i)
        #         coords_mode = excite_mode_int(protein, dphi, num=num, scale=0.5)
        #         write_pdb_sample(coords_mode, seq, pdb_id, data_path, nm=f'mode_int{i}')
        #
        #         sample_rmsd, coords_align = align_rmsd(coords_mode, num // 2)
        #         print(sample_rmsd)
        #         write_pdb_sample(coords_align, seq, pdb_id, data_path, nm=f'mode_int{i}_align')


def nma_funnel():
    data_path = f'../hhsuite/hhsuite_beads/hhsuite/'
    protein_sample = pd.read_csv(f'data/hhsuite_CB_cullpdb_cath_funnel.csv')
    # data_path = f'data/fold/cullpdb_val_deep'
    # protein_sample = pd.read_csv(f'{data_path}/sample.csv')
    pdb_selected = protein_sample['pdb'].values

    hh_data_coords = h5py.File('data/hhsuite_CB_cullpdb_cath_funnel.h5', 'r', libver='latest', swmr=True)
    coords_dict = {}
    for pdb in tqdm(pdb_selected):
        coords_dict[pdb] = hh_data_coords[pdb][()]

    n_modes = 10
    with h5py.File('data/hhsuite_CB_cullpdb_cath_funnel_normal_modes.h5', 'w') as f:
        for pdb_id in tqdm(pdb_selected):
            coords_native = torch.tensor(coords_dict[pdb_id], dtype=torch.float, device=device)
            # seq, coords_native, profile = load_protein(data_path, pdb_id, device)
            protein = Protein(None, coords_native, None)
            e, v = calc_modes_cart(pdb_id, protein, data_path)
            modes_vec = v[:, 6:6+n_modes].transpose(0, 1).cpu().numpy().reshape(n_modes, -1, 3)
            dset = f.create_dataset(f'{pdb_id}', shape=modes_vec.shape, data=modes_vec, dtype='f4')


if __name__ == '__main__':
    args = None
    device = torch.device('cpu')

    # nma_funnel()
    nma_test()



